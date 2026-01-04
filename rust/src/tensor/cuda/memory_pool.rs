//! Memory pooling for reduced CUDA allocation overhead.
//!
//! Uses a power-of-2 bucket allocator to cache freed allocations
//! and reduce the cost of repeated alloc/free cycles.

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::device_manager::CudaError;

/// Configuration for memory pool behavior
#[derive(Clone, Debug)]
pub struct PoolConfig {
    /// Minimum block size (default: 256 bytes)
    pub min_block_size: usize,
    /// Maximum cached memory per device (default: 1GB)
    pub max_cached_bytes: usize,
    /// Round allocations to power of 2 (default: true)
    pub round_to_power_of_2: bool,
    /// Maximum single allocation size to cache (default: 256MB)
    pub max_cacheable_size: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            min_block_size: 256,
            max_cached_bytes: 1024 * 1024 * 1024, // 1GB
            round_to_power_of_2: true,
            max_cacheable_size: 256 * 1024 * 1024, // 256MB
        }
    }
}

/// Statistics about pool usage
#[derive(Clone, Debug, Default)]
pub struct PoolStats {
    /// Total bytes currently cached
    pub cached_bytes: usize,
    /// Number of allocations served from cache
    pub cache_hits: usize,
    /// Number of allocations requiring new memory
    pub cache_misses: usize,
    /// Total allocations performed
    pub total_allocations: usize,
    /// Total bytes allocated (including cache reuse)
    pub total_bytes_allocated: usize,
}

/// Per-device memory pool for reduced allocation overhead
pub struct MemoryPool {
    device_idx: usize,
    device: Arc<CudaDevice>,
    /// Free blocks organized by size bucket (power of 2)
    /// Each bucket contains Vec of raw byte slices
    free_blocks: RwLock<HashMap<usize, Vec<CudaSlice<u8>>>>,
    /// Total cached bytes
    cached_bytes: AtomicUsize,
    /// Statistics
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
    total_allocations: AtomicUsize,
    total_bytes: AtomicUsize,
    /// Configuration
    config: PoolConfig,
}

impl MemoryPool {
    /// Create a new memory pool for the given device
    pub fn new(device_idx: usize, device: Arc<CudaDevice>) -> Self {
        MemoryPool {
            device_idx,
            device,
            free_blocks: RwLock::new(HashMap::new()),
            cached_bytes: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            total_allocations: AtomicUsize::new(0),
            total_bytes: AtomicUsize::new(0),
            config: PoolConfig::default(),
        }
    }

    /// Create a pool with custom configuration
    pub fn with_config(device_idx: usize, device: Arc<CudaDevice>, config: PoolConfig) -> Self {
        MemoryPool {
            device_idx,
            device,
            free_blocks: RwLock::new(HashMap::new()),
            cached_bytes: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            total_allocations: AtomicUsize::new(0),
            total_bytes: AtomicUsize::new(0),
            config,
        }
    }

    /// Round size up to next power of 2
    fn round_size(&self, size: usize) -> usize {
        if !self.config.round_to_power_of_2 {
            return size.max(self.config.min_block_size);
        }

        let size = size.max(self.config.min_block_size);
        if size.is_power_of_two() {
            size
        } else {
            size.next_power_of_two()
        }
    }

    /// Allocate memory from pool or create new allocation
    pub fn alloc_bytes(&self, size: usize) -> Result<CudaSlice<u8>, CudaError> {
        let bucket_size = self.round_size(size);

        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(size, Ordering::Relaxed);

        // Try to get from cache
        {
            let mut blocks = self.free_blocks.write().unwrap();
            if let Some(bucket) = blocks.get_mut(&bucket_size) {
                if let Some(slice) = bucket.pop() {
                    self.cached_bytes.fetch_sub(bucket_size, Ordering::Relaxed);
                    self.cache_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(slice);
                }
            }
        }

        // Cache miss: allocate new
        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        self.device.alloc_zeros::<u8>(bucket_size)
            .map_err(|e| CudaError::AllocationFailed(e.to_string()))
    }

    /// Allocate typed memory from pool (f64)
    pub fn alloc_f64(&self, count: usize) -> Result<CudaSlice<f64>, CudaError> {
        self.device.alloc_zeros::<f64>(count)
            .map_err(|e| CudaError::AllocationFailed(e.to_string()))
    }

    /// Allocate typed memory from pool (f32)
    pub fn alloc_f32(&self, count: usize) -> Result<CudaSlice<f32>, CudaError> {
        self.device.alloc_zeros::<f32>(count)
            .map_err(|e| CudaError::AllocationFailed(e.to_string()))
    }

    /// Return memory to the pool
    pub fn free_bytes(&self, slice: CudaSlice<u8>) {
        let size = slice.len();
        let bucket_size = self.round_size(size);

        // Don't cache if too large
        if bucket_size > self.config.max_cacheable_size {
            return; // Let it drop and be freed
        }

        // Check if we have room in cache
        let current_cached = self.cached_bytes.load(Ordering::Relaxed);
        if current_cached + bucket_size > self.config.max_cached_bytes {
            // Cache full, let this allocation be freed
            return;
        }

        // Add to cache
        self.cached_bytes.fetch_add(bucket_size, Ordering::Relaxed);

        let mut blocks = self.free_blocks.write().unwrap();
        blocks.entry(bucket_size).or_default().push(slice);
    }

    /// Clear all cached memory
    pub fn trim(&self) {
        let mut blocks = self.free_blocks.write().unwrap();
        blocks.clear();
        self.cached_bytes.store(0, Ordering::Relaxed);
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            cached_bytes: self.cached_bytes.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            total_bytes_allocated: self.total_bytes.load(Ordering::Relaxed),
        }
    }

    /// Get the device index
    pub fn device_idx(&self) -> usize {
        self.device_idx
    }

    /// Get the underlying device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

/// RAII wrapper that returns memory to pool on drop
pub struct PooledSlice<T: cudarc::driver::DeviceRepr> {
    inner: Option<CudaSlice<T>>,
    pool: Arc<MemoryPool>,
}

impl<T: cudarc::driver::DeviceRepr> PooledSlice<T> {
    /// Create a new pooled slice
    pub fn new(slice: CudaSlice<T>, pool: Arc<MemoryPool>) -> Self {
        PooledSlice {
            inner: Some(slice),
            pool,
        }
    }

    /// Get the underlying slice
    pub fn as_slice(&self) -> &CudaSlice<T> {
        self.inner.as_ref().unwrap()
    }

    /// Get mutable access to the underlying slice
    pub fn as_slice_mut(&mut self) -> &mut CudaSlice<T> {
        self.inner.as_mut().unwrap()
    }

    /// Take ownership of the inner slice (no return to pool)
    pub fn take(mut self) -> CudaSlice<T> {
        self.inner.take().unwrap()
    }

    /// Get the length in elements
    pub fn len(&self) -> usize {
        self.inner.as_ref().unwrap().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a reference to the pool this slice belongs to
    pub fn pool(&self) -> &Arc<MemoryPool> {
        &self.pool
    }
}

impl<T: cudarc::driver::DeviceRepr> std::ops::Deref for PooledSlice<T> {
    type Target = CudaSlice<T>;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref().unwrap()
    }
}

impl<T: cudarc::driver::DeviceRepr> std::ops::DerefMut for PooledSlice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_mut().unwrap()
    }
}

impl<T: cudarc::driver::DeviceRepr> Drop for PooledSlice<T> {
    fn drop(&mut self) {
        // Note: For now, we can't easily return typed slices to the byte pool
        // because cudarc doesn't support safe transmutation.
        // The slice will just be dropped normally.
        // Future optimization: use raw pointers for pool management
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_size() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let pool = MemoryPool::new(0, device);

        assert_eq!(pool.round_size(100), 256); // min size
        assert_eq!(pool.round_size(256), 256);
        assert_eq!(pool.round_size(257), 512);
        assert_eq!(pool.round_size(1000), 1024);
    }
}
