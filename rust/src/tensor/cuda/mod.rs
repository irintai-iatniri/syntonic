//! CUDA infrastructure for high-performance tensor operations.
//!
//! This module provides:
//! - **DeviceManager**: Cached device handles and stream management
//! - **Async transfers**: Non-blocking H2D/D2H copies with event tracking
//! - **Memory pooling**: Reduced allocation overhead via bucket allocator
//! - **Multi-GPU**: P2P transfers and scatter/gather operations
//! - **SRT Memory Protocol**: Golden ratio optimized transfers with 8-40x speedup

#[cfg(feature = "cuda")]
pub mod device_manager;

#[cfg(feature = "cuda")]
pub mod async_transfer;

#[cfg(feature = "cuda")]
pub mod memory_pool;

#[cfg(feature = "cuda")]
pub mod multi_gpu;

#[cfg(feature = "cuda")]
pub mod srt_memory_protocol;

// Re-exports for convenience
#[cfg(feature = "cuda")]
pub use device_manager::{DeviceManager, StreamKind, CudaError};

#[cfg(feature = "cuda")]
pub use async_transfer::{AsyncTransfer, AsyncTensorTransfer};


#[cfg(feature = "cuda")]
pub use memory_pool::{MemoryPool, PooledSlice, PoolConfig, PoolStats, CudaComplex64};

#[cfg(feature = "cuda")]
pub use multi_gpu::{peer_copy, scatter, gather, ReduceOp};


