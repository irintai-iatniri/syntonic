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
pub use device_manager::{CudaError, DeviceManager, StreamKind};

#[cfg(feature = "cuda")]
pub use async_transfer::{AsyncTensorTransfer, AsyncTransfer, TransferComputeOverlap};

#[cfg(feature = "cuda")]
pub use memory_pool::{CudaComplex64, MemoryPool, PoolConfig, PoolStats, PooledSlice};

#[cfg(feature = "cuda")]
pub use multi_gpu::{gather, peer_copy, scatter, ReduceOp};
