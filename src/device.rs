//! Device abstraction for CPU and CUDA.

#[cfg(not(feature = "cuda"))]
use crate::error::Error;
use crate::error::Result;
use crate::runtime::Device;

/// Returns the default CPU device.
///
/// ```rust,no_run
/// use multiscreen_rs::prelude::*;
/// fn main() -> multiscreen_rs::Result<()> {
///     let device = cpu()?;
///     Ok(())
/// }
/// ```
pub fn cpu() -> Result<Device> {
    Ok(Device::default())
}

/// Returns a CUDA device for the given GPU index.
///
/// Only available with the `cuda` feature. Returns a clear error if CUDA
/// is not available.
///
/// ```toml
/// [dependencies]
/// multiscreen-rs = { version = "0.1", features = ["cuda"] }
/// ```
#[cfg(feature = "cuda")]
pub fn cuda(index: usize) -> Result<crate::runtime::CudaDevice> {
    // For now, Burn's Cuda device doesn't support selecting GPU index via
    // the simple API, so we just return the default CUDA device.
    // The index parameter is reserved for future multi-GPU support.
    let _ = index;
    Ok(crate::runtime::CudaDevice::default())
}

/// Returns a CUDA device for the given GPU index.
///
/// Without the `cuda` feature, this always returns an error.
#[cfg(not(feature = "cuda"))]
pub fn cuda(_index: usize) -> Result<Device> {
    Err(Error::Config(
        "CUDA is not available. Enable the 'cuda' feature in Cargo.toml:\n\
         [dependencies]\n\
         multiscreen-rs = { version = \"0.1\", features = [\"cuda\"] }"
            .to_string(),
    ))
}

/// Returns the best available device.
///
/// Always returns the default Flex (CPU) device. To use CUDA, construct
/// a [`MultiscreenModel`](crate::MultiscreenModel) with
/// [`CudaAutodiffBackend`](crate::runtime::CudaAutodiffBackend) directly.
pub fn auto_device() -> Result<Device> {
    cpu()
}
