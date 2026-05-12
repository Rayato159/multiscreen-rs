use crate::error::{Error, Result};
use burn::{
    backend::{Autodiff, Flex},
    tensor::backend::{Backend, BackendTypes},
};
use std::env;

/// Default Candle-free CPU backend used by this crate.
pub type DefaultBackend = Flex;

/// Default training backend. Burn's `Autodiff` decorator adds backpropagation.
pub type DefaultAutodiffBackend = Autodiff<DefaultBackend>;

/// Device type for the default Burn backend.
pub type Device = <DefaultAutodiffBackend as BackendTypes>::Device;

/// Chooses the default Burn device from `MULTISCREEN_DEVICE`.
///
/// Values: `auto`, `cpu`, `flex`.
pub fn default_device() -> Result<Device> {
    let requested = env::var("MULTISCREEN_DEVICE").unwrap_or_else(|_| "auto".to_string());
    match requested.trim().to_ascii_lowercase().as_str() {
        "" | "auto" | "cpu" | "flex" => Ok(Device::default()),
        "cuda" | "gpu" => Err(Error::Config(
            "Burn CUDA is backend-generic; instantiate MultiscreenModel with a CUDA Burn backend \
             instead of using the default Flex device"
                .to_string(),
        )),
        other => Err(Error::Config(format!(
            "unsupported MULTISCREEN_DEVICE={other:?}; use auto, cpu, or flex"
        ))),
    }
}

/// Human-readable label for the default Burn device.
pub fn device_label(device: &Device) -> String {
    <DefaultAutodiffBackend as Backend>::name(device)
}

// ---------------------------------------------------------------------------
// CUDA support (enabled by the `cuda` feature flag)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub use burn::backend::Cuda;

/// Autodiff-wrapped CUDA backend for training on GPU.
#[cfg(feature = "cuda")]
pub type CudaAutodiffBackend = Autodiff<Cuda>;

/// Device type for the CUDA backend.
#[cfg(feature = "cuda")]
pub type CudaDevice = <CudaAutodiffBackend as BackendTypes>::Device;

/// Convenience alias for a Multiscreen model backed by CUDA.
#[cfg(feature = "cuda")]
pub type CudaMultiscreenModel = crate::model::MultiscreenModel<CudaAutodiffBackend>;

/// Returns the default CUDA device (GPU index 0).
///
/// Enable the `cuda` feature flag and make sure the Burn CUDA runtime
/// and compatible NVIDIA driver are available on the system.
///
/// ```toml
/// [dependencies]
/// multiscreen-rs = { version = "0.1", features = ["cuda"] }
/// ```
#[cfg(feature = "cuda")]
pub fn cuda_device() -> Result<CudaDevice> {
    Ok(CudaDevice::default())
}
