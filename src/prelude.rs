// ---- High-level API (recommended for most users) ----
pub use crate::{
    auto_device, cuda, ChatModel, GenerationConfig, ParameterBudget, Trainer, TrainingReport,
};

#[cfg(not(feature = "cuda"))]
pub use crate::cpu;

// ---- Core types ----
pub use crate::{
    device_label, DefaultAutodiffBackend, DefaultBackend, DefaultMultiscreenModel, Device, Error,
    Result,
};

#[cfg(not(feature = "cuda"))]
pub use crate::default_device;

// ---- Model configuration ----
pub use crate::{
    cross_entropy_loss_with_mask, EvaluationResult, ModelInferenceConfig, ModelTrainingConfig,
    ModelTrainingReport, MultiscreenModel, MultiscreenModelConfig, MultiscreenModelOutput,
    MultiscreenParameterBudget,
};

// ---- Engine (lightweight transition engine) ----
pub use crate::{InferenceOutput, MultiscreenEngine, TrainInput, TrainReport};

// ---- Layout utilities ----
pub use crate::{
    causal_softmask, causal_trim_relevance, trim_and_square, InferenceConfig, Int,
    MultiscreenConfig, Screen, ScreenConfig, ScreenLayout, ScreeningGridConfig, Tensor, TensorData,
    TileConfig, TokenSpan, TrimConfig,
};

#[cfg(feature = "cuda")]
pub use crate::{CudaAutodiffBackend, CudaDevice, CudaMultiscreenModel};

pub use crate::AutodiffBackend;
pub use crate::Backend;
