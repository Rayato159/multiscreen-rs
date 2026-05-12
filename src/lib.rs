//! # multiscreen-rs
//!
//! > A Rust implementation of the Multiscreen neural language model — training
//! > and inference — powered by [Burn](https://github.com/tracel-ai/burn).
//!
//! ## Quick Start — Training
//!
//! ```rust,no_run
//! use multiscreen_rs::prelude::*;
//!
//! fn main() -> multiscreen_rs::Result<()> {
//!     let mut trainer = Trainer::builder()
//!         .vocab_size(1000)
//!         .budget(ParameterBudget::Params10M)
//!         .device(auto_device()?)
//!         .batch_size(16)
//!         .seq_len(128)
//!         .steps(50_000)
//!         .build()?;
//!
//!     let sequences = vec![vec![1, 2, 3, 4], vec![1, 2, 5, 4]];
//!     let report = trainer.train_on_token_sequences(&sequences)?;
//!     println!("trained {} steps, final loss {:.4}", report.steps, report.final_loss);
//!     Ok(())
//! }
//! ```
//!
//! ## Quick Start — Chat / Inference
//!
//! ### Non-streaming (all tokens at once)
//!
//! ```rust,no_run
//! use multiscreen_rs::prelude::*;
//!
//! fn main() -> multiscreen_rs::Result<()> {
//!     let model = ChatModel::load("checkpoints/latest.mpk")?;
//!     let token_ids = model.generate(&[1, 2, 3], GenerationConfig::default())?;
//!     println!("generated tokens: {:?}", token_ids);
//!     Ok(())
//! }
//! ```
//!
//! ### Streaming (token by token, like ChatGPT)
//!
//! ```rust,no_run
//! use multiscreen_rs::prelude::*;
//!
//! fn main() -> multiscreen_rs::Result<()> {
//!     let model = ChatModel::load("checkpoints/latest.mpk")?;
//!     let full = model.generate_stream(
//!         &[1, 2, 3],
//!         GenerationConfig::default(),
//!         |token_id, _index| {
//!             // Decode with YOUR tokenizer and print word-by-word
//!             print!("{} ", token_id);
//!             true // return false to stop early
//!         },
//!     )?;
//!     Ok(())
//! }
//! ```
//!
//! ## Device Selection
//!
//! ```rust,no_run
//! use multiscreen_rs::prelude::*;
//!
//! fn main() -> multiscreen_rs::Result<()> {
//!     let device = auto_device()?;  // best available (CPU or CUDA)
//!     // let device = cuda(0)?;       // CUDA GPU (requires "cuda" feature)
//!     Ok(())
//! }
//! ```
//!
//! ## Low-Level In-Memory Quick Start
//!
//! ```rust
//! use multiscreen_rs::prelude::*;
//!
//! fn main() -> multiscreen_rs::Result<()> {
//!     let device = auto_device()?;
//!     let mut model = DefaultMultiscreenModel::new(
//!         MultiscreenModelConfig::tiny_for_tests(),
//!         &device,
//!     )?;
//!
//!     model.train_token_sequences(
//!         &[vec![1, 2, 3, 4], vec![1, 2, 5, 4]],
//!         &ModelTrainingConfig {
//!             steps: 2,
//!             batch_size: 2,
//!             learning_rate: 1e-3,
//!             weight_decay: 0.0,
//!             grad_clip_norm: Some(1.0),
//!             pad_token_id: 0,
//!         },
//!         &device,
//!         |_, _| {},
//!     )?;
//!
//!     let output = model.infer_tokens(
//!         &[1, 2],
//!         &ModelInferenceConfig {
//!             max_new_tokens: 2,
//!             pad_token_id: 0,
//!         },
//!         &device,
//!     )?;
//!     println!("tokens: {:?}", output.token_ids);
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! The default neural path uses Burn Flex for Candle-free CPU training.
//! Enable the `cuda` feature for GPU acceleration.

// ---- Public modules (the only ones users should care about) ----
pub mod device;
pub mod inference;
pub mod prelude;
pub mod training;

// ---- Internal modules ----
pub(crate) mod config;
pub(crate) mod engine;
pub(crate) mod error;
pub(crate) mod layout;
pub(crate) mod lm;
pub(crate) mod model;
pub(crate) mod optim;
pub(crate) mod param_io;
pub(crate) mod runtime;
pub(crate) mod screen;
pub(crate) mod tile;

// ---- High-level API re-exports ----
#[cfg(not(feature = "cuda"))]
pub use device::cpu;
pub use device::{auto_device, cuda};
pub use inference::{ChatModel, GenerationConfig};
pub use training::{ParameterBudget, Trainer, TrainingReport};

// ---- Core types (available through prelude) ----
pub use error::{Error, Result};
pub use model::{
    cross_entropy_loss_with_mask, DefaultMultiscreenModel, EvaluationResult, ModelInferenceConfig,
    ModelTrainingConfig, ModelTrainingReport, MultiscreenModel, MultiscreenModelConfig,
    MultiscreenModelOutput, MultiscreenParameterBudget,
};
pub use runtime::{device_label, DefaultAutodiffBackend, DefaultBackend, Device};

#[cfg(not(feature = "cuda"))]
pub use runtime::default_device;

#[cfg(feature = "cuda")]
pub use runtime::{CudaAutodiffBackend, CudaDevice, CudaMultiscreenModel};

// ---- Engine types (lightweight transition engine) ----
pub use config::{InferenceConfig, MultiscreenConfig, TrimConfig};
pub use engine::{InferenceOutput, MultiscreenEngine, TrainInput, TrainReport};
pub use layout::{
    causal_softmask, causal_trim_relevance, trim_and_square, ScreenLayout, TokenSpan,
};
pub use screen::{Screen, ScreenConfig};
pub use tile::{ScreeningGridConfig, Tile, TileConfig};

// ---- Burn re-exports ----
pub use burn::{
    tensor::backend::{AutodiffBackend, Backend},
    tensor::{Int, Tensor, TensorData},
};

#[cfg(feature = "cuda")]
pub use burn::backend::Cuda;

#[deprecated(note = "use MultiscreenConfig; the paper styles this as one word")]
pub type MultiScreenConfig = MultiscreenConfig;

#[deprecated(note = "use MultiscreenEngine; the paper styles this as one word")]
pub type MultiScreenEngine = MultiscreenEngine;

#[deprecated(note = "use ScreeningGridConfig for N_L x N_H naming")]
pub type GridConfig = ScreeningGridConfig;
