use crate::{
    error::{Error, Result},
    runtime::DefaultAutodiffBackend,
};
use burn::{
    grad_clipping::GradientClippingConfig,
    module::{Module, Param},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{
        activation,
        backend::{AutodiffBackend, Backend},
        Int, Tensor, TensorData,
    },
};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::path::Path;

const EPS: f32 = 1e-6;

/// Supported neural Multiscreen parameter budgets.
///
/// The final count is approximate because the token embedding scales with the
/// caller's vocabulary size. Use `MultiscreenModelConfig::estimated_parameter_count`
/// to see the exact count for a resolved config.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultiscreenParameterBudget {
    Params1M,
    Params5M,
    Params10M,
    Params50M,
    Params100M,
}

impl MultiscreenParameterBudget {
    pub const ALL: [Self; 5] = [
        Self::Params1M,
        Self::Params5M,
        Self::Params10M,
        Self::Params50M,
        Self::Params100M,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Params1M => "1M",
            Self::Params5M => "5M",
            Self::Params10M => "10M",
            Self::Params50M => "50M",
            Self::Params100M => "100M",
        }
    }

    pub fn target_parameter_count(self) -> usize {
        match self {
            Self::Params1M => 1_000_000,
            Self::Params5M => 5_000_000,
            Self::Params10M => 10_000_000,
            Self::Params50M => 50_000_000,
            Self::Params100M => 100_000_000,
        }
    }
}

/// Paper-faithful neural Multiscreen model configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MultiscreenModelConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Context length `T`.
    pub seq_len: usize,
    /// Paper `N_L`: number of residual Multiscreen layers.
    pub layers: usize,
    /// Paper `N_H`: number of gated screening tiles per layer.
    pub tiles: usize,
    /// Paper `d_E`: token embedding/model width.
    pub d_model: usize,
    /// Paper `d_K`: query/key width. Must be at least 2 for MiPE.
    pub d_key: usize,
    /// Paper `d_V`: value/gate width.
    pub d_value: usize,
    /// Paper MiPE threshold `w_th`.
    pub w_th: f32,
}

impl MultiscreenModelConfig {
    pub fn tiny() -> Self {
        Self {
            vocab_size: 64,
            seq_len: 64,
            layers: 2,
            tiles: 4,
            d_model: 64,
            d_key: 16,
            d_value: 32,
            w_th: 32.0,
        }
    }

    pub fn tiny_for_tests() -> Self {
        Self {
            vocab_size: 32,
            seq_len: 8,
            layers: 1,
            tiles: 2,
            d_model: 16,
            d_key: 4,
            d_value: 8,
            w_th: 8.0,
        }
    }

    pub fn for_parameter_budget(
        budget: MultiscreenParameterBudget,
        vocab_size: usize,
        seq_len: usize,
    ) -> Self {
        match budget {
            MultiscreenParameterBudget::Params1M => Self::preset_1m(vocab_size, seq_len),
            MultiscreenParameterBudget::Params5M => Self::preset_5m(vocab_size, seq_len),
            MultiscreenParameterBudget::Params10M => Self::preset_10m(vocab_size, seq_len),
            MultiscreenParameterBudget::Params50M => Self::preset_50m(vocab_size, seq_len),
            MultiscreenParameterBudget::Params100M => Self::preset_100m(vocab_size, seq_len),
        }
    }

    pub fn preset_1m(vocab_size: usize, seq_len: usize) -> Self {
        Self::from_dimensions(vocab_size, seq_len, 2, 2, 128, 32, 64)
    }

    pub fn preset_5m(vocab_size: usize, seq_len: usize) -> Self {
        Self::from_dimensions(vocab_size, seq_len, 2, 4, 384, 96, 192)
    }

    pub fn preset_10m(vocab_size: usize, seq_len: usize) -> Self {
        Self::from_dimensions(vocab_size, seq_len, 3, 4, 512, 128, 256)
    }

    pub fn preset_50m(vocab_size: usize, seq_len: usize) -> Self {
        Self::from_dimensions(vocab_size, seq_len, 6, 4, 960, 240, 480)
    }

    pub fn preset_100m(vocab_size: usize, seq_len: usize) -> Self {
        Self::from_dimensions(vocab_size, seq_len, 8, 4, 1216, 304, 608)
    }

    pub fn paper_10m(vocab_size: usize, seq_len: usize) -> Self {
        Self::preset_10m(vocab_size, seq_len)
    }

    pub fn estimated_parameter_count(&self) -> usize {
        let embedding_params = self.vocab_size.saturating_mul(self.d_model);
        let per_tile_params = self
            .d_model
            .saturating_mul(
                2usize
                    .saturating_mul(self.d_key)
                    .saturating_add(3usize.saturating_mul(self.d_value)),
            )
            .saturating_add(3);
        let tile_params = self
            .layers
            .saturating_mul(self.tiles)
            .saturating_mul(per_tile_params);
        embedding_params
            .saturating_add(2)
            .saturating_add(tile_params)
    }

    fn from_dimensions(
        vocab_size: usize,
        seq_len: usize,
        layers: usize,
        tiles: usize,
        d_model: usize,
        d_key: usize,
        d_value: usize,
    ) -> Self {
        Self {
            vocab_size,
            seq_len,
            layers,
            tiles,
            d_model,
            d_key,
            d_value,
            w_th: 32.0,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure(self.vocab_size > 0, "vocab_size must be greater than zero")?;
        ensure(self.seq_len > 0, "seq_len must be greater than zero")?;
        ensure(self.layers > 0, "layers must be greater than zero")?;
        ensure(self.tiles > 0, "tiles must be greater than zero")?;
        ensure(self.d_model > 0, "d_model must be greater than zero")?;
        ensure(self.d_key >= 2, "d_key must be at least 2 for MiPE")?;
        ensure(self.d_value > 0, "d_value must be greater than zero")?;
        ensure(
            self.w_th.is_finite() && self.w_th > 0.0,
            "w_th must be positive and finite",
        )?;
        Ok(())
    }
}

/// Training options for token-sequence training.
#[derive(Clone, Debug)]
pub struct ModelTrainingConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub grad_clip_norm: Option<f64>,
    pub pad_token_id: u32,
}

impl Default for ModelTrainingConfig {
    fn default() -> Self {
        Self {
            steps: 100,
            batch_size: 4,
            learning_rate: 2e-4,
            weight_decay: 0.01,
            grad_clip_norm: Some(1.0),
            pad_token_id: 0,
        }
    }
}

/// Greedy generation options.
#[derive(Clone, Debug)]
pub struct ModelInferenceConfig {
    pub max_new_tokens: usize,
    pub pad_token_id: u32,
}

impl Default for ModelInferenceConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 16,
            pad_token_id: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelTrainingReport {
    pub steps: usize,
    pub final_loss: f32,
    pub training_window_count: usize,
    pub parameter_count: usize,
}

/// Result of evaluating a model on held-out sequences.
#[derive(Clone, Debug)]
pub struct EvaluationResult {
    /// Average cross-entropy loss across all batches.
    pub loss: f32,
    /// Perplexity = exp(average_loss).
    pub perplexity: f32,
    /// Fraction of tokens where argmax(logits) == target (next-token accuracy).
    pub accuracy: f64,
    /// Number of batches evaluated.
    pub num_batches: usize,
    /// Total number of (unmasked) tokens evaluated.
    pub total_tokens: usize,
}

pub struct MultiscreenModelOutput {
    pub token_ids: Vec<u32>,
}

/// Burn-backed neural Multiscreen language model ported from
/// `multiscreen-testing`.
#[derive(Module, Debug)]
pub struct MultiscreenModel<B: Backend = DefaultAutodiffBackend> {
    #[module(skip)]
    config: MultiscreenModelConfig,
    token_embedding: Param<Tensor<B, 2>>,
    s_e: Param<Tensor<B, 1>>,
    s_f: Param<Tensor<B, 1>>,
    layers: Vec<MultiscreenLayer<B>>,
}

/// Convenience alias for the default Burn Flex autodiff model.
pub type DefaultMultiscreenModel = MultiscreenModel<DefaultAutodiffBackend>;

#[derive(Module, Debug)]
struct MultiscreenLayer<B: Backend> {
    tiles: Vec<GatedScreeningTile<B>>,
}

#[derive(Module, Debug)]
struct GatedScreeningTile<B: Backend> {
    w_q: Param<Tensor<B, 2>>,
    w_k: Param<Tensor<B, 2>>,
    w_v: Param<Tensor<B, 2>>,
    w_g: Param<Tensor<B, 2>>,
    w_o: Param<Tensor<B, 2>>,
    s_w: Param<Tensor<B, 1>>,
    s_r: Param<Tensor<B, 1>>,
    s_o: Param<Tensor<B, 1>>,
    #[module(skip)]
    w_th: f32,
}

impl<B: Backend> MultiscreenModel<B> {
    pub fn new(config: MultiscreenModelConfig, device: &B::Device) -> Result<Self> {
        config.validate()?;

        let mut seed = 0x4d55_4c54_4953_4352;
        let token_embedding = init_matrix(
            config.vocab_size,
            config.d_model,
            0.1 / (config.d_model as f32).sqrt(),
            &mut seed,
            device,
        );
        let s_e = init_scalar(0.0, device);
        let s_f = init_scalar((config.d_model as f32).sqrt().ln(), device);

        let mut layers = Vec::with_capacity(config.layers);
        for _layer_idx in 0..config.layers {
            let mut tiles = Vec::with_capacity(config.tiles);
            for tile_idx in 0..config.tiles {
                let w_q = init_matrix(
                    config.d_model,
                    config.d_key,
                    0.1 / (config.d_key as f32).sqrt(),
                    &mut seed,
                    device,
                );
                let w_k = init_matrix(
                    config.d_model,
                    config.d_key,
                    0.1 / (config.d_key as f32).sqrt(),
                    &mut seed,
                    device,
                );
                let w_v = init_matrix(
                    config.d_model,
                    config.d_value,
                    0.1 / (config.d_value as f32).sqrt(),
                    &mut seed,
                    device,
                );
                let w_g = init_matrix(config.d_model, config.d_value, 0.1, &mut seed, device);
                let w_o = init_matrix(
                    config.d_value,
                    config.d_model,
                    0.1 / (config.d_model as f32).sqrt(),
                    &mut seed,
                    device,
                );

                let window_frac = if config.tiles == 1 {
                    0.0
                } else {
                    tile_idx as f32 / (config.tiles - 1) as f32
                };
                let s_w = init_scalar(window_frac * config.w_th.ln(), device);
                let s_r = init_scalar(0.0, device);
                let s_o = init_scalar(-0.5 * ((config.layers * config.tiles) as f32).ln(), device);

                tiles.push(GatedScreeningTile {
                    w_q,
                    w_k,
                    w_v,
                    w_g,
                    w_o,
                    s_w,
                    s_r,
                    s_o,
                    w_th: config.w_th,
                });
            }
            layers.push(MultiscreenLayer { tiles });
        }

        Ok(Self {
            config,
            token_embedding,
            s_e,
            s_f,
            layers,
        })
    }

    pub fn config(&self) -> &MultiscreenModelConfig {
        &self.config
    }

    pub fn parameter_count(&self) -> usize {
        self.num_params()
    }

    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq_len] = tokens.dims();
        let embedding = row_unit_normalize(self.token_embedding.val());
        let one_hot = tokens.one_hot::<3>(self.config().vocab_size).float();
        let mut x =
            linear(one_hot, embedding.clone()).reshape([batch, seq_len, self.config().d_model]);
        let x_dims = x.dims();
        x = x * expand_scalar3(self.s_e.val().exp(), x_dims);

        for layer in &self.layers {
            let mut layer_update = Tensor::<B, 3>::zeros(x.dims(), &x.device());
            for tile in &layer.tiles {
                layer_update = layer_update + tile.forward(x.clone());
            }
            x = x + layer_update;
        }

        let logits_weight = embedding.swap_dims(0, 1);
        let logits = linear(x, logits_weight);
        logits.clone() * expand_scalar3(self.s_f.val().exp(), logits.dims())
    }

    pub fn save_parameters(&self, path: impl AsRef<Path>) -> Result<()> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.clone()
            .save_file(path.as_ref().to_path_buf(), &recorder)
            .map_err(|err| Error::Serialization(err.to_string()))
    }

    pub fn load_parameters(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let device =
            self.devices().into_iter().next().ok_or_else(|| {
                Error::Serialization("model has no device for parameter load".into())
            })?;
        let loaded = self
            .clone()
            .load_file(path.as_ref().to_path_buf(), &recorder, &device)
            .map_err(|err| Error::Serialization(err.to_string()))?;
        *self = loaded;
        Ok(())
    }

    /// Greedy token generation.
    /// Generate tokens one at a time, invoking a callback for each newly
    /// produced token. This enables streaming / word-by-word output similar
    /// to ChatGPT.
    ///
    /// The callback receives `(token_id, index)` where `index` is the
    /// zero-based position of the *new* token (0 = first generated token).
    /// If the callback returns `false`, generation stops early.
    ///
    /// Returns the full output (prompt + generated) token sequence.
    pub fn infer_tokens_stream(
        &self,
        prompt: &[u32],
        inference: &ModelInferenceConfig,
        device: &B::Device,
        mut on_token: impl FnMut(u32, usize) -> bool,
    ) -> Result<MultiscreenModelOutput> {
        if prompt.is_empty() {
            return Err(Error::Inference(
                "prompt must contain at least one token".to_string(),
            ));
        }

        let mut output = prompt.to_vec();
        for i in 0..inference.max_new_tokens {
            let next = self.predict_next_token(&output, inference.pad_token_id, device)?;
            output.push(next);
            if !on_token(next, i) {
                break;
            }
        }

        Ok(MultiscreenModelOutput { token_ids: output })
    }

    /// Generate tokens and return them all at once (non-streaming).
    ///
    /// For streaming / token-by-token output, use [`Self::infer_tokens_stream`].
    pub fn infer_tokens(
        &self,
        prompt: &[u32],
        inference: &ModelInferenceConfig,
        device: &B::Device,
    ) -> Result<MultiscreenModelOutput> {
        self.infer_tokens_stream(prompt, inference, device, |_, _| true)
    }

    pub fn predict_next_token(
        &self,
        context: &[u32],
        pad_token_id: u32,
        device: &B::Device,
    ) -> Result<u32> {
        if context.is_empty() {
            return Err(Error::Inference(
                "context must contain at least one token".to_string(),
            ));
        }

        let input = context_window(context, self.config().seq_len, pad_token_id);
        let input = tensor_from_u32::<B, 2>(input, [1, self.config().seq_len], device)?;
        let logits = self.forward(input);
        let last_logits = logits
            .slice([
                0..1,
                self.config().seq_len - 1..self.config().seq_len,
                0..self.config().vocab_size,
            ])
            .reshape([self.config().vocab_size]);
        let values = tensor_to_vec(last_logits)?;
        argmax(&values).map(|idx| idx as u32)
    }
}

impl<B> MultiscreenModel<B>
where
    B: AutodiffBackend,
{
    /// Trains this model directly on token sequences.
    ///
    /// The optional `on_step` callback is invoked after each optimizer step with
    /// `(step_index, loss_value)`. Use it for progress logging, CSV export, etc.
    pub fn train_token_sequences(
        &mut self,
        sequences: &[Vec<u32>],
        training: &ModelTrainingConfig,
        device: &B::Device,
        mut on_step: impl FnMut(usize, f32),
    ) -> Result<ModelTrainingReport> {
        if training.batch_size == 0 {
            return Err(Error::Training(
                "batch_size must be greater than zero".to_string(),
            ));
        }
        let windows = TrainingWindows::from_sequences(
            sequences,
            self.config().seq_len,
            training.pad_token_id,
        )?;
        if windows.is_empty() {
            return Err(Error::Training(
                "training requires at least one sequence with two or more tokens".to_string(),
            ));
        }

        let mut optimizer_config =
            AdamWConfig::new().with_weight_decay(training.weight_decay as f32);
        if let Some(max_norm) = training.grad_clip_norm.filter(|value| *value > 0.0) {
            optimizer_config = optimizer_config
                .with_grad_clipping(Some(GradientClippingConfig::Norm(max_norm as f32)));
        }
        let mut optimizer = optimizer_config.init::<B, Self>();
        let mut model = self.clone();
        let mut final_loss = f32::NAN;

        for step in 0..training.steps {
            let batch = windows.batch::<B>(step, training.batch_size, device)?;
            let logits = model.forward(batch.inputs);
            let loss = cross_entropy_loss_with_mask(logits, batch.targets, batch.loss_mask);
            final_loss = tensor_scalar(loss.clone())?;
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(training.learning_rate, model, grads);
            on_step(step, final_loss);
        }

        if training.steps == 0 {
            let batch = windows.batch::<B>(0, training.batch_size, device)?;
            final_loss = tensor_scalar(cross_entropy_loss_with_mask(
                model.forward(batch.inputs),
                batch.targets,
                batch.loss_mask,
            ))?;
        }

        *self = model;

        Ok(ModelTrainingReport {
            steps: training.steps,
            final_loss,
            training_window_count: windows.len(),
            parameter_count: self.parameter_count(),
        })
    }

    /// Evaluates the model on token sequences, returning average loss,
    /// perplexity, and next-token prediction accuracy.
    pub fn evaluate_on_sequences(
        &self,
        sequences: &[Vec<u32>],
        seq_len: usize,
        batch_size: usize,
        pad_token_id: u32,
        device: &B::Device,
    ) -> Result<EvaluationResult> {
        let windows = TrainingWindows::from_sequences(sequences, seq_len, pad_token_id)?;
        if windows.is_empty() {
            return Ok(EvaluationResult {
                loss: f32::NAN,
                perplexity: f32::NAN,
                accuracy: 0.0,
                num_batches: 0,
                total_tokens: 0,
            });
        }

        let num_batches = windows.len().div_ceil(batch_size);
        let mut total_loss = 0.0_f64;
        let mut total_correct = 0_usize;
        let mut total_tokens = 0_usize;

        for step in 0..num_batches {
            let batch = windows.batch::<B>(step, batch_size, device)?;
            let logits = self.forward(batch.inputs); // [B, S, V]
            let loss = cross_entropy_loss_with_mask(
                logits.clone(),
                batch.targets.clone(),
                batch.loss_mask.clone(),
            );
            let loss_val = tensor_scalar(loss)? as f64;
            total_loss += loss_val;

            // Compute accuracy: argmax(logits) == target, masked
            let [b, s, v] = logits.dims();
            let mask_vec: Vec<f32> = batch
                .loss_mask
                .clone()
                .reshape([b * s])
                .into_data()
                .into_vec::<f32>()
                .map_err(|e| Error::Inference(e.to_string()))?;
            let target_vec: Vec<i32> = batch
                .targets
                .clone()
                .reshape([b * s])
                .into_data()
                .into_vec::<i32>()
                .map_err(|e| Error::Inference(e.to_string()))?;
            let logit_vec: Vec<f32> = logits
                .reshape([b * s * v])
                .into_data()
                .into_vec::<f32>()
                .map_err(|e| Error::Inference(e.to_string()))?;

            for bi in 0..b {
                for si in 0..s {
                    let mi = bi * s + si;
                    if mask_vec[mi] < 0.5 {
                        continue;
                    }
                    total_tokens += 1;
                    let base = bi * s * v + si * v;
                    let mut best_idx = 0;
                    let mut best_val = f32::NEG_INFINITY;
                    for vi in 0..v {
                        let val = logit_vec[base + vi];
                        if val > best_val {
                            best_val = val;
                            best_idx = vi;
                        }
                    }
                    if best_idx == target_vec[mi] as usize {
                        total_correct += 1;
                    }
                }
            }
        }

        let avg_loss = total_loss / num_batches as f64;
        let perplexity = avg_loss.exp();
        let accuracy = if total_tokens > 0 {
            total_correct as f64 / total_tokens as f64
        } else {
            0.0
        };

        Ok(EvaluationResult {
            loss: avg_loss as f32,
            perplexity: perplexity as f32,
            accuracy,
            num_batches,
            total_tokens,
        })
    }
}

impl<B: Backend> crate::lm::LanguageModel<B> for MultiscreenModel<B> {
    fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        MultiscreenModel::forward(self, tokens)
    }
}

impl<B: Backend> crate::lm::TrainableLanguageModel<B> for MultiscreenModel<B> {}

impl<B: Backend> GatedScreeningTile<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = row_unit_normalize(linear(x.clone(), self.w_q.val()));
        let k = row_unit_normalize(linear(x.clone(), self.w_k.val()));
        let v = row_unit_normalize(linear(x.clone(), self.w_v.val()));
        let g = linear(x, self.w_g.val());

        let w = self.s_w.val().clamp(-10.0, 8.0).exp() + 1.0;
        let r = activation::sigmoid(self.s_r.val().clamp(-10.0, 8.0));

        let q = apply_mipe(q, w.clone(), self.w_th);
        let k = apply_mipe(k, w.clone(), self.w_th);

        let similarity = q.matmul(k.swap_dims(1, 2));
        let alpha = trim_and_square_tensor(similarity.clone(), r);
        let softmask = causal_softmask_tensor::<B>(similarity.dims()[1], w, &similarity.device());
        let relevance = alpha * softmask.unsqueeze();
        let h = relevance.matmul(v);
        let u = tanh_norm(h);
        let gate = activation::silu(g).tanh();
        let gated = u * gate;
        let out = linear(gated, self.w_o.val());
        out.clone() * expand_scalar3(self.s_o.val().exp(), out.dims())
    }
}

#[allow(dead_code)]
pub fn cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    let device = logits.device();
    let [batch, seq_len, _] = logits.dims();
    let loss_mask = Tensor::<B, 2>::ones([batch, seq_len], &device);
    cross_entropy_loss_with_mask(logits, targets, loss_mask)
}

pub fn cross_entropy_loss_with_mask<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let [batch, seq_len, vocab_size] = logits.dims();
    let token_count = batch * seq_len;
    let flat_logits = logits.reshape([token_count, vocab_size]);
    let flat_targets = targets.reshape([token_count]);
    let flat_mask = loss_mask.reshape([token_count]);
    let log_probs = activation::log_softmax(flat_logits, 1);
    let target_probs = flat_targets.one_hot::<2>(vocab_size).float();
    let picked = (log_probs * target_probs).sum_dim(1).reshape([token_count]);
    let denom = flat_mask.clone().sum().add_scalar(EPS);
    (picked.neg() * flat_mask).sum() / denom
}

pub fn row_unit_normalize<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let denom = x.clone().square().sum_dim(D - 1).add_scalar(EPS).sqrt();
    x / denom
}

fn trim_and_square_tensor<B: Backend>(similarity: Tensor<B, 3>, r: Tensor<B, 1>) -> Tensor<B, 3> {
    let distance_from_one = similarity.mul_scalar(-1.0).add_scalar(1.0);
    let scaled = distance_from_one.clone() / expand_scalar3(r, distance_from_one.clone().dims());
    scaled
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .clamp(0.0, 1.0)
        .square()
}

fn causal_softmask_tensor<B: Backend>(
    seq_len: usize,
    w: Tensor<B, 1>,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut distances = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            distances.push(j as f32 - i as f32);
        }
    }

    let dist = Tensor::<B, 2>::from_data(TensorData::new(distances, [seq_len, seq_len]), device);
    let w = expand_scalar2(w, [seq_len, seq_len]);
    let causal = dist.clone().lower_equal_elem(0.0);
    let within_window = dist.clone().greater(w.clone().neg());
    let active = causal.float() * within_window.float();
    let tapered = (dist / w)
        .mul_scalar(PI)
        .cos()
        .mul_scalar(0.5)
        .add_scalar(0.5);
    active * tapered
}

pub fn tanh_norm<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let norm = x.clone().square().sum_dim(D - 1).add_scalar(EPS).sqrt();
    let scale = norm.clone().tanh() / norm;
    x * scale
}

fn apply_mipe<B: Backend>(z: Tensor<B, 3>, w: Tensor<B, 1>, w_th: f32) -> Tensor<B, 3> {
    let [_batch, seq_len, d_key] = z.dims();
    debug_assert!(d_key >= 2);

    let positions = Tensor::<B, 3>::from_data(
        TensorData::new(
            (0..seq_len).map(|idx| idx as f32).collect::<Vec<_>>(),
            [1, seq_len, 1],
        ),
        &z.device(),
    );

    let gamma_raw = w
        .clone()
        .mul_scalar(PI / w_th)
        .cos()
        .mul_scalar(0.5)
        .add_scalar(0.5);
    let gamma = gamma_raw.mask_fill(w.clone().greater_equal_elem(w_th), 0.0);
    let gamma = expand_scalar3(gamma, [1, seq_len, 1]);
    let w = expand_scalar3(w, [1, seq_len, 1]);
    let phi = (positions * gamma / w).mul_scalar(PI);
    let cos_phi = phi.clone().cos();
    let sin_phi = phi.sin();

    let x0 = z.clone().narrow(2, 0, 1);
    let x1 = z.clone().narrow(2, 1, 1);
    let rot0 = x0.clone() * cos_phi.clone() - x1.clone() * sin_phi.clone();
    let rot1 = x0 * sin_phi + x1 * cos_phi;

    if d_key == 2 {
        Tensor::cat(vec![rot0, rot1], 2)
    } else {
        let rest = z.narrow(2, 2, d_key - 2);
        Tensor::cat(vec![rot0, rot1, rest], 2)
    }
}

fn linear<B: Backend>(x: Tensor<B, 3>, weight: Tensor<B, 2>) -> Tensor<B, 3> {
    let [batch, _seq_len, _in_dim] = x.dims();
    let [weight_in, out_dim] = weight.dims();
    x.matmul(weight.unsqueeze::<3>().expand([batch, weight_in, out_dim]))
}

fn expand_scalar2<B: Backend>(value: Tensor<B, 1>, dims: [usize; 2]) -> Tensor<B, 2> {
    value.unsqueeze::<2>().expand(dims)
}

fn expand_scalar3<B: Backend>(value: Tensor<B, 1>, dims: [usize; 3]) -> Tensor<B, 3> {
    value.unsqueeze::<3>().expand(dims)
}

fn init_scalar<B: Backend>(value: f32, device: &B::Device) -> Param<Tensor<B, 1>> {
    Param::from_tensor(Tensor::<B, 1>::from_data([value], device))
}

fn init_matrix<B: Backend>(
    rows: usize,
    cols: usize,
    std: f32,
    seed: &mut u64,
    device: &B::Device,
) -> Param<Tensor<B, 2>> {
    let values = gaussian_values(rows * cols, std, seed);
    Param::from_tensor(Tensor::<B, 2>::from_data(
        TensorData::new(values, [rows, cols]),
        device,
    ))
}

fn gaussian_values(len: usize, std: f32, seed: &mut u64) -> Vec<f32> {
    let mut values = Vec::with_capacity(len);
    while values.len() < len {
        let u1 = next_uniform(seed).max(1e-7);
        let u2 = next_uniform(seed);
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        values.push(radius * theta.cos() * std);
        if values.len() < len {
            values.push(radius * theta.sin() * std);
        }
    }
    values
}

fn next_uniform(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (*seed >> 40) as u32;
    (bits as f32 + 1.0) / ((1u32 << 24) as f32 + 2.0)
}

fn context_window(context: &[u32], seq_len: usize, pad_token_id: u32) -> Vec<u32> {
    let mut input = vec![pad_token_id; seq_len];
    let suffix = if context.len() > seq_len {
        &context[context.len() - seq_len..]
    } else {
        context
    };
    let start = seq_len - suffix.len();
    input[start..].copy_from_slice(suffix);
    input
}

fn argmax(values: &[f32]) -> Result<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .ok_or_else(|| Error::Inference("cannot argmax empty logits".to_string()))
}

fn tensor_scalar<B: Backend>(tensor: Tensor<B, 1>) -> Result<f32> {
    tensor
        .into_data()
        .into_vec::<f32>()
        .map_err(|err| Error::Training(err.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| Error::Training("expected scalar tensor".to_string()))
}

fn tensor_to_vec<B: Backend>(tensor: Tensor<B, 1>) -> Result<Vec<f32>> {
    tensor
        .into_data()
        .into_vec::<f32>()
        .map_err(|err| Error::Inference(err.to_string()))
}

fn tensor_from_u32<B: Backend, const D: usize>(
    values: Vec<u32>,
    shape: [usize; D],
    device: &B::Device,
) -> Result<Tensor<B, D, Int>> {
    let values = values
        .into_iter()
        .map(|value| {
            i32::try_from(value)
                .map_err(|_| Error::Config(format!("token id {value} exceeds i32::MAX")))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::<B, D, Int>::from_data(
        TensorData::new(values, shape),
        device,
    ))
}

fn ensure(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Config(message.to_string()))
    }
}

struct TrainingWindow {
    inputs: Vec<u32>,
    targets: Vec<u32>,
    loss_mask: Vec<f32>,
}

struct TrainingWindows {
    windows: Vec<TrainingWindow>,
    seq_len: usize,
}

impl TrainingWindows {
    fn from_sequences(sequences: &[Vec<u32>], seq_len: usize, pad_token_id: u32) -> Result<Self> {
        let mut windows = Vec::new();
        for sequence in sequences {
            if sequence.len() < 2 {
                continue;
            }

            let mut start = 0;
            while start + 1 < sequence.len() {
                let end = (start + seq_len + 1).min(sequence.len());
                let chunk = &sequence[start..end];
                let prediction_count = chunk.len() - 1;

                let mut inputs = vec![pad_token_id; seq_len];
                let mut targets = vec![pad_token_id; seq_len];
                let mut loss_mask = vec![0.0; seq_len];
                inputs[..prediction_count].copy_from_slice(&chunk[..prediction_count]);
                targets[..prediction_count].copy_from_slice(&chunk[1..]);
                loss_mask[..prediction_count].fill(1.0);

                windows.push(TrainingWindow {
                    inputs,
                    targets,
                    loss_mask,
                });

                if end == sequence.len() {
                    break;
                }
                start += seq_len;
            }
        }

        Ok(Self { windows, seq_len })
    }

    fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    fn len(&self) -> usize {
        self.windows.len()
    }

    fn batch<B: Backend>(
        &self,
        step: usize,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<TokenBatch<B>> {
        let mut inputs = Vec::with_capacity(batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(batch_size * self.seq_len);
        let mut loss_mask = Vec::with_capacity(batch_size * self.seq_len);

        for batch_idx in 0..batch_size {
            let index = (step * batch_size + batch_idx) % self.windows.len();
            let window = &self.windows[index];
            inputs.extend_from_slice(&window.inputs);
            targets.extend_from_slice(&window.targets);
            loss_mask.extend_from_slice(&window.loss_mask);
        }

        Ok(TokenBatch {
            inputs: tensor_from_u32(inputs, [batch_size, self.seq_len], device)?,
            targets: tensor_from_u32(targets, [batch_size, self.seq_len], device)?,
            loss_mask: Tensor::<B, 2>::from_data(
                TensorData::new(loss_mask, [batch_size, self.seq_len]),
                device,
            ),
        })
    }
}

struct TokenBatch<B: Backend> {
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Tensor<B, 2>,
}

#[cfg(test)]
#[allow(dead_code)]
pub fn make_batch<B: Backend>(
    step: usize,
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    device: &B::Device,
) -> Result<(Tensor<B, 2, Int>, Tensor<B, 2, Int>)> {
    let mut inputs = Vec::with_capacity(batch_size * seq_len);
    let mut targets = Vec::with_capacity(batch_size * seq_len);

    for batch in 0..batch_size {
        let offset = (step * 7 + batch * 13) % vocab_size;
        for pos in 0..seq_len {
            let token = ((offset + pos) % vocab_size) as u32;
            let next = ((offset + pos + 1) % vocab_size) as u32;
            inputs.push(token);
            targets.push(next);
        }
    }

    Ok((
        tensor_from_u32(inputs, [batch_size, seq_len], device)?,
        tensor_from_u32(targets, [batch_size, seq_len], device)?,
    ))
}
