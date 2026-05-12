use burn::{
    module::Module,
    tensor::{backend::Backend, Int, Tensor},
};

/// Minimal language-model trait used by the Burn-backed Multiscreen model.
#[allow(dead_code)]
pub trait LanguageModel<B: Backend> {
    fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3>;
}

/// Trainable Burn language model.
#[allow(dead_code)]
pub trait TrainableLanguageModel<B: Backend>: LanguageModel<B> + Module<B> {
    fn parameter_count(&self) -> usize {
        self.num_params()
    }
}
