# multiscreen-rs

A Rust implementation of the Multiscreen neural language model — training and inference — powered by [Burn](https://github.com/tracel-ai/burn).

- **CPU** by default (Burn Flex with runtime SIMD detection: SSE, AVX, AVX2, AVX-512, NEON)
- **CUDA GPU** via `--features cuda` — runs natively on NVIDIA GPUs
- No built-in tokenizer — encode/decode with your own and pass `Vec<u32>` token IDs directly
- **Streaming inference** — token-by-token output like ChatGPT
- **Chat-aware training** — loss masking prevents the model from learning role labels (`system:`, `user:`, `assistant:`)
- **Sampling-based generation** — temperature, top-k, and repetition penalty for coherent output
- **Eval-only mode** — run evaluation + report on an existing checkpoint without retraining

## Installation

```toml
[dependencies]
multiscreen-rs = "0.2"
```

### CUDA GPU Support

```toml
[dependencies]
multiscreen-rs = { version = "0.2", features = ["cuda"] }
```

## Quick Start — Training

```rust
use multiscreen_rs::prelude::*;

fn main() -> multiscreen_rs::Result<()> {
    let mut trainer = Trainer::builder()
        .vocab_size(1000)
        .budget(ParameterBudget::Params10M)
        .device(auto_device()?)
        .batch_size(16)
        .seq_len(128)
        .steps(50_000)
        .build()?;

    // Token sequences from YOUR tokenizer
    let sequences = vec![
        vec![1, 2, 3, 4, 5],
        vec![1, 2, 6, 7, 5],
    ];

    let report = trainer.train_on_token_sequences(&sequences)?;
    println!("trained {} steps, final loss {:.4}", report.steps, report.final_loss);
    Ok(())
}
```

### Chat-Aware Training (Loss Masking)

When training on chat data with role labels (e.g. `system: ...`, `user: ...`, `assistant: ...`), use `train_on_chat_sequences` to mask loss on prompt tokens. The model learns to generate only the assistant's response:

```rust
use multiscreen_rs::prelude::*;

fn main() -> multiscreen_rs::Result<()> {
    let mut trainer = Trainer::builder()
        .vocab_size(1000)
        .budget(ParameterBudget::Params10M)
        .device(auto_device()?)
        .batch_size(16)
        .seq_len(128)
        .steps(50_000)
        .build()?;

    // (prompt_tokens, response_tokens) pairs
    let chat_pairs = vec![
        (vec![1, 2, 3], vec![4, 5, 6]),  // prompt → response
        (vec![1, 7, 8], vec![9, 10, 11]),
    ];

    let report = trainer.train_on_chat_sequences(&chat_pairs)?;
    println!("trained {} steps, final loss {:.4}", report.steps, report.final_loss);
    Ok(())
}
```

## Quick Start — Inference

```rust
use multiscreen_rs::prelude::*;

fn main() -> multiscreen_rs::Result<()> {
    let model = ChatModel::load("checkpoints/latest.mpk")?;
    let token_ids = model.generate(&[1, 2, 3], GenerationConfig::default())?;
    println!("generated tokens: {:?}", token_ids);
    Ok(())
}
```

### Streaming (token by token, like ChatGPT)

```rust
use multiscreen_rs::prelude::*;

fn main() -> multiscreen_rs::Result<()> {
    let model = ChatModel::load("checkpoints/latest.mpk")?;
    model.generate_stream(&[1, 2, 3], GenerationConfig::default(), |token_id, _index| {
        // Decode with YOUR tokenizer and print word-by-word
        print!("{} ", token_id);
        true // return false to stop early
    })?;
    Ok(())
}
```

### Custom Sampling (temperature, top-k, repetition penalty)

For custom sampling strategies, use `predict_logits` to get raw logits and apply your own sampling:

```rust
use multiscreen_rs::prelude::*;

fn main() -> multiscreen_rs::Result<()> {
    let model = ChatModel::load("checkpoints/latest.mpk")?;
    let context = vec![1, 2, 3];

    // Get logits: shape [1, seq_len, vocab_size]
    let logits = model.predict_logits(&context)?;
    // Apply temperature, top-k, nucleus sampling, etc.
    // (see examples/chat_with_tokenizer.rs for a full implementation)
    Ok(())
}
```

## Examples

The crate ships with two self-contained examples that use [SentencePiece](https://github.com/google/sentencepiece) tokenization.

### Train a Model

```bash
# CPU: Train 10M params, 10k steps
cargo run --release --example train_with_tokenizer -- \
    --train-dir examples/data --run-dir runs/10m-10k --budget 10m --steps 10000

# CUDA: Train 50M params, 30k steps on GPU
cargo run --release --features cuda --example train_with_tokenizer -- \
    --train-dir examples/data --run-dir runs/50m-30k --budget 50m --seq-len 256 --batch-size 4 --steps 30000 --lr 1e-4

# Quick test: 1M params, 500 steps
cargo run --release --example train_with_tokenizer -- \
    --train-dir examples/data --run-dir runs/test --budget 1m --steps 500

# Train with your own data
cargo run --release --example train_with_tokenizer -- \
    --train-dir /path/to/my/data --run-dir runs/custom --budget 10m --steps 50000
```

### Eval-Only Mode (skip training)

If training completed but evaluation/report failed (e.g. crashed mid-way), you can re-run **just the evaluation and report generation** without retraining:

```bash
# Load existing checkpoint, evaluate, and write report
cargo run --release --features cuda --example train_with_tokenizer -- \
    --train-dir examples/data --run-dir runs/50m-30k --budget 50m --seq-len 256 --batch-size 4 --eval-only
```

This loads `latest.mpk` + `latest.json` from the checkpoint directory, runs validation/test evaluation, measures inference latency, generates sample output, and writes the full `report.json` + `report.md`.

### Chat with a Trained Model

```bash
# Interactive mode (streaming word-by-word output)
cargo run --release --features cuda --example chat_with_tokenizer -- --run-dir runs/50m-30k

# One-shot prompt
cargo run --release --features cuda --example chat_with_tokenizer -- \
    --run-dir runs/50m-30k --prompt "สวัสดี"

# With custom sampling parameters
cargo run --release --features cuda --example chat_with_tokenizer -- \
    --run-dir runs/50m-30k \
    --temperature 0.8 \
    --top-k 40 \
    --repetition-penalty 1.2 \
    --max-new-tokens 256

# With a custom system prompt
cargo run --release --features cuda --example chat_with_tokenizer -- \
    --run-dir runs/50m-30k \
    --system-prompt "You are a helpful coding assistant."
```

#### Chat CLI Options

| Option | Default | Description |
|---|---|---|
| `--run-dir` | `runs/my-model` | Directory with checkpoints/ |
| `--checkpoint` | auto (latest.mpk) | Specific checkpoint path |
| `--prompt` | none (interactive) | One-shot prompt, skips interactive mode |
| `--max-new-tokens` | 128 | Max tokens to generate per response |
| `--temperature` | 0.8 | Sampling temperature (0 = greedy, 1.0 = normal, >1 = more random) |
| `--top-k` | 40 | Only consider top K most likely tokens |
| `--repetition-penalty` | 1.2 | Penalizes repeated tokens (>1.0 = stronger penalty) |
| `--system-prompt` | หมิว character | Custom system prompt |

### Generate a Loss Plot

```bash
# Requires Python + matplotlib + numpy
python examples/plot_loss.py runs/10m-10k/loss.csv
```

### Training CLI Options

| Option | Default | Description |
|---|---|---|
| `--train-dir` | `examples/data` | Directory with `tokenizer.model` + `.txt`/`.jsonl` files |
| `--run-dir` | `runs/my-model` | Output directory (checkpoints, reports, loss CSV) |
| `--budget` | `10m` | Parameter budget: `1m`, `5m`, `10m`, `50m`, `100m` |
| `--steps` | `10000` | Total optimizer steps |
| `--batch-size` | `4` | Batch size |
| `--seq-len` | `128` | Sequence length |
| `--lr` | `0.0002` | Learning rate |
| `--val-split` | `0.1` | Fraction of data for validation |
| `--log-interval` | `100` | Print loss every N steps |
| `--latency-tokens` | `20` | Tokens to generate for latency benchmark |
| `--eval-only` | `false` | Skip training — load existing checkpoint and only run evaluation + report |

## Training Reports

Every training run produces a complete report in `--run-dir`:

```
runs/my-model/
├── checkpoints/
│   ├── config.json       # Model architecture config
│   ├── latest.mpk        # Trained weights
│   └── latest.json       # Run metadata
├── tokenizer.model       # Copy of the tokenizer
├── loss.csv              # Per-step loss values (step,loss)
├── report.json           # Machine-readable full report
└── report.md             # Human-readable training report
```

The report includes:

- **Configuration** — budget, parameter count, seq len, batch size, learning rate
- **Training** — duration, throughput (steps/s), final loss, best loss
- **Validation** — loss, perplexity, next-token accuracy
- **Test** — loss, perplexity, next-token accuracy
- **Inference** — average latency per token, total generation time

All files under `runs/` are excluded from git via `.gitignore`.

### Loss Plot

The loss CSV can be plotted with the bundled Python script:

```bash
python examples/plot_loss.py runs/10m-10k/loss.csv
python examples/plot_loss.py runs/10m-10k/loss.csv --smooth 100
```

This generates `loss_plot.png` in the same directory.

### Evaluation Metrics

The model is automatically evaluated on held-out data after training:

| Metric | Description |
|---|---|
| **Loss** | Average cross-entropy loss |
| **Perplexity** | `exp(loss)` — lower is better |
| **Accuracy** | Fraction of tokens where `argmax(logits) == target` |

Data is split 80/10/10 (train/val/test) by default. Override with `--val-split`.

## Architecture

### Inference-Only Backend

`ChatModel` loads the model with the Autodiff backend (required for parameter loading), then calls `.valid()` to strip the autodiff wrapper. This produces a `MultiscreenModel<DefaultBackend>` that:

- Uses no gradient tracking during inference
- Does not build computation graphs → no VRAM leak
- Is safe for autoregressive generation loops that run many forward passes

### Loss Masking

When training on chat data, role labels like `system:`, `user:`, `assistant:` should not be learned as generation targets. `TrainingWindows::from_chat_sequences` creates a binary loss mask:

- `loss_mask = 0.0` for prompt tokens (system + user) — model sees them but doesn't learn to generate them
- `loss_mask = 1.0` for response tokens (assistant content) — model learns to generate these

This prevents the model from outputting role labels during inference.

## Device Selection

```rust
use multiscreen_rs::prelude::*;

// Recommended: auto-select best available backend
let device = auto_device()?;

// Explicit CPU (only without --features cuda)
let device = cpu()?;

// Explicit CUDA GPU (requires "cuda" feature)
let device = cuda(0)?;
```

## Parameter Budgets

Choose a model size with `ParameterBudget` — presets range from **1M to 100M parameters**:

```rust
ParameterBudget::Params1M   // ~1.2M
ParameterBudget::Params5M   // ~5.5M
ParameterBudget::Params10M  // ~10.5M  (default)
ParameterBudget::Params50M  // ~52.1M
ParameterBudget::Params100M // ~104.6M
```

## Bundled Data

`examples/data/` contains everything needed to run the examples out of the box:

- `tokenizer.model` — SentencePiece model (5487 vocab)
- `sample_chat.jsonl` — 100,000 chat lines in OpenAI message format for training

The training example supports two data formats:

- **`.txt`** — one sample per line
- **`.jsonl`** — each line is a JSON object:
  - `{"text": "your text here"}` — raw text format
  - `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` — OpenAI chat format

JSONL chat-format data is automatically detected and trained with loss masking.

## Contributing

Contributions are welcome! Keep patches focused, maintain the default CPU/Flex path, and run:

```bash
cargo fmt --all --check
cargo check --all-targets
cargo test
cargo clippy --all-targets -- -D warnings
```

## License

[MIT](LICENSE) · Copyright (c) 2026 multiscreen-rs contributors.
