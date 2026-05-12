# multiscreen-rs

A Rust implementation of the Multiscreen neural language model — training and inference — powered by [Burn](https://github.com/tracel-ai/burn). CPU by default (Burn Flex), optional CUDA GPU support via feature flag. No built-in tokenizer — you encode/decode text with your own tokenizer and pass `Vec<u32>` token IDs directly.

## Installation

```toml
[dependencies]
multiscreen-rs = "0.1"
```

### CUDA GPU Support

```toml
[dependencies]
multiscreen-rs = { version = "0.1", features = ["cuda"] }
```

## Quick Start — Training

```rust
use multiscreen_rs::prelude::*;

fn main() -> multiscreen_rs::Result<()> {
    let mut trainer = Trainer::builder()
        .vocab_size(1000)
        .budget(ParameterBudget::Params10M)
        .device(cpu()?)
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

## Examples

The crate ships with two self-contained examples that use [SentencePiece](https://github.com/google/sentencepiece) tokenization.

### Train a Model

```bash
# Train 10M params, 10k steps
cargo run --release --example train_with_tokenizer -- \
    --train-dir examples/data --run-dir runs/10m-10k --budget 10m --steps 10000

# Train 1M params, 500 steps (for quick testing)
cargo run --release --example train_with_tokenizer -- \
    --train-dir examples/data --run-dir runs/test --budget 1m --steps 500

# Train with your own data
cargo run --release --example train_with_tokenizer -- \
    --train-dir /path/to/my/data --run-dir runs/custom --budget 10m --steps 50000
```

### Chat with a Trained Model

```bash
# Interactive mode (streaming output)
cargo run --release --example chat_with_tokenizer -- --run-dir runs/10m-10k

# One-shot prompt
cargo run --release --example chat_with_tokenizer -- \
    --run-dir runs/10m-10k --prompt "User: hello Assistant:"
```

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

## Device Selection

```rust
use multiscreen_rs::prelude::*;

let device = cpu()?;         // CPU (always available)
let device = cuda(0)?;       // CUDA GPU 0 (requires "cuda" feature)
let device = auto_device()?; // picks best available
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

- `tokenizer.model` — SentencePiece model (~280KB, 2778 vocab)
- `sample_chat.txt` — 15 English chat lines for quick testing

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
