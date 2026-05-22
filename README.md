# multiscreen-rs

A Rust implementation of the Multiscreen neural language model — training and inference — powered by [Burn](https://github.com/tracel-ai/burn). Based on [Screening Is Enough](https://arxiv.org/pdf/2604.01178).

## Quickstart

```bash
# Train (10M params, ~25 min on CUDA)
cargo run --release --features cuda --example train_with_tokenizer -- \
    --budget 10m --steps 10000 --batch-size 16 --seq-len 256

# Chat
cargo run --release --features cuda --example chat_with_tokenizer -- --run-dir runs/my-model
```

Data and tokenizer bundled in `examples/data/`. No setup needed.

---

## CLI

### Train

```bash
# Quick test (CPU, ~1 min)
cargo run --release --example train_with_tokenizer -- --budget 1m --steps 500

# 10M on CUDA (~25 min)
cargo run --release --features cuda --example train_with_tokenizer -- \
    --budget 10m --steps 10000 --batch-size 16 --seq-len 256

# 50M on CUDA (~2-3 hrs)
cargo run --release --features cuda --example train_with_tokenizer -- \
    --budget 50m --steps 10000 --batch-size 4 --seq-len 256

# Your own data (put tokenizer.model + .jsonl in a directory)
cargo run --release --features cuda --example train_with_tokenizer -- \
    --train-dir /path/to/data --run-dir runs/custom --budget 10m --steps 50000
```

### Chat

```bash
# Interactive (streaming)
cargo run --release --features cuda --example chat_with_tokenizer -- --run-dir runs/my-model

# One-shot
cargo run --release --features cuda --example chat_with_tokenizer -- \
    --run-dir runs/my-model --prompt "Once upon a time"

# Custom sampling
cargo run --release --features cuda --example chat_with_tokenizer -- \
    --run-dir runs/my-model --temperature 0.8 --top-k 40 --max-new-tokens 256
```

> Run with `--help` to see all options (training & chat).

### Loss Plot

```bash
python examples/plot_loss.py runs/my-model/loss.csv
# Creates runs/my-model/loss.png
```

---

## Data Formats

Put training data alongside `tokenizer.model` in any directory:

| Format | Description |
|---|---|
| `.txt` | Blank lines separate samples; split 60/40 into prompt/response |
| `.jsonl` text | `{"text": "your text here"}` — one sample per line |
| `.jsonl` chat | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` — OpenAI format |

> **Tip:** Aim for ≥10× more tokens than parameters. 10M params → ~83M+ tokens.

---

## Library Usage

```toml
[dependencies]
multiscreen-rs = "0.2"
# multiscreen-rs = { version = "0.2", features = ["cuda"] }
```

```rust
use multiscreen_rs::prelude::*;

// Training
let mut trainer = Trainer::builder()
    .vocab_size(1000)
    .budget(ParameterBudget::Params10M)
    .device(auto_device()?)
    .batch_size(16).seq_len(128).steps(50_000)
    .build()?;

let report = trainer.train_on_token_sequences(&sequences)?;
// or: trainer.train_on_chat_sequences(&chat_pairs)?;

// Inference
let model = ChatModel::load("checkpoints/latest.mpk")?;
let tokens = model.generate(&[1, 2, 3], GenerationConfig::default())?;

model.generate_stream(&[1, 2, 3], GenerationConfig::default(), |id, _| {
    print!("{id} "); true
})?;
```

---

## Architecture

```
Transformer:    score = QK^T/√d → softmax → attn · V
Multiscreen:    alpha = clamp(1 - r*(1 - QK^T), 0, 1)²   (trim-and-square)
                relevance = alpha * causal_cosine_window   (NO softmax)
                out = tanh_norm(relevance · V) * silu_gate(g)
```

No softmax → no attention dilution at long contexts. Advantages show at seq_len 4K+.

Parameter budgets: `1m` `5m` `10m` `50m` `100m`

## License

[MIT](LICENSE)
