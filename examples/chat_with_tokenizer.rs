//! Interactive chat with a trained Multiscreen model — streaming word-by-word output.
//!
//! # Quick start
//!
//! ```sh
//! # First, train a model:
//! cargo run --release --example train_with_tokenizer -- \
//!     --train-dir examples/data --run-dir runs/my-model --steps 5000
//!
//! # Then chat with it:
//! cargo run --release --example chat_with_tokenizer -- --run-dir runs/my-model
//!
//! # One-shot mode:
//! cargo run --release --example chat_with_tokenizer -- \
//!     --run-dir runs/my-model --prompt "user: สวัสดี assistant:"
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use multiscreen_rs::prelude::*;
use sentencepiece_rs::SentencePieceProcessor;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// SentencePiece adapter
// ---------------------------------------------------------------------------

struct SpTokenizer {
    proc: SentencePieceProcessor,
}

impl SpTokenizer {
    fn load(path: &Path) -> Result<Self> {
        Ok(Self {
            proc: SentencePieceProcessor::open(path)
                .with_context(|| format!("failed to load {}", path.display()))?,
        })
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.proc
            .encode_to_ids(text)
            .unwrap_or_default()
            .into_iter()
            .map(|id| id as u32)
            .collect()
    }

    fn eos_id(&self) -> Option<u32> {
        self.proc.eos_id().map(|id| id as u32)
    }

    fn id_to_piece(&self, id: u32) -> String {
        self.proc
            .id_to_piece(id as usize)
            .map(|s| s.to_owned())
            .unwrap_or_default()
    }
}

/// Decode a token piece from SentencePiece: replace `▁` with space.
fn piece_to_text(piece: &str) -> &str {
    // SentencePiece uses U+2581 (▁) as space marker
    piece.strip_prefix('\u{2581}').unwrap_or(piece)
}

// ---------------------------------------------------------------------------
// Tokenizer discovery
// ---------------------------------------------------------------------------

fn find_tokenizer(run_dir: &Path) -> Result<PathBuf> {
    // 1. Inside run dir (copied by train_with_tokenizer)
    let p = run_dir.join("tokenizer.model");
    if p.exists() {
        return Ok(p);
    }
    // 2. Bundled in examples/data/
    let p = PathBuf::from("examples/data/tokenizer.model");
    if p.exists() {
        return Ok(p);
    }
    anyhow::bail!(
        "tokenizer.model not found in {} or examples/data/",
        run_dir.display()
    )
}

// ---------------------------------------------------------------------------
// Sampling with temperature + top-k + repetition penalty
// ---------------------------------------------------------------------------

/// Sample from logits with temperature, top-k filtering, and repetition penalty.
/// Returns a sampled token ID.
fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    repetition_penalty: f32,
    recent_tokens: &[u32],
) -> u32 {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    if temperature <= 0.0 {
        // Greedy — but still apply repetition penalty
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, &v) in logits.iter().enumerate() {
            let mut score = v;
            if recent_tokens.contains(&(i as u32)) && repetition_penalty > 1.0 {
                score /= repetition_penalty;
            }
            if score > best.1 {
                best = (i, score);
            }
        }
        return best.0 as u32;
    }

    // Apply repetition penalty
    let mut scores: Vec<f32> = logits.to_vec();
    for &tok in recent_tokens {
        let idx = tok as usize;
        if idx < scores.len() && repetition_penalty > 1.0 {
            if scores[idx] > 0.0 {
                scores[idx] /= repetition_penalty;
            } else {
                scores[idx] *= repetition_penalty;
            }
        }
    }

    // Temperature scaling
    let scores: Vec<f32> = scores.iter().map(|s| s / temperature).collect();

    // Top-k filtering: keep only the top-k highest scores
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);

    // Softmax over remaining
    let max_val = indexed[0].1;
    let exps: Vec<(usize, f32)> = indexed
        .iter()
        .map(|&(idx, s)| (idx, (s - max_val).exp()))
        .collect();
    let sum: f32 = exps.iter().map(|(_, e)| *e).sum();
    let probs: Vec<(usize, f32)> = exps.iter().map(|&(idx, e)| (idx, e / sum)).collect();

    // Weighted random sample
    let r: f32 = rng.gen();
    let mut cumulative = 0.0f32;
    for &(idx, p) in &probs {
        cumulative += p;
        if r <= cumulative {
            return idx as u32;
        }
    }
    probs.last().map(|&(idx, _)| idx as u32).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "chat_with_tokenizer",
    about = "Chat with a trained Multiscreen model using streaming output"
)]
struct Args {
    /// Run directory from train_with_tokenizer (contains checkpoints/).
    #[arg(long, default_value = "runs/my-model")]
    run_dir: PathBuf,

    /// Checkpoint path. Defaults to run_dir/checkpoints/latest.mpk.
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// One-shot prompt. If omitted, starts interactive mode.
    #[arg(long)]
    prompt: Option<String>,

    /// Max tokens to generate per response.
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    /// Sampling temperature (0 = greedy, 1.0 = normal, >1 = more random).
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Top-k sampling: only consider top K most likely tokens.
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// Repetition penalty (>1.0 penalizes repeated tokens).
    #[arg(long, default_value_t = 1.2)]
    repetition_penalty: f32,

    /// System prompt to prepend (sets the character personality).
    #[arg(long)]
    system_prompt: Option<String>,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();

    // --- Resolve checkpoint path ---
    let ckpt = args
        .checkpoint
        .unwrap_or_else(|| args.run_dir.join("checkpoints/latest.mpk"));
    anyhow::ensure!(ckpt.exists(), "checkpoint not found: {}", ckpt.display());

    // --- Load tokenizer ---
    let tok_path = find_tokenizer(&args.run_dir)?;
    let sp = SpTokenizer::load(&tok_path)?;
    eprintln!("tokenizer: {}", tok_path.display());

    // --- Load model ---
    let model = ChatModel::load(&ckpt)?;
    eprintln!(
        "loaded model: {} params",
        model.config().estimated_parameter_count()
    );
    let seq_len = model.config().seq_len;
    let vocab_size = model.config().vocab_size;
    eprintln!("seq_len={seq_len}, vocab_size={vocab_size}");

    let eos_id = sp.eos_id();
    let pad_token_id: u32 = 0;

    // System prompt — match the training format: "system: ...\n"
    let system_text = args.system_prompt.as_deref().unwrap_or(
        "You are หมิว, a shy but sharp ม.6 student who secretly likes ธันวา. \
         Speak Thai mixed with English naturally. Be caring indirectly, thoughtful, and concise.",
    );
    let system_line = format!("system: {system_text}");

    eprintln!(
        "sampling: temperature={:.2}, top_k={}, repetition_penalty={:.2}",
        args.temperature, args.top_k, args.repetition_penalty
    );
    eprintln!();

    // --- Generate function (manual autoregressive with sampling) ---
    let generate = |prompt_text: &str,
                    max_new_tokens: usize,
                    temperature: f32,
                    top_k: usize,
                    repetition_penalty: f32|
     -> Result<String> {
        let prompt_ids = sp.encode(prompt_text);
        if prompt_ids.is_empty() {
            anyhow::bail!("prompt tokenized to empty sequence");
        }

        let mut output_ids = prompt_ids.clone();
        let mut full_text = String::new();
        let mut recent_tokens: Vec<u32> = Vec::new();
        const RECENT_WINDOW: usize = 16; // track last N tokens for repetition penalty

        for _i in 0..max_new_tokens {
            // Run one forward pass to get logits
            let input = context_window(&output_ids, seq_len, pad_token_id);
            let logits_tensor = model.predict_logits(&input)?;
            // logits_tensor shape: [1, seq_len, vocab_size]
            // Extract the last position's logits: slice [0, seq_len-1, :]
            let last_logits = logits_tensor
                .slice([0..1, seq_len - 1..seq_len, 0..vocab_size])
                .reshape([vocab_size]);
            let logits_data = last_logits.into_data();
            let logits_vec: Vec<f32> = logits_data.to_vec().unwrap_or_default();

            // Sample next token
            let next_token = sample_token(
                &logits_vec,
                temperature,
                top_k,
                repetition_penalty,
                &recent_tokens,
            );

            // Stop at EOS
            if Some(next_token) == eos_id {
                break;
            }

            output_ids.push(next_token);
            recent_tokens.push(next_token);
            if recent_tokens.len() > RECENT_WINDOW {
                recent_tokens.remove(0);
            }

            // Decode and print
            let piece = sp.id_to_piece(next_token);
            let text = piece_to_text(&piece);
            full_text.push_str(text);
            print!("{text}");
            io::stdout().flush().ok();
        }

        println!(); // newline after generation
        Ok(full_text)
    };

    // --- One-shot mode ---
    if let Some(prompt) = &args.prompt {
        // If prompt already starts with "system:" or "user:", use as-is
        let full_prompt = if prompt.contains(':')
            && (prompt.starts_with("system:") || prompt.starts_with("user:"))
        {
            prompt.clone()
        } else {
            format!("{system_line}\nuser: {prompt}\nassistant:")
        };
        let _ = generate(
            &full_prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.repetition_penalty,
        )?;
        return Ok(());
    }

    // --- Interactive mode ---
    eprintln!("interactive mode — type a message, press Enter. Empty line to exit.");
    let stdin = io::stdin();
    loop {
        print!("you> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if stdin.read_line(&mut input)? == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            break;
        }

        // Format: "system: ...\nuser: ...\nassistant:"  — matches training data
        let prompt = format!("{system_line}\nuser: {input}\nassistant:");
        print!("ai> ");
        io::stdout().flush()?;
        let _ = generate(
            &prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.repetition_penalty,
        )?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pad/truncate context to seq_len (left-pad with pad_token_id).
fn context_window(tokens: &[u32], seq_len: usize, pad_token_id: u32) -> Vec<u32> {
    if tokens.len() >= seq_len {
        tokens[(tokens.len() - seq_len)..].to_vec()
    } else {
        let pad_count = seq_len - tokens.len();
        let mut padded = vec![pad_token_id; pad_count];
        padded.extend_from_slice(tokens);
        padded
    }
}
