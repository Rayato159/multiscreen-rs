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
//!     --run-dir runs/my-model --prompt "User: hello Assistant:"
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
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,
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

    let eos_id = sp.eos_id();
    let gen_config = GenerationConfig {
        max_new_tokens: args.max_new_tokens,
        ..Default::default()
    };

    // --- Generate function ---
    let generate = |prompt_text: &str| -> Result<String> {
        let prompt_ids = sp.encode(prompt_text);
        let mut full_text = String::new();

        let _output =
            model.generate_stream(&prompt_ids, gen_config.clone(), |token_id, _idx| {
                if Some(token_id) == eos_id {
                    return false; // stop at EOS
                }
                let piece = sp.id_to_piece(token_id);
                let text = piece_to_text(&piece);
                full_text.push_str(text);
                print!("{text}");
                io::stdout().flush().ok();
                true
            })?;

        println!(); // newline after generation
        Ok(full_text)
    };

    // --- One-shot mode ---
    if let Some(prompt) = &args.prompt {
        let _ = generate(prompt)?;
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

        let prompt = format!("User: {input} Assistant:");
        print!("ai> ");
        io::stdout().flush()?;
        let _ = generate(&prompt)?;
    }

    Ok(())
}
