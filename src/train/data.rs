//! Training data: a tokenizer-agnostic [`CorpusReader`] that serves token windows
//! to the training loop, plus helpers to fetch a tokenizer + text dataset from the
//! Hugging Face hub.
//!
//! Tokenization is decoupled from batching on purpose: [`CorpusReader`] only sees a
//! flat token stream, so the loop is exercised offline on synthetic tokens, while
//! the network-bound HF fetch lives behind [`hf_corpus`] and is used by the binary.

use std::io::Read;
use std::path::Path;

use tokenizers::Tokenizer;

use crate::train::r#loop::BatchSource;

/// A flat token stream served as next-token-prediction windows. The cursor wraps to
/// the start when fewer than `seq_len + 1` tokens remain, so a fixed-size corpus
/// drives an arbitrary number of training steps.
#[derive(Clone)]
pub struct CorpusReader {
    tokens: Vec<u32>,
    pos: usize,
}

impl CorpusReader {
    pub fn new(tokens: Vec<u32>) -> Self {
        Self { tokens, pos: 0 }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Split off a held-out tail: the last `val_frac` fraction of tokens becomes a
    /// second reader, the rest stays in `self`. Used to get a validation split that
    /// never overlaps training windows.
    pub fn split_val(mut self, val_frac: f32) -> (CorpusReader, CorpusReader) {
        let val_n = ((self.tokens.len() as f32) * val_frac) as usize;
        let split_at = self.tokens.len() - val_n;
        let val_tokens = self.tokens.split_off(split_at);
        (self, CorpusReader::new(val_tokens))
    }
}

impl BatchSource for CorpusReader {
    fn next_batch(&mut self, seq_len: usize) -> Option<(Vec<usize>, Vec<usize>)> {
        // Need seq_len inputs plus one shifted target token.
        if self.tokens.len() < seq_len + 1 {
            return None;
        }
        if self.pos + seq_len + 1 > self.tokens.len() {
            self.pos = 0;
        }
        let ids = self.tokens[self.pos..self.pos + seq_len].iter().map(|&t| t as usize).collect();
        let targets =
            self.tokens[self.pos + 1..self.pos + seq_len + 1].iter().map(|&t| t as usize).collect();
        self.pos += seq_len;
        Some((ids, targets))
    }
}

/// Default Hugging Face tokenizer repo (`tokenizer.json` is pulled from its root).
pub const DEFAULT_TOKENIZER_REPO: &str = "gpt2";
/// Default text dataset file on the hub (a small validation split for first runs).
pub const DEFAULT_DATASET_URL: &str =
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt";

fn http_get(url: &str) -> Result<Vec<u8>, String> {
    let resp = ureq::get(url).call().map_err(|e| format!("GET {url}: {e}"))?;
    let mut buf = Vec::new();
    resp.into_reader().read_to_end(&mut buf).map_err(|e| format!("read {url}: {e}"))?;
    Ok(buf)
}

/// Download `url` to `cache_path` unless it already exists (a simple on-disk cache so
/// repeated runs don't re-fetch the tokenizer/dataset).
fn cached_download(url: &str, cache_path: &Path) -> Result<(), String> {
    if cache_path.exists() {
        return Ok(());
    }
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let bytes = http_get(url)?;
    std::fs::write(cache_path, bytes).map_err(|e| e.to_string())
}

/// Encode `text` into token ids with a loaded tokenizer.
pub fn tokenize(tok: &Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    let enc = tok.encode(text, false).map_err(|e| e.to_string())?;
    Ok(enc.get_ids().to_vec())
}

/// Fetch a tokenizer + text dataset from the Hugging Face hub (cached under
/// `cache_dir`), tokenize at most `max_chars` of the text, and return the tokenizer
/// (its vocab size should drive `ModelConfig::vocab`) alongside a ready reader.
pub fn hf_corpus(
    tokenizer_repo: &str,
    dataset_url: &str,
    cache_dir: &Path,
    max_chars: Option<usize>,
) -> Result<(Tokenizer, CorpusReader), String> {
    let tok_url = format!("https://huggingface.co/{tokenizer_repo}/resolve/main/tokenizer.json");
    let tok_path = cache_dir.join("tokenizer.json");
    cached_download(&tok_url, &tok_path)?;
    let tok = Tokenizer::from_file(&tok_path).map_err(|e| e.to_string())?;

    let data_path = cache_dir.join("corpus.txt");
    cached_download(dataset_url, &data_path)?;
    let mut text = std::fs::read_to_string(&data_path).map_err(|e| e.to_string())?;
    if let Some(m) = max_chars {
        text.truncate(text.char_indices().nth(m).map_or(text.len(), |(i, _)| i));
    }

    let tokens = tokenize(&tok, &text)?;
    Ok((tok, CorpusReader::new(tokens)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::optimizer::AdaFactor;
    use crate::model::config::ModelConfig;
    use crate::model::model::Model;
    use crate::train::optim::Optimizer;
    use crate::train::r#loop::{TrainConfig, train};
    use crate::train::schedule::WsdSchedule;

    #[test]
    fn reader_yields_shifted_windows_and_wraps() {
        let mut r = CorpusReader::new((0u32..10).collect());
        let (ids, tgt) = r.next_batch(4).unwrap();
        assert_eq!(ids, vec![0, 1, 2, 3]);
        assert_eq!(tgt, vec![1, 2, 3, 4]);
        let (ids2, _) = r.next_batch(4).unwrap();
        assert_eq!(ids2, vec![4, 5, 6, 7]);
        // pos would be 8; 8 + 4 + 1 > 10 ⇒ wrap back to the start.
        let (ids3, _) = r.next_batch(4).unwrap();
        assert_eq!(ids3, vec![0, 1, 2, 3]);
    }

    #[test]
    fn too_short_corpus_yields_none() {
        let mut r = CorpusReader::new(vec![1, 2, 3]);
        assert!(r.next_batch(4).is_none());
    }

    fn tiny_cfg() -> ModelConfig {
        let mut c = ModelConfig::default();
        c.n_layers = 2;
        c.hidden = 8;
        c.n_q_heads = 2;
        c.n_kv_heads = 1;
        c.head_dim = 4;
        c.ffn_dim = 12;
        c.block = 4;
        c.n_active = 2;
        c.dict_k = 6;
        c.kv_block = 2;
        c.window = 3;
        c.full_attn_layers = 1;
        c.vocab = 16;
        c.max_seq = 16;
        c.validate();
        c
    }

    /// End-to-end (offline): a `CorpusReader` over synthetic tokens drives the real
    /// training loop, and cross-entropy descends — the full data→loop→optimizer path.
    #[test]
    fn corpus_reader_drives_training_loop() {
        let mut model = Model::new(tiny_cfg(), 0xDA7A);
        let af = AdaFactor { relative_step: false, ..AdaFactor::default() };
        let mut opt = Optimizer::with_config(&model, af);
        // A repeating token pattern (vocab 16) is highly learnable from windows.
        let pattern = [1u32, 5, 2, 9, 3, 0, 7, 4];
        let tokens: Vec<u32> = (0..256).map(|i| pattern[i % pattern.len()]).collect();
        let mut reader = CorpusReader::new(tokens);

        let cfg = TrainConfig {
            total_steps: 120,
            micro_batches: 2,
            seq_len_init: 8,
            seq_len_final: 8,
            curriculum_switch: 60,
            schedule: WsdSchedule::new(0.05, 0.0, 5, 30, 120),
            probe_weight: 0.1,
            probe_anneal_steps: 40,
        };

        let hist = train(&mut model, &mut opt, &mut reader, &cfg, None, 0, 0);
        assert_eq!(hist.len(), 120);
        let first = hist[0].ce;
        let last = hist.last().unwrap().ce;
        assert!(last.is_finite(), "ce diverged to {last}");
        assert!(last < first * 0.6, "ce did not descend through CorpusReader: {first} -> {last}");
    }
}
