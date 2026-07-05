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

/// Train a small BPE tokenizer directly on `corpus_text` (rather than loading one
/// pretrained on a much larger, generic vocabulary) -- right-sizes the vocab to
/// the actual corpus, which matters here beyond just tokenizer quality: `vocab`
/// directly drives the tied LM head's memory-bandwidth cost (it sweeps the full
/// `[vocab, hidden]` embedding table every step -- see RESEARCH_LOG.md
/// 2026-07-04/05), so a restricted-domain corpus (e.g. TinyStories' ~1,500-word
/// lexicon) with a right-sized vocab (e.g. 8192, vs. GPT-2's 50257) is a real,
/// free win, not just a tokenization-quality nicety.
///
/// Cached at `cache_path`: if a tokenizer already exists there, it's loaded
/// as-is (retraining is not idempotent-cheap) rather than retrained.
pub fn train_bpe_tokenizer(corpus_text: &str, vocab_size: usize, cache_path: &Path) -> Result<Tokenizer, String> {
    if cache_path.exists() {
        return Tokenizer::from_file(cache_path).map_err(|e| e.to_string());
    }
    use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
    use tokenizers::models::TrainerWrapper;
    use tokenizers::normalizers::NFC;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    let mut tokenizer = Tokenizer::new(BPE::default());
    tokenizer.with_normalizer(Some(NFC));
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));

    let mut trainer: TrainerWrapper = BpeTrainerBuilder::new()
        .vocab_size(vocab_size)
        .min_frequency(2)
        .special_tokens(vec![tokenizers::AddedToken::from("<unk>", true)])
        .build()
        .into();

    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    // `train_from_files` needs on-disk files; write the corpus to a temp file
    // beside the cache path instead of requiring the caller to have one.
    let tmp_path = cache_path.with_extension("corpus.tmp.txt");
    std::fs::write(&tmp_path, corpus_text).map_err(|e| e.to_string())?;
    tokenizer
        .train_from_files(&mut trainer, vec![tmp_path.to_string_lossy().to_string()])
        .map_err(|e| e.to_string())?;
    std::fs::remove_file(&tmp_path).ok();

    tokenizer.save(cache_path, false).map_err(|e| e.to_string())?;
    Ok(tokenizer)
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
    fn train_bpe_tokenizer_produces_working_tokenizer() {
        // A tiny repetitive corpus so BPE has something to merge, keeping the
        // requested vocab_size well above what's actually reachable (BPE
        // trainers cap out at base-alphabet + merges + specials, whichever
        // is smaller) -- this specifically checks training + round-trip
        // tokenization work, not that any particular vocab_size is hit.
        let corpus = "the cat sat on the mat. the cat ran. the dog sat on the mat too.".repeat(50);
        let dir = std::env::temp_dir().join(format!("fydel_tok_test_{}", std::process::id()));
        let cache_path = dir.join("tokenizer.json");
        let _ = std::fs::remove_file(&cache_path);

        let tok = train_bpe_tokenizer(&corpus, 64, &cache_path).expect("training failed");
        assert!(cache_path.exists(), "tokenizer should be cached to disk");
        assert!(tok.get_vocab_size(true) > 0);

        let ids = tokenize(&tok, "the cat sat").expect("tokenize failed");
        assert!(!ids.is_empty());

        // Loading again should hit the cache (no retrain) and produce the
        // same vocab.
        let tok2 = train_bpe_tokenizer(&corpus, 64, &cache_path).expect("cached load failed");
        assert_eq!(tok.get_vocab_size(true), tok2.get_vocab_size(true));

        std::fs::remove_file(&cache_path).ok();
    }

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
