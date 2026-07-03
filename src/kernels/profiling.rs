//! Cheap in-process wall-clock instrumentation for `TransformerLayer::forward`/
//! `backward`'s named sub-blocks. Always-on (a handful of `Instant::now()` calls
//! per layer per step is noise-level overhead against the work being measured);
//! not a general profiling framework — just enough to answer "where does a
//! training step's time go" without a sampling profiler. See
//! `src/bin/profile.rs` for a caller that prints `report()`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

macro_rules! counters {
    ($($name:ident),+ $(,)?) => {
        $(pub static $name: AtomicU64 = AtomicU64::new(0);)+
        const ALL: &[(&str, &AtomicU64)] = &[$((stringify!($name), &$name)),+];
    };
}

counters!(
    QKV_FWD, QKV_BWD,
    WO_FWD, WO_BWD,
    ATTN_CORE_FWD, ATTN_CORE_BWD,
    FFN_SELECT, FFN_FWD, FFN_BWD,
    NORM_FWD, NORM_BWD,
);

/// RAII timer: accumulates elapsed wall time into `counter` on drop.
pub struct Timer(Instant, &'static AtomicU64);

impl Timer {
    #[inline]
    pub fn start(counter: &'static AtomicU64) -> Self {
        Timer(Instant::now(), counter)
    }
}

impl Drop for Timer {
    #[inline]
    fn drop(&mut self) {
        self.1.fetch_add(self.0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Print accumulated totals (ms) for every named sub-block, sorted descending.
pub fn report() {
    let mut rows: Vec<(&str, f64)> =
        ALL.iter().map(|(name, c)| (*name, c.load(Ordering::Relaxed) as f64 / 1e6)).collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let total: f64 = rows.iter().map(|(_, ms)| ms).sum();
    eprintln!("--- sub-block breakdown (ms, summed over the run) ---");
    for (name, ms) in &rows {
        eprintln!("{name:<14} {ms:10.2}  ({:4.1}%)", ms / total * 100.0);
    }
    eprintln!("{:<14} {total:10.2}", "sum");
}

/// Zero every counter (call before a measurement run, after warmup).
pub fn reset() {
    for (_, c) in ALL {
        c.store(0, Ordering::Relaxed);
    }
}
