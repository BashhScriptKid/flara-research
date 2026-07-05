//! Warmup-Stable-Decay (WSD) learning-rate schedule.
//!
//! Linear warmup from 0 to the peak LR over `warmup` steps, a flat plateau at the
//! peak, then a linear decay to `min_lr` over the final `decay` steps. Unlike
//! cosine, the long flat plateau lets the curriculum (seq 512→1024) and batch size
//! change without the LR drifting underneath them, and the schedule can be extended
//! by moving `total` without re-running warmup.

/// A WSD schedule over a fixed horizon of `total` steps.
#[derive(Clone, Copy, Debug)]
pub struct WsdSchedule {
    /// Learning rate at the plateau.
    pub peak_lr: f32,
    /// Floor the decay descends to (often 0, or a small fraction of `peak_lr`).
    pub min_lr: f32,
    /// Number of warmup steps (linear 0 → `peak_lr`).
    pub warmup: usize,
    /// Number of decay steps at the tail (linear `peak_lr` → `min_lr`).
    pub decay: usize,
    /// Total horizon; decay occupies `[total - decay, total)`.
    pub total: usize,
}

impl WsdSchedule {
    /// Build a schedule, clamping so warmup and decay always fit within `total`.
    pub fn new(peak_lr: f32, min_lr: f32, warmup: usize, decay: usize, total: usize) -> Self {
        let warmup = warmup.min(total);
        let decay = decay.min(total - warmup);
        Self { peak_lr, min_lr, warmup, decay, total }
    }

    /// The first step index at which decay begins.
    pub fn decay_start(&self) -> usize {
        self.total - self.decay
    }

    /// Learning rate at `step` (0-indexed). Steps at or beyond `total` return `min_lr`.
    pub fn lr(&self, step: usize) -> f32 {
        if step < self.warmup {
            // Linear warmup; +1 so step 0 is a nonzero nudge rather than a dead step.
            return self.peak_lr * (step as f32 + 1.0) / (self.warmup as f32 + 1.0);
        }
        let decay_start = self.decay_start();
        if step < decay_start {
            return self.peak_lr;
        }
        if step >= self.total || self.decay == 0 {
            return self.min_lr;
        }
        let frac = (step - decay_start) as f32 / self.decay as f32;
        self.peak_lr + (self.min_lr - self.peak_lr) * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sched() -> WsdSchedule {
        WsdSchedule::new(1.0, 0.0, 100, 200, 1000)
    }

    #[test]
    fn phases_have_expected_endpoints() {
        let s = sched();
        assert!(s.lr(0) > 0.0 && s.lr(0) < 0.02, "warmup starts small: {}", s.lr(0));
        assert!((s.lr(99) - 1.0).abs() < 0.02, "warmup ends near peak: {}", s.lr(99));
        assert_eq!(s.lr(100), 1.0, "plateau at peak");
        assert_eq!(s.lr(500), 1.0, "mid plateau at peak");
        assert_eq!(s.lr(800), 1.0, "plateau holds until decay_start");
        assert!((s.lr(900) - 0.5).abs() < 1e-6, "halfway through decay: {}", s.lr(900));
        assert!(s.lr(1000) <= 1e-6, "decayed to min: {}", s.lr(1000));
        assert!(s.lr(5000) <= 1e-6, "stays at min past horizon");
    }

    #[test]
    fn warmup_and_decay_are_monotone() {
        let s = sched();
        for step in 0..s.warmup {
            assert!(s.lr(step) <= s.lr(step + 1), "warmup not increasing at {step}");
        }
        for step in s.decay_start()..s.total {
            assert!(s.lr(step) >= s.lr(step + 1), "decay not decreasing at {step}");
        }
    }

    #[test]
    fn oversized_phases_are_clamped() {
        // warmup + decay exceed total: must not panic or overflow.
        let s = WsdSchedule::new(1.0, 0.0, 800, 800, 1000);
        assert_eq!(s.warmup, 800);
        assert_eq!(s.decay, 200);
        let _ = s.lr(0);
        let _ = s.lr(999);
    }
}
