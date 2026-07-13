//! Cheap probe for RESEARCH_LOG.md's open AdaFactor design question: "Does
//! the frequency-domain second moment actually beat the spatial
//! factorization, or is the energy-compaction intuition wrong for
//! *gradients* (as opposed to weights/activations)?"
//!
//! AdaFactor's row/col factorization (optimizer.rs) is a rank-1 outer-product
//! approximation of the elementwise second moment G^2 (via row-mean / col-mean
//! sums). The conjecture: an FFT of the gradient compacts energy into fewer
//! coefficients, so the *same* row/col rank-1 factorization applied to the
//! power spectrum |FFT(G)|^2 would fit tighter than it does on G^2 directly.
//!
//! This probe measures exactly that, on REAL gradients from one real forward
//! + backward pass of the actual production model (not synthetic data) --
//! extracts a handful of real Monarch-coefficient gradient matrices at their
//! real (rows, cols) shapes, and reports the relative Frobenius
//! reconstruction error of the row/col rank-1 fit in both domains. Does NOT
//! attempt to build a working optimizer variant -- that requires solving how
//! to invert the frequency-domain estimate back to a valid per-element
//! second moment, a separate design problem this probe deliberately doesn't
//! touch. Answers the premise question first.

use fydel::model::config::ModelConfig;
use fydel::model::model::{Model, cross_entropy};
use num_complex::Complex32;
use rustfft::FftPlanner;

/// AdaFactor's exact row/col rank-1 factorization of `g_sq[i,j] = g[i,j]^2`,
/// mirroring `optimizer.rs::step`'s formula (eps1 omitted -- this probe
/// measures the factorization's own fit, not AdaFactor's numerical stability
/// epsilon).
fn rank1_relative_error(g_sq: &[f32], rows: usize, cols: usize) -> f32 {
    let mut r = vec![0.0f32; rows];
    let mut c = vec![0.0f32; cols];
    for i in 0..rows {
        let mut s = 0.0f32;
        for j in 0..cols {
            s += g_sq[i * cols + j];
        }
        r[i] = s / cols as f32;
    }
    for j in 0..cols {
        let mut s = 0.0f32;
        for i in 0..rows {
            s += g_sq[i * cols + j];
        }
        c[j] = s / rows as f32;
    }
    let r_sum: f32 = r.iter().sum();

    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for i in 0..rows {
        for j in 0..cols {
            let true_v = g_sq[i * cols + j];
            let approx = r[i] * c[j] / r_sum;
            num += (true_v - approx).powi(2);
            den += true_v.powi(2);
        }
    }
    (num / den).sqrt()
}

/// Separable 2D DFT (rows then cols) via rustfft, real input -> complex output.
fn fft2(g: &[f32], rows: usize, cols: usize) -> Vec<Complex32> {
    let mut planner = FftPlanner::<f32>::new();
    let fft_cols = planner.plan_fft_forward(cols);
    let fft_rows = planner.plan_fft_forward(rows);

    let mut buf: Vec<Complex32> = g.iter().map(|&x| Complex32::new(x, 0.0)).collect();
    // FFT along each row.
    for i in 0..rows {
        fft_cols.process(&mut buf[i * cols..(i + 1) * cols]);
    }
    // Transpose, FFT along each (former) column, transpose back.
    let mut t = vec![Complex32::new(0.0, 0.0); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = buf[i * cols + j];
        }
    }
    for j in 0..cols {
        fft_rows.process(&mut t[j * rows..(j + 1) * rows]);
    }
    for i in 0..rows {
        for j in 0..cols {
            buf[i * cols + j] = t[j * rows + i];
        }
    }
    buf
}

fn probe(name: &str, g: &[f32], rows: usize, cols: usize) {
    let g_sq: Vec<f32> = g.iter().map(|&x| x * x).collect();
    let spatial_err = rank1_relative_error(&g_sq, rows, cols);

    let spectrum = fft2(g, rows, cols);
    let power: Vec<f32> = spectrum.iter().map(|c| c.norm_sqr()).collect();
    let freq_err = rank1_relative_error(&power, rows, cols);

    let ratio = freq_err / spatial_err;
    println!(
        "{name:<18} shape=({rows:>5}x{cols:<3}) spatial_err={spatial_err:.4} freq_err={freq_err:.4} freq/spatial={ratio:.3}"
    );
}

fn main() {
    let mut cfg = ModelConfig::default();
    cfg.validate();
    let vocab = cfg.vocab;
    let seq = 64usize; // small enough for one fast fwd+bwd pass; shapes below are all production-scale (per-layer weight shapes are seq-independent)
    println!("model: {} layers, hidden {}, ffn {}, seq {}", cfg.n_layers, cfg.hidden, cfg.ffn_dim, seq);

    let mut model = Model::new(cfg, 0xF1DE1);
    let mut pool = fydel::kernels::scratch::BufPool::new();
    let ids: Vec<usize> = (0..seq).map(|i| i.wrapping_mul(2654435761) % vocab).collect();
    let targets: Vec<usize> = (0..seq).map(|i| (i + 1).wrapping_mul(2654435761) % vocab).collect();

    let fwd = model.forward(&ids, &mut pool);
    let (_, d_logits) = cross_entropy(&fwd.logits, vocab, &targets);
    let grads = model.backward(fwd, &d_logits, None, &mut pool);

    println!();
    println!("--- rank-1 factorization fit: spatial (AdaFactor's current formula) vs frequency-domain ---");
    println!("(freq/spatial < 1.0 means the FFT premise holds: the power spectrum is MORE row/col-separable than G^2 itself)");
    println!();

    let l0 = &grads.layers[0]; // layer 0: Full attention
    probe("wq", &l0.d_wq, 2 * 14 * 14 * 8, 32);
    probe("wo", &l0.d_wo, 2 * 14 * 14 * 8, 32);
    probe("ffn_up", &l0.d_up_coeffs, 2 * 48 * 14 * 8, 32);
    probe("ffn_down", &l0.d_down_coeffs, 2 * 14 * 48 * 8, 32);

    // A Sliding-attention layer's gradients too, for a second data point on a
    // structurally different attention path.
    let l_sliding = &grads.layers[cfg_full_attn_layers_hint()];
    probe("wq (sliding)", &l_sliding.d_wq, 2 * 14 * 14 * 8, 32);
}

fn cfg_full_attn_layers_hint() -> usize {
    ModelConfig::default().full_attn_layers
}
