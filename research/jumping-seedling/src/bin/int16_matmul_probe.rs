// Falsification probe for the "safe path" of the int16 plan: fake-quantize
// weights/activations to int16 per-call (fresh scale each time, nothing
// carried across steps — unlike the optimizer-state probes), matmul with
// int32-equivalent accumulation, rescale, dequantize. This is the
// well-trodden QAT pattern; the open question is just how much accuracy it
// costs on this codebase's actual hot shape.
//
// Mirrors `kernels::monarch::apply_block` (src/kernels/monarch.rs:1599)
// almost exactly — m=8 block size (the AVX2 lane width used throughout that
// file) — since that's the forward half of the ~72%-of-backward Monarch path
// identified as the actual throughput target.

const M: usize = 8; // block size, matches the AVX2 m==8 fast path
const B: usize = M * M;
const TRIALS: usize = 500;

/// Symmetric per-tensor int16 fake-quant: value -> round(value/scale) clamped
/// to i16 range -> back to f32 * scale. `scale` is fresh from this call's data
/// (the QAT-standard "activations/weights requantized every forward" pattern,
/// not a scale carried across steps the way the optimizer-state probes were).
fn fake_quant(values: &[f32]) -> Vec<f32> {
    let max_abs = values.iter().fold(1e-12f32, |m, &v| m.max(v.abs()));
    let scale = max_abs / 32767.0;
    values
        .iter()
        .map(|&v| {
            let q = (v / scale).round().clamp(i16::MIN as f32, i16::MAX as f32);
            q * scale
        })
        .collect()
}

/// Exact reimplementation of `apply_block`'s fp32 math (see monarch.rs:1599),
/// parameterized so it can be called with either raw or fake-quantized inputs.
fn apply_block(eff1: &[f32], eff2: &[f32], x_blk: &[f32]) -> Vec<f32> {
    let mut y1 = vec![0.0f32; M * M];
    let mut z = vec![0.0f32; M * M];
    let mut out = vec![0.0f32; M * M];
    for i in 0..M {
        let xi = &x_blk[i * M..(i + 1) * M];
        let e = &eff1[i * B..(i + 1) * B];
        for r in 0..M {
            let mut acc = 0.0f32;
            for c in 0..M {
                acc += e[r * M + c] * xi[c];
            }
            y1[i * M + r] = acc;
        }
    }
    for i in 0..M {
        for j in 0..M {
            z[j * M + i] = y1[i * M + j];
        }
    }
    for j in 0..M {
        let zj = &z[j * M..(j + 1) * M];
        let e = &eff2[j * B..(j + 1) * B];
        for r in 0..M {
            let mut acc = 0.0f32;
            for c in 0..M {
                acc += e[r * M + c] * zj[c];
            }
            out[j * M + r] += acc;
        }
    }
    out
}

fn relative_l2_error(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt();
    let den: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt().max(1e-12);
    num / den
}

fn rand_vec(n: usize, seed: &mut u64, scale: f32, outlier_prob: f32) -> Vec<f32> {
    let mut next = || {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let u1 = next().max(1e-12);
            let u2 = next();
            let mag = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let mut v = mag as f32 * scale;
            // Occasionally inject a heavy-tailed outlier, since real
            // activations aren't perfectly Gaussian and outliers are what
            // blow up a single shared quantization scale.
            if next() < outlier_prob as f64 {
                v *= 15.0;
            }
            v
        })
        .collect()
}

fn run_condition(label: &str, weight_scale: f32, act_scale: f32, weight_outlier_prob: f32, act_outlier_prob: f32) {
    let mut seed: u64 = 0x243F6A8885A308D3;
    let mut errs_out = Vec::with_capacity(TRIALS);
    let mut errs_y1 = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let eff1 = rand_vec(M * B, &mut seed, weight_scale, weight_outlier_prob);
        let eff2 = rand_vec(M * B, &mut seed, weight_scale, weight_outlier_prob);
        let x_blk = rand_vec(M * M, &mut seed, act_scale, act_outlier_prob);

        let out_ref = apply_block(&eff1, &eff2, &x_blk);

        let eff1_q = fake_quant(&eff1);
        let eff2_q = fake_quant(&eff2);
        let x_q = fake_quant(&x_blk);
        let out_q = apply_block(&eff1_q, &eff2_q, &x_q);

        errs_out.push(relative_l2_error(&out_q, &out_ref));

        // Also isolate just the first matmul stage's error (y1), to see how
        // much error compounds through the second stage vs. originates fresh.
        let y1_ref = {
            let mut y1 = vec![0.0f32; M * M];
            for i in 0..M {
                let xi = &x_blk[i * M..(i + 1) * M];
                let e = &eff1[i * B..(i + 1) * B];
                for r in 0..M {
                    y1[i * M + r] = (0..M).map(|c| e[r * M + c] * xi[c]).sum();
                }
            }
            y1
        };
        let y1_q = {
            let mut y1 = vec![0.0f32; M * M];
            for i in 0..M {
                let xi = &x_q[i * M..(i + 1) * M];
                let e = &eff1_q[i * B..(i + 1) * B];
                for r in 0..M {
                    y1[i * M + r] = (0..M).map(|c| e[r * M + c] * xi[c]).sum();
                }
            }
            y1
        };
        errs_y1.push(relative_l2_error(&y1_q, &y1_ref));
    }
    errs_out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    errs_y1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |v: &[f32], p: f32| v[((v.len() - 1) as f32 * p) as usize];
    println!(
        "{label:38} | y1 median={:.2e} p99={:.2e}  | out median={:.2e} p99={:.2e} max={:.2e}",
        pct(&errs_y1, 0.5), pct(&errs_y1, 0.99),
        pct(&errs_out, 0.5), pct(&errs_out, 0.99), errs_out.last().unwrap()
    );
}

fn main() {
    println!("int16 fake-quant matmul error, {M}x{M} block (apply_block shape), {TRIALS} trials\n");
    run_condition("clean gaussian (typical)", 0.15, 1.0, 0.0, 0.0);
    run_condition("clean gaussian, small weights", 0.02, 1.0, 0.0, 0.0);
    run_condition("5% heavy-tail outliers in activations", 0.15, 1.0, 0.0, 0.05);
    run_condition("5% heavy-tail outliers in weights", 0.15, 1.0, 0.05, 0.0);
    run_condition("outliers in both", 0.15, 1.0, 0.05, 0.05);
}
