// Second matmul-side probe: the real hot path isn't a single apply_block
// call, it's backward_block_phase1's batched outer-product accumulation
// (src/kernels/monarch.rs:1707-1732 docstring) — s1 += Σ_t dy1_t ⊗ x_t,
// s2 += Σ_t dout_t ⊗ z_t summed across every token in the batch, followed by
// ONE dictionary contraction at the end (this is the 2026-07-03 backward
// rewrite that made the per-token dictionary contraction go away).
//
// int16_matmul_probe.rs showed single-block error is tiny and non-compounding.
// The open question here is different: does per-token int16 quantization
// noise in the accumulated outer product *grow* relative to the true signal
// as more tokens are summed (bad — same shape of problem as the optimizer
// probes), or average out (good — independent per-token noise should shrink
// relative to a growing signal)? This is exactly the kind of thing that's
// cheap to falsify standalone and expensive to discover mid-training.

const M: usize = 8;
const B: usize = M * M;
const ND: usize = 8; // dict_k
const TOKENS: &[usize] = &[1, 8, 32, 128, 512, 2048];

fn fake_quant(values: &[f32]) -> Vec<f32> {
    let max_abs = values.iter().fold(1e-12f32, |m, &v| m.max(v.abs()));
    let scale = max_abs / 32767.0;
    values.iter().map(|&v| (v / scale).round().clamp(i16::MIN as f32, i16::MAX as f32) * scale).collect()
}

fn rand_vec(n: usize, seed: &mut u64, scale: f32) -> Vec<f32> {
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
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32 * scale
        })
        .collect()
}

fn relative_l2_error(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt();
    let den: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt().max(1e-12);
    num / den
}

/// One token's contribution to s1 (dy1 ⊗ x, m×m) and s2 (dout ⊗ z, m×m),
/// optionally through int16 fake-quant on the per-token operands (eff1/eff2
/// stay fixed for the whole batch — a step's weights — so they're quantized
/// once outside this function).
fn token_outer_products(dy1: &[f32], x: &[f32], dout: &[f32], z: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut s1 = vec![0.0f32; B];
    let mut s2 = vec![0.0f32; B];
    for i in 0..M {
        for c in 0..M {
            s1[i * M + c] += dy1[i] * x[c];
        }
    }
    for j in 0..M {
        for c in 0..M {
            s2[j * M + c] += dout[j] * z[c];
        }
    }
    (s1, s2)
}

/// Single end-of-batch dictionary contraction against accumulated s1:
/// g_dd1[d*m+r, c] += a1_typical * s1[r,c] summed appropriately — simplified
/// to the actual shape that matters for this probe: the dictionary itself
/// (d1) is what's quantized once per step, contracted against the (fp32 or
/// quantized) accumulated s1.
fn contract(s1: &[f32], d_scale_probe: bool) -> Vec<f32> {
    // Represents g_da[d] = dot(d_row_d, s1_row) style contraction, collapsed
    // to a single dict entry per row for this probe (nd copies, same math).
    let mut out = vec![0.0f32; ND * M];
    let s1_use: Vec<f32> = if d_scale_probe { fake_quant(s1) } else { s1.to_vec() };
    for d in 0..ND {
        for r in 0..M {
            let mut acc = 0.0f32;
            for c in 0..M {
                acc += s1_use[r * M + c] * (1.0 + 0.1 * d as f32); // stand-in dict row
            }
            out[d * M + r] = acc;
        }
    }
    out
}

fn main() {
    println!("Does per-token int16 quantization noise grow or shrink relative to signal as tokens accumulate?\n");
    println!("{:>8} | {:>14} | {:>14}", "tokens", "rel_err s1", "rel_err g_dd (post-contract)");

    let mut seed: u64 = 0xD1B54A32D192ED03;
    for &t in TOKENS {
        let mut s1_ref = vec![0.0f32; B];
        let mut s2_ref = vec![0.0f32; B];
        let mut s1_q = vec![0.0f32; B];
        let mut s2_q = vec![0.0f32; B];

        for _ in 0..t {
            // Per-token activations, quantized fresh each token (realistic:
            // activations vary per token in a real batch).
            let dy1 = rand_vec(M, &mut seed, 0.3);
            let x = rand_vec(M, &mut seed, 1.0);
            let dout = rand_vec(M, &mut seed, 0.3);
            let z = rand_vec(M, &mut seed, 1.0);

            let (s1_t, s2_t) = token_outer_products(&dy1, &x, &dout, &z);
            for i in 0..B {
                s1_ref[i] += s1_t[i];
                s2_ref[i] += s2_t[i];
            }

            let dy1_q = fake_quant(&dy1);
            let x_q = fake_quant(&x);
            let dout_q = fake_quant(&dout);
            let z_q = fake_quant(&z);
            let (s1_tq, s2_tq) = token_outer_products(&dy1_q, &x_q, &dout_q, &z_q);
            for i in 0..B {
                s1_q[i] += s1_tq[i];
                s2_q[i] += s2_tq[i];
            }
        }

        let err_s1 = relative_l2_error(&s1_q, &s1_ref);
        let g_dd_ref = contract(&s1_ref, false);
        let g_dd_q = contract(&s1_q, true); // dictionary itself also quantized at contract time
        let err_gdd = relative_l2_error(&g_dd_q, &g_dd_ref);

        println!("{t:>8} | {err_s1:>14.2e} | {err_gdd:>14.2e}");
    }

    println!("\nIf error shrinks as tokens grow: independent per-token quantization noise averages out — safe to accumulate in fp32 after per-token int16 quantize.");
    println!("If error grows or plateaus high: quantization noise is correlated/biased and compounds — needs per-token rescaling or higher-precision accumulation.");
}
