//! Fast, vectorized exp/sigmoid for the SwiGLU activation hot loop.
//!
//! Cephes-derived polynomial exp (same one used in avx_mathfun / many SIMD
//! math libs): range-reduce via x = n*ln2 + r, r in [-ln2/2, ln2/2], compute
//! 2^n by directly packing the float exponent bits, and approximate e^r with
//! a degree-5 minimax polynomial. ~1e-6 relative error, no libm call.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::*;

const EXP_HI: f32 = 88.376_26;
const EXP_LO: f32 = -88.376_26;
const LOG2EF: f32 = 1.442_695_1;
const EXP_C1: f32 = 0.693_359_38;
const EXP_C2: f32 = -2.121_944_4e-4;
const EXP_P0: f32 = 1.987_569_15e-4;
const EXP_P1: f32 = 1.398_199_95e-3;
const EXP_P2: f32 = 8.333_451_9e-3;
const EXP_P3: f32 = 4.166_579_6e-2;
const EXP_P4: f32 = 1.666_666_55e-1;
const EXP_P5: f32 = 5.000_000_1e-1;

#[inline]
pub fn fast_exp(x: f32) -> f32 {
    let x = x.clamp(EXP_LO, EXP_HI);
    let fx = (x * LOG2EF + 0.5).floor();
    let x = x - fx * EXP_C1;
    let x = x - fx * EXP_C2;
    let z = x * x;
    let mut y = EXP_P0;
    y = y * x + EXP_P1;
    y = y * x + EXP_P2;
    y = y * x + EXP_P3;
    y = y * x + EXP_P4;
    y = y * x + EXP_P5;
    y = y * z + x + 1.0;
    let n = fx as i32;
    let pow2n = f32::from_bits(((n + 127) as u32) << 23);
    y * pow2n
}

#[inline]
pub fn fast_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-x))
}

/// act[i] = up[i] * gate[i] * sigmoid(gate[i])   (SwiGLU forward)
pub fn swiglu_forward(up: &[f32], gate: &[f32], act: &mut [f32]) {
    let n = up.len();
    debug_assert_eq!(gate.len(), n);
    debug_assert_eq!(act.len(), n);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { swiglu_forward_avx2(up, gate, act) };
        return;
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    for i in 0..n {
        act[i] = up[i] * gate[i] * fast_sigmoid(gate[i]);
    }
}

/// d_up[i]   = d_act[i] * gate[i] * sigmoid(gate[i])
/// d_gate[i] = d_act[i] * up[i] * sigmoid(gate[i]) * (1 + gate[i]*(1 - sigmoid(gate[i])))
pub fn swiglu_backward(
    up: &[f32], gate: &[f32], d_act: &[f32],
    d_up: &mut [f32], d_gate: &mut [f32],
) {
    let n = up.len();
    debug_assert_eq!(gate.len(), n);
    debug_assert_eq!(d_act.len(), n);
    debug_assert_eq!(d_up.len(), n);
    debug_assert_eq!(d_gate.len(), n);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { swiglu_backward_avx2(up, gate, d_act, d_up, d_gate) };
        return;
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    for i in 0..n {
        let g = gate[i];
        let sig = fast_sigmoid(g);
        d_up[i] = d_act[i] * g * sig;
        d_gate[i] = d_act[i] * up[i] * sig * (1.0 + g * (1.0 - sig));
    }
}

/// act[i] = up[i] * sigmoid(gate[i])   (GLU forward, used by the dense baseline)
pub fn glu_forward(up: &[f32], gate: &[f32], act: &mut [f32]) {
    let n = up.len();
    debug_assert_eq!(gate.len(), n);
    debug_assert_eq!(act.len(), n);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { glu_forward_avx2(up, gate, act) };
        return;
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    for i in 0..n {
        act[i] = up[i] * fast_sigmoid(gate[i]);
    }
}

/// d_up[i]   = sigmoid(gate[i]) * d_act[i]
/// d_gate[i] = up[i] * sigmoid(gate[i]) * (1 - sigmoid(gate[i])) * d_act[i]
pub fn glu_backward(
    up: &[f32], gate: &[f32], d_act: &[f32],
    d_up: &mut [f32], d_gate: &mut [f32],
) {
    let n = up.len();
    debug_assert_eq!(gate.len(), n);
    debug_assert_eq!(d_act.len(), n);
    debug_assert_eq!(d_up.len(), n);
    debug_assert_eq!(d_gate.len(), n);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { glu_backward_avx2(up, gate, d_act, d_up, d_gate) };
        return;
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    for i in 0..n {
        let sig = fast_sigmoid(gate[i]);
        d_up[i] = sig * d_act[i];
        d_gate[i] = up[i] * sig * (1.0 - sig) * d_act[i];
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn fast_exp_avx2(x: __m256) -> __m256 {
    let hi = _mm256_set1_ps(EXP_HI);
    let lo = _mm256_set1_ps(EXP_LO);
    let x = _mm256_min_ps(x, hi);
    let x = _mm256_max_ps(x, lo);

    let log2ef = _mm256_set1_ps(LOG2EF);
    let half = _mm256_set1_ps(0.5);
    let fx = _mm256_floor_ps(_mm256_fmadd_ps(x, log2ef, half));

    let c1 = _mm256_set1_ps(EXP_C1);
    let c2 = _mm256_set1_ps(EXP_C2);
    let x = _mm256_fnmadd_ps(fx, c1, x);
    let x = _mm256_fnmadd_ps(fx, c2, x);

    let z = _mm256_mul_ps(x, x);
    let mut y = _mm256_set1_ps(EXP_P0);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(EXP_P1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(EXP_P2));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(EXP_P3));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(EXP_P4));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(EXP_P5));
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0));

    let n = _mm256_cvttps_epi32(fx);
    let n = _mm256_add_epi32(n, _mm256_set1_epi32(127));
    let n = _mm256_slli_epi32(n, 23);
    let pow2n = _mm256_castsi256_ps(n);

    _mm256_mul_ps(y, pow2n)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_avx2(x: __m256) -> __m256 {
    let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    let e = fast_exp_avx2(neg_x);
    let one = _mm256_set1_ps(1.0);
    _mm256_div_ps(one, _mm256_add_ps(one, e))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_forward_avx2(up: &[f32], gate: &[f32], act: &mut [f32]) {
    let n = up.len();
    let chunks = n / 8;
    for c in 0..chunks {
        let o = c * 8;
        let u = _mm256_loadu_ps(up.as_ptr().add(o));
        let g = _mm256_loadu_ps(gate.as_ptr().add(o));
        let sig = sigmoid_avx2(g);
        let out = _mm256_mul_ps(_mm256_mul_ps(u, g), sig);
        _mm256_storeu_ps(act.as_mut_ptr().add(o), out);
    }
    for i in (chunks * 8)..n {
        act[i] = up[i] * gate[i] * fast_sigmoid(gate[i]);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_backward_avx2(
    up: &[f32], gate: &[f32], d_act: &[f32],
    d_up: &mut [f32], d_gate: &mut [f32],
) {
    let n = up.len();
    let chunks = n / 8;
    let one = _mm256_set1_ps(1.0);
    for c in 0..chunks {
        let o = c * 8;
        let u = _mm256_loadu_ps(up.as_ptr().add(o));
        let g = _mm256_loadu_ps(gate.as_ptr().add(o));
        let da = _mm256_loadu_ps(d_act.as_ptr().add(o));
        let sig = sigmoid_avx2(g);
        let du = _mm256_mul_ps(_mm256_mul_ps(da, g), sig);
        let one_minus_sig = _mm256_sub_ps(one, sig);
        let inner = _mm256_fmadd_ps(g, one_minus_sig, one);
        let dg = _mm256_mul_ps(_mm256_mul_ps(da, u), _mm256_mul_ps(sig, inner));
        _mm256_storeu_ps(d_up.as_mut_ptr().add(o), du);
        _mm256_storeu_ps(d_gate.as_mut_ptr().add(o), dg);
    }
    for i in (chunks * 8)..n {
        let g = gate[i];
        let sig = fast_sigmoid(g);
        d_up[i] = d_act[i] * g * sig;
        d_gate[i] = d_act[i] * up[i] * sig * (1.0 + g * (1.0 - sig));
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn glu_forward_avx2(up: &[f32], gate: &[f32], act: &mut [f32]) {
    let n = up.len();
    let chunks = n / 8;
    for c in 0..chunks {
        let o = c * 8;
        let u = _mm256_loadu_ps(up.as_ptr().add(o));
        let g = _mm256_loadu_ps(gate.as_ptr().add(o));
        let sig = sigmoid_avx2(g);
        _mm256_storeu_ps(act.as_mut_ptr().add(o), _mm256_mul_ps(u, sig));
    }
    for i in (chunks * 8)..n {
        act[i] = up[i] * fast_sigmoid(gate[i]);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn glu_backward_avx2(
    up: &[f32], gate: &[f32], d_act: &[f32],
    d_up: &mut [f32], d_gate: &mut [f32],
) {
    let n = up.len();
    let chunks = n / 8;
    let one = _mm256_set1_ps(1.0);
    for c in 0..chunks {
        let o = c * 8;
        let u = _mm256_loadu_ps(up.as_ptr().add(o));
        let g = _mm256_loadu_ps(gate.as_ptr().add(o));
        let da = _mm256_loadu_ps(d_act.as_ptr().add(o));
        let sig = sigmoid_avx2(g);
        _mm256_storeu_ps(d_up.as_mut_ptr().add(o), _mm256_mul_ps(sig, da));
        let one_minus_sig = _mm256_sub_ps(one, sig);
        let dg = _mm256_mul_ps(_mm256_mul_ps(u, sig), _mm256_mul_ps(one_minus_sig, da));
        _mm256_storeu_ps(d_gate.as_mut_ptr().add(o), dg);
    }
    for i in (chunks * 8)..n {
        let sig = fast_sigmoid(gate[i]);
        d_up[i] = sig * d_act[i];
        d_gate[i] = up[i] * sig * (1.0 - sig) * d_act[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_exp_matches_std_exp() {
        for i in -50..50 {
            let x = i as f32 * 0.37;
            let got = fast_exp(x);
            let want = x.exp();
            let rel_err = ((got - want) / want).abs();
            assert!(rel_err < 1e-5, "x={x} got={got} want={want} rel_err={rel_err}");
        }
    }

    #[test]
    fn swiglu_forward_matches_reference() {
        let n = 37; // deliberately not a multiple of 8 to exercise the tail
        let up: Vec<f32> = (0..n).map(|i| (i as f32 - 18.0) * 0.13).collect();
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 10.0) * 0.21).collect();
        let mut act = vec![0.0f32; n];
        swiglu_forward(&up, &gate, &mut act);
        for i in 0..n {
            let want = up[i] * gate[i] / (1.0 + (-gate[i]).exp());
            assert!((act[i] - want).abs() < 1e-4, "i={i} got={} want={want}", act[i]);
        }
    }

    #[test]
    fn swiglu_backward_matches_reference() {
        let n = 37;
        let up: Vec<f32> = (0..n).map(|i| (i as f32 - 18.0) * 0.13).collect();
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 10.0) * 0.21).collect();
        let d_act: Vec<f32> = (0..n).map(|i| (i as f32 - 5.0) * 0.05).collect();
        let mut d_up = vec![0.0f32; n];
        let mut d_gate = vec![0.0f32; n];
        swiglu_backward(&up, &gate, &d_act, &mut d_up, &mut d_gate);
        for i in 0..n {
            let g = gate[i];
            let sig = 1.0 / (1.0 + (-g).exp());
            let want_du = d_act[i] * g * sig;
            let want_dg = d_act[i] * up[i] * sig * (1.0 + g * (1.0 - sig));
            assert!((d_up[i] - want_du).abs() < 1e-4, "d_up i={i} got={} want={want_du}", d_up[i]);
            assert!((d_gate[i] - want_dg).abs() < 1e-4, "d_gate i={i} got={} want={want_dg}", d_gate[i]);
        }
    }

    #[test]
    fn glu_forward_matches_reference() {
        let n = 37;
        let up: Vec<f32> = (0..n).map(|i| (i as f32 - 18.0) * 0.13).collect();
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 10.0) * 0.21).collect();
        let mut act = vec![0.0f32; n];
        glu_forward(&up, &gate, &mut act);
        for i in 0..n {
            let want = up[i] / (1.0 + (-gate[i]).exp());
            assert!((act[i] - want).abs() < 1e-4, "i={i} got={} want={want}", act[i]);
        }
    }

    #[test]
    fn glu_backward_matches_reference() {
        let n = 37;
        let up: Vec<f32> = (0..n).map(|i| (i as f32 - 18.0) * 0.13).collect();
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 10.0) * 0.21).collect();
        let d_act: Vec<f32> = (0..n).map(|i| (i as f32 - 5.0) * 0.05).collect();
        let mut d_up = vec![0.0f32; n];
        let mut d_gate = vec![0.0f32; n];
        glu_backward(&up, &gate, &d_act, &mut d_up, &mut d_gate);
        for i in 0..n {
            let sig = 1.0 / (1.0 + (-gate[i]).exp());
            let want_du = sig * d_act[i];
            let want_dg = up[i] * sig * (1.0 - sig) * d_act[i];
            assert!((d_up[i] - want_du).abs() < 1e-4, "d_up i={i} got={} want={want_du}", d_up[i]);
            assert!((d_gate[i] - want_dg).abs() < 1e-4, "d_gate i={i} got={} want={want_dg}", d_gate[i]);
        }
    }
}
