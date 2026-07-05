//! Confirms (or refutes) the hypothesis that `BasisMatmul` is dominated by its many
//! small rustfft transforms. Measures the cost of one size-`b` FFT, then runs full
//! forward+backward on representative projection shapes and uses the FFT call counter
//! to estimate what fraction of the time is spent inside rustfft.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use fydel::kernels::fft::{BasisMatmul, Fft, fft_calls, fft_calls_reset};
use rustfft::num_complex::Complex32;

// Counting allocator: every allocation path bumps a global counter so the bench can
// isolate exactly how many allocations (and bytes) a single forward/backward incurs.
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new as u64, Ordering::Relaxed);
        unsafe { System.realloc(p, l, new) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static GA: Counting = Counting;

fn measure_allocs(label: &str, out: usize, in_: usize, b: usize, k: usize) {
    let mm = BasisMatmul::new(out, in_, b, k);
    let mut rng = Lcg(0x99 ^ (out as u64) ^ ((in_ as u64) << 20));
    let dict: Vec<Complex32> = (0..mm.dict_len()).map(|_| Complex32::new(rng.f(), rng.f())).collect();
    let coeffs: Vec<f32> = (0..mm.coeff_len()).map(|_| rng.f() * 0.2).collect();
    let x: Vec<f32> = (0..in_).map(|_| rng.f()).collect();
    let dy: Vec<f32> = (0..out).map(|_| rng.f()).collect();

    let (a0, b0) = (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed));
    std::hint::black_box(mm.forward(&dict, &coeffs, &x));
    let fa = ALLOCS.load(Ordering::Relaxed) - a0;
    let fb = BYTES.load(Ordering::Relaxed) - b0;

    let (a0, b0) = (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed));
    std::hint::black_box(mm.backward(&dict, &coeffs, &x, &dy));
    let ba = ALLOCS.load(Ordering::Relaxed) - a0;
    let bb = BYTES.load(Ordering::Relaxed) - b0;

    println!("{label:14} fwd {fa:4} allocs/{fb:6}B   bwd {ba:4} allocs/{bb:7}B");
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

/// Mean cost of a forward and an inverse size-`b` transform.
fn per_fft_cost(b: usize) -> (f64, f64) {
    let fft = Fft::new(b);
    let mut buf = vec![Complex32::new(1.0, 0.0); b];
    let m = 2_000_000u64;
    for _ in 0..2000 {
        fft.fft(&mut buf);
    }
    let t = Instant::now();
    for _ in 0..m {
        fft.fft(&mut buf);
    }
    let pf = t.elapsed().as_secs_f64() / m as f64;
    let t = Instant::now();
    for _ in 0..m {
        fft.ifft(&mut buf);
    }
    let pi = t.elapsed().as_secs_f64() / m as f64;
    (pf, pi)
}

/// Returns (forward us/op, backward us/op, FFTs per fwd+bwd, FFT% of total).
fn bench_split(out: usize, in_: usize, b: usize, k: usize, iters: u64, p_avg: f64) -> (f64, f64, f64, f64) {
    let mm = BasisMatmul::new(out, in_, b, k);
    let mut rng = Lcg(0xABCDEF ^ (out as u64) ^ ((in_ as u64) << 20) ^ ((k as u64) << 40));
    let dict: Vec<Complex32> = (0..mm.dict_len()).map(|_| Complex32::new(rng.f(), rng.f())).collect();
    let coeffs: Vec<f32> = (0..mm.coeff_len()).map(|_| rng.f() * 0.2).collect();
    let x: Vec<f32> = (0..in_).map(|_| rng.f()).collect();
    let dy: Vec<f32> = (0..out).map(|_| rng.f()).collect();

    for _ in 0..5 {
        std::hint::black_box(mm.forward(&dict, &coeffs, &x));
        std::hint::black_box(mm.backward(&dict, &coeffs, &x, &dy));
    }

    fft_calls_reset();
    let t = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(mm.forward(&dict, &coeffs, &x));
    }
    let fwd = t.elapsed().as_secs_f64() / iters as f64;
    let fwd_calls = fft_calls();

    fft_calls_reset();
    let t = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(mm.backward(&dict, &coeffs, &x, &dy));
    }
    let bwd = t.elapsed().as_secs_f64() / iters as f64;
    let bwd_calls = fft_calls();

    let calls_per_op = (fwd_calls + bwd_calls) as f64 / iters as f64;
    let fft_pct = calls_per_op * p_avg / (fwd + bwd) * 100.0;
    (fwd * 1e6, bwd * 1e6, calls_per_op, fft_pct)
}

fn main() {
    let b = 64;
    let (pf, pi) = per_fft_cost(b);
    let p_avg = (pf + pi) / 2.0;
    println!(
        "per size-{b} transform: fwd {:.1} ns, inv {:.1} ns (avg {:.1} ns)\n",
        pf * 1e9,
        pi * 1e9,
        p_avg * 1e9
    );

    let iters = 4000;
    // block_eigs is the ONLY dict_k-dependent cost. Sweeping k isolates it: the
    // k=1 row is the FFT+pointwise+alloc residual; the slope per +k is block_eigs.
    println!("attn 512x512  (sweep dict_k to isolate block_eigs):");
    println!("  {:>3}  {:>10}  {:>10}  {:>10}  {:>6}", "k", "fwd_us", "bwd_us", "tot_us", "FFT%");
    for k in [1usize, 8, 16, 32] {
        let (fwd, bwd, _calls, fft_pct) = bench_split(512, 512, b, k, iters, p_avg);
        println!("  {k:>3}  {fwd:>10.2}  {bwd:>10.2}  {:>10.2}  {fft_pct:>5.1}%", fwd + bwd);
    }
    println!("\nshapes at k=32 (fwd / bwd us/op):");
    for (lbl, o, i) in [("attn 512x512", 512, 512), ("ffn down", 512, 2048), ("ffn up", 2048, 512)] {
        let (fwd, bwd, calls, _) = bench_split(o, i, b, 32, iters, p_avg);
        println!("  {lbl:14} fwd {fwd:8.2}  bwd {bwd:8.2}  ({calls:.0} FFTs)");
    }

    println!("\nallocations per op (k=32):");
    measure_allocs("attn 512x512", 512, 512, b, 32);
    measure_allocs("ffn down", 512, 2048, b, 32);
    measure_allocs("ffn up", 2048, 512, b, 32);
}
