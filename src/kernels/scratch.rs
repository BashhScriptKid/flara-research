//! A recycling pool of `f32` buffers, threaded explicitly through the
//! forward/backward call chain to avoid the repeated allocator round-trips
//! that come from freshly `vec![0.0; n]`-allocating the same handful of
//! shapes every training step (RESEARCH_LOG.md 2026-07-04: measured at
//! roughly 8-9% of a training step at toy scale).
//!
//! Buffers like `SharedMonarchProj`'s `zs` cache genuinely need to survive
//! from one layer's forward call until that same layer's backward call --
//! potentially many other layers' worth of work later. A literal borrowed
//! `&'a [f32]` tied to that lifetime would need either `unsafe` pointers or
//! `RefCell`-style runtime-checked borrows to express "mutate now, read
//! later, guaranteed non-overlapping" in safe Rust. This pool sidesteps that
//! entirely: buffers stay fully owned (`Vec<f32>`, moved around exactly as
//! before) -- the only change is where a fresh buffer's memory comes from.
//! `take_*` pulls a same-length buffer back out of the free list if one was
//! previously `give`n back (same cost as `Vec::resize`/`clear`, no syscall);
//! `give` returns a buffer for the next call to reuse. Every model shape is
//! fixed for the model's lifetime, so after a short warmup the free list
//! converges to exactly the working set of distinct shapes in use and every
//! subsequent `take` is a pure reuse, no allocation.
//!
//! Single-threaded by design: the *outer* call sequence (one projection
//! after another within a layer, one layer after another within a step) is
//! always sequential on one thread -- the only concurrency in this codebase
//! is rayon's *internal* token-parallelism inside one monarch call, which
//! always completes before that call returns. A `&mut BufPool` threaded
//! through the outer sequential chain is therefore sufficient; no locking
//! needed.

#[derive(Default)]
pub struct BufPool {
    free: Vec<Vec<f32>>,
    /// Separate free list for fp16-stored buffers (e.g. Monarch's `zs`
    /// cache, fp16-migration branch RESEARCH_LOG.md 2026-07-05) -- kept
    /// apart from `free` since an `f32` and an `half::f16` buffer of the
    /// same element count aren't interchangeable (different byte sizes).
    free16: Vec<Vec<half::f16>>,
}

impl BufPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// A buffer of exactly `len` elements, zero-initialized -- for
    /// accumulator buffers (`dx`, `s1`/`s2`) that get `+=`'d into.
    pub fn take_zeroed(&mut self, len: usize) -> Vec<f32> {
        let mut buf = self.take_raw(len);
        buf.iter_mut().for_each(|x| *x = 0.0);
        buf
    }

    /// A buffer of exactly `len` elements, contents unspecified -- for
    /// write-before-read buffers (e.g. `zs`, `y`) where every element gets
    /// overwritten before it's read, so zeroing on take would be wasted work.
    pub fn take_uninit(&mut self, len: usize) -> Vec<f32> {
        self.take_raw(len)
    }

    fn take_raw(&mut self, len: usize) -> Vec<f32> {
        if let Some(pos) = self.free.iter().position(|b| b.len() == len) {
            self.free.swap_remove(pos)
        } else {
            vec![0.0f32; len]
        }
    }

    /// Return a buffer for a future `take_*` call to reuse. Safe to call
    /// with any buffer (including ones this pool never handed out) -- it's
    /// just added to the free list.
    pub fn give(&mut self, buf: Vec<f32>) {
        self.free.push(buf);
    }

    /// fp16 equivalent of `take_uninit` -- write-before-read buffers only
    /// (no `take_f16_zeroed`: nothing fp16-stored in this codebase is an
    /// accumulator, per `f16_simd`'s module doc).
    pub fn take_f16_uninit(&mut self, len: usize) -> Vec<half::f16> {
        if let Some(pos) = self.free16.iter().position(|b| b.len() == len) {
            self.free16.swap_remove(pos)
        } else {
            vec![half::f16::from_f32(0.0); len]
        }
    }

    /// fp16 equivalent of `give`.
    pub fn give_f16(&mut self, buf: Vec<half::f16>) {
        self.free16.push(buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_after_give_reuses_the_same_allocation() {
        let mut pool = BufPool::new();
        let buf = pool.take_zeroed(1024);
        let ptr_before = buf.as_ptr();
        pool.give(buf);
        let buf2 = pool.take_zeroed(1024);
        assert_eq!(buf2.as_ptr(), ptr_before, "expected the exact same allocation back");
        assert_eq!(buf2.len(), 1024);
        assert!(buf2.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn take_zeroed_zeroes_stale_contents() {
        let mut pool = BufPool::new();
        let mut buf = pool.take_zeroed(8);
        buf.iter_mut().for_each(|x| *x = 7.0);
        pool.give(buf);
        let buf2 = pool.take_zeroed(8);
        assert!(buf2.iter().all(|&x| x == 0.0), "take_zeroed must not leak stale contents");
    }

    #[test]
    fn take_with_no_matching_size_allocates_fresh() {
        let mut pool = BufPool::new();
        pool.give(vec![1.0f32; 4]);
        let buf = pool.take_zeroed(16); // no 16-length buffer in the pool
        assert_eq!(buf.len(), 16);
        assert!(buf.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn f16_take_after_give_reuses_the_same_allocation() {
        let mut pool = BufPool::new();
        let buf = pool.take_f16_uninit(1024);
        let ptr_before = buf.as_ptr();
        pool.give_f16(buf);
        let buf2 = pool.take_f16_uninit(1024);
        assert_eq!(buf2.as_ptr(), ptr_before, "expected the exact same allocation back");
        assert_eq!(buf2.len(), 1024);
    }
}
