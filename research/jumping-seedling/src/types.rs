//! Core data types for Fydel Jumping Seedling.
//!
//! Everything compute-facing is built on [`AlignedVec`], a 64-byte aligned
//! heap buffer. AVX2 aligned loads (`_mm256_load_ps`) require 32-byte
//! alignment; cache lines are 64 bytes. Aligning to 64 satisfies both and
//! keeps tensor rows from straddling cache lines.

use std::alloc::{self, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// Alignment for all tensor-backing allocations (cache line on x86-64).
pub const ALIGN: usize = 64;

/// A heap buffer of `T` aligned to [`ALIGN`] bytes.
///
/// Bounded to `T: Copy` so we never have to run element destructors on drop —
/// the only `T` we use are `f32`, `u8`, `i8`. The buffer is always exactly
/// `len` elements (no separate capacity); these tensors are sized once at
/// allocation and not grown.
pub struct AlignedVec<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
}

impl<T: Copy> AlignedVec<T> {
    fn layout(len: usize) -> Layout {
        Layout::from_size_align(len * std::mem::size_of::<T>(), ALIGN)
            .expect("invalid aligned layout")
    }

    /// Allocate `len` elements without initializing them.
    ///
    /// # Safety
    /// Caller must write every element before reading. Prefer [`Self::zeroed`]
    /// or [`Self::from_slice`] unless the fill is guaranteed downstream.
    pub unsafe fn new_uninit(len: usize) -> Self {
        if len == 0 {
            return Self { ptr: NonNull::dangling(), len: 0 };
        }
        let layout = Self::layout(len);
        // SAFETY: layout has non-zero size (len > 0).
        let raw = unsafe { alloc::alloc(layout) } as *mut T;
        let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Self { ptr, len }
    }

    /// Allocate `len` elements, zero-filled.
    pub fn zeroed(len: usize) -> Self {
        if len == 0 {
            return Self { ptr: NonNull::dangling(), len: 0 };
        }
        let layout = Self::layout(len);
        // SAFETY: layout has non-zero size; zeroed bit-pattern is a valid T
        // for the Copy primitives we use (f32 0.0, integer 0).
        let raw = unsafe { alloc::alloc_zeroed(layout) } as *mut T;
        let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Self { ptr, len }
    }

    /// Allocate and copy from an existing slice.
    pub fn from_slice(src: &[T]) -> Self {
        // SAFETY: we fill all `len` elements immediately via copy below.
        let mut out = unsafe { Self::new_uninit(src.len()) };
        out.as_mut_slice().copy_from_slice(src);
        out
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: ptr is valid for `len` initialized elements (callers of
        // new_uninit are responsible for filling before any read reaches here).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: exclusive borrow, ptr valid for `len` elements.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY: ptr came from alloc with this exact layout; len != 0.
            unsafe { alloc::dealloc(self.ptr.as_ptr() as *mut u8, Self::layout(self.len)) }
        }
    }
}

impl<T: Copy> Deref for AlignedVec<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy> DerefMut for AlignedVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Copy> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

// SAFETY: AlignedVec owns its allocation uniquely; sending/sharing the raw
// pointer is sound for `T` that are themselves Send/Sync (our Copy primitives).
unsafe impl<T: Copy + Send> Send for AlignedVec<T> {}
unsafe impl<T: Copy + Sync> Sync for AlignedVec<T> {}

/// A dense 2D tensor of `f32`, row-major, cache-line aligned.
///
/// Shape is `[rows, cols]`. Row-major so a row is contiguous — the unit of
/// work for tiled kernels and the ReGLU sparsity skip (whole rows of W_down).
#[derive(Clone)]
pub struct Tensor {
    data: AlignedVec<f32>,
    shape: [usize; 2],
}

impl Tensor {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { data: AlignedVec::zeroed(rows * cols), shape: [rows, cols] }
    }

    pub fn from_vec(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "data length does not match shape");
        Self { data: AlignedVec::from_slice(&data), shape: [rows, cols] }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    #[inline]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        self.data.as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.data.as_mut_slice()
    }

    /// Borrow row `r` as a contiguous slice of length `cols`.
    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        let c = self.cols();
        &self.data.as_slice()[r * c..(r + 1) * c]
    }

    /// Mutably borrow row `r`.
    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        let c = self.cols();
        &mut self.data.as_mut_slice()[r * c..(r + 1) * c]
    }
}

/// A weight matrix held in compressed form: quantized per-block coefficients
/// `α` over the shared circular dictionary `G`.
///
/// This is the load-time-compressed representation that stays resident in L3.
/// The dictionary `G` is shared across all matrices and stored once externally
/// — it is *not* carried here. `bit_widths[g]` is the precision chosen for
/// group `g` by the loss-tolerance dial; `scales[g]` is its dequantization
/// scale; `packed` holds the quantized coefficient codes (little-endian i8/i16,
/// concatenated group by group).
pub struct CompressedWeight {
    /// Per-group quantization bit-width chosen at load time (8 or 16).
    pub bit_widths: AlignedVec<u8>,
    /// Per-group symmetric dequantization scale; `scales[g]` applies to group `g`.
    pub scales: AlignedVec<f32>,
    /// Quantized coefficient codes, concatenated per group.
    pub packed: AlignedVec<u8>,
    /// Logical shape of the reconstructed dense matrix `[rows, cols]`.
    pub shape: [usize; 2],
    /// Number of coefficients per quantization group.
    pub group_size: usize,
}

impl CompressedWeight {
    #[inline]
    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    #[inline]
    pub fn num_groups(&self) -> usize {
        self.bit_widths.len()
    }
}

/// INT8 momentum for the frequency-domain AdaFactor variant, with a per-group
/// dequantization scale. Stored at 1 byte/param instead of 4 (FP32) to keep
/// optimizer state cache-resident during the tiled update.
pub struct QuantizedMomentum {
    /// Quantized momentum codes, one i8 per parameter.
    pub data: AlignedVec<i8>,
    /// Per-group dequantization scale; `scales[g]` applies to group `g`.
    pub scales: AlignedVec<f32>,
    /// Parameters per group (must match the group used to derive `scales`).
    pub group_size: usize,
}

impl QuantizedMomentum {
    /// Allocate zeroed momentum for `n` parameters at the given group size.
    pub fn zeroed(n: usize, group_size: usize) -> Self {
        let num_groups = n.div_ceil(group_size);
        Self {
            data: AlignedVec::zeroed(n),
            scales: AlignedVec::zeroed(num_groups),
            group_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_alloc_is_aligned() {
        let v = AlignedVec::<f32>::zeroed(1000);
        assert_eq!(v.as_ptr() as usize % ALIGN, 0);
        assert_eq!(v.len(), 1000);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn empty_alloc_is_safe() {
        let v = AlignedVec::<f32>::zeroed(0);
        assert!(v.is_empty());
        assert_eq!(v.as_slice().len(), 0);
    }

    #[test]
    fn from_slice_roundtrips() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let v = AlignedVec::from_slice(&src);
        assert_eq!(v.as_slice(), &src);
        assert_eq!(v.as_ptr() as usize % ALIGN, 0);
    }

    #[test]
    fn tensor_row_access() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert_eq!(t.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(t.row(1), &[4.0, 5.0, 6.0]);
        assert_eq!(t.as_slice().as_ptr() as usize % ALIGN, 0);
    }

    #[test]
    fn tensor_clone_is_independent() {
        let mut a = Tensor::zeros(2, 2);
        let b = a.clone();
        a.as_mut_slice()[0] = 9.0;
        assert_eq!(b.as_slice()[0], 0.0);
    }

    #[test]
    fn quantized_momentum_group_count() {
        let m = QuantizedMomentum::zeroed(100, 64);
        assert_eq!(m.scales.len(), 2); // ceil(100/64)
        assert_eq!(m.data.len(), 100);
    }
}
