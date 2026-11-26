//! C interface of the faer sparse direct solvers for CalculiX (FEA Studio, stage C of the F17 plan).
//!
//! Thin glue, no solver of its own: the factorizations are faer's (supernodal LL^T, LDL^T, intranode
//! Bunch-Kaufman LBL^T for symmetric matrices, supernodal LU with partial pivoting for unsymmetric ones).
//! The caller (`faer.c` in CalculiX) computes the fill-reducing ordering of the symmetric case with the
//! nested dissection of SPOOLES and passes it as a permutation; the symbolic analysis is kept in the
//! handle, so a repeated factorization of a matrix with the same pattern (Newton iterations) costs only
//! the numeric factorization.
//!
//! No panic may cross the C boundary: every exported function catches it and returns FEA_FAER_PANIC.
//! No call exits the process on a failed allocation that faer can report: the status says so and the
//! caller falls back to SPOOLES.
//!
//! Licence of this file: MIT (faer, rayon and the other crates are MIT or dual licensed; the MIT terms
//! are used for every dual licensed crate, see licenses/ and THIRD_PARTY_LICENSES.md).

use std::ffi::{c_char, c_double, c_int, c_uint};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::{LdltError, LdltRegularization};
use faer::linalg::cholesky::llt::factor::{LltError, LltRegularization};
use faer::perm::PermRef;
use faer::sparse::linalg::cholesky::{
    self as chol, CholeskySymbolicParams, IntranodeLbltRef, LdltRef, LltRef, SymbolicCholesky,
    SymbolicCholeskyRaw, SymmetricOrdering,
};
use faer::sparse::linalg::lu::{self, LuRef, LuSymbolicParams, NumericLu, SymbolicLu};
use faer::sparse::linalg::{LuError, SupernodalThreshold};
use faer::sparse::{FaerError, SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par, Side};

type Ix = u32;

pub const FEA_FAER_OK: c_int = 0;
/// Cholesky: a non-positive pivot (the matrix is not positive definite)
pub const FEA_FAER_NOT_PD: c_int = 1;
/// LDL^T: zero pivot, LU: singular matrix
pub const FEA_FAER_ZERO_PIVOT: c_int = 2;
pub const FEA_FAER_NOMEM: c_int = 3;
/// the matrix does not fit the 32 bit indices of this interface
pub const FEA_FAER_TOO_BIG: c_int = 4;
pub const FEA_FAER_BAD_ARG: c_int = 5;
pub const FEA_FAER_PANIC: c_int = 6;
pub const FEA_FAER_OTHER: c_int = 7;

/// factorization method of a symmetric matrix
const M_LLT: c_int = 0;
const M_LDLT: c_int = 1;
const M_LBLT: c_int = 2;

static NTHREADS: AtomicUsize = AtomicUsize::new(1);
static SOLVE_THREADS: AtomicUsize = AtomicUsize::new(1);
/// 0 = faer chooses between the simplicial and the supernodal LU, 1 = always supernodal, 2 = always simplicial
static LU_MODE: AtomicUsize = AtomicUsize::new(0);

fn par_of(n: usize) -> Par {
    if n <= 1 {
        Par::Seq
    } else {
        Par::rayon(n)
    }
}

enum Sym {
    Chol(SymbolicCholesky<Ix>),
    Lu(SymbolicLu<Ix>),
}

enum Num {
    None,
    Llt(Vec<f64>),
    Ldlt(Vec<f64>),
    Lblt { l: Vec<f64>, sub: Vec<f64>, pf: Vec<Ix>, pi: Vec<Ix> },
    Lu(NumericLu<Ix, f64>),
}

pub struct Ctx {
    n: usize,
    col_ptr: Vec<Ix>,
    row_idx: Vec<Ix>,
    sym: Sym,
    num: Num,
    scratch: Option<MemBuffer>,
}

fn map_faer(e: &FaerError) -> c_int {
    match e {
        FaerError::OutOfMemory => FEA_FAER_NOMEM,
        FaerError::IndexOverflow => FEA_FAER_TOO_BIG,
        _ => FEA_FAER_OTHER,
    }
}

fn try_vec<T: Clone>(n: usize, v: T) -> Result<Vec<T>, c_int> {
    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(|_| FEA_FAER_NOMEM)?;
    out.resize(n, v);
    Ok(out)
}

fn copy_slice<T: Copy>(p: *const T, n: usize) -> Result<Vec<T>, c_int> {
    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(|_| FEA_FAER_NOMEM)?;
    // SAFETY: the caller guarantees n readable elements at p (documented in fea_faer.h)
    out.extend_from_slice(unsafe { std::slice::from_raw_parts(p, n) });
    Ok(out)
}

fn guard<F: FnOnce() -> c_int>(f: F) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => FEA_FAER_PANIC,
    }
}

/// Creates the global thread pool (once). nthreads <= 1: no pool, everything runs on the calling thread.
#[no_mangle]
pub extern "C" fn fea_faer_init(nthreads: c_int) -> c_int {
    guard(|| {
        let n = nthreads.max(1) as usize;
        NTHREADS.store(n, Ordering::Relaxed);
        if n > 1 {
            // a pool that exists already (faer, rayon) is fine: the number of splits below is what counts
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .stack_size(16 << 20)
                .thread_name(|i| format!("fea-faer-{i}"))
                .build_global();
        }
        FEA_FAER_OK
    })
}

/// Threads of the triangular solves (default 1: one right hand side is too little work for a team).
#[no_mangle]
pub extern "C" fn fea_faer_set_solve_threads(nthreads: c_int) {
    // no panic may cross the C boundary (nothing here can panic; the guard keeps the rule without exceptions)
    let _ = catch_unwind(|| SOLVE_THREADS.store(nthreads.max(1) as usize, Ordering::Relaxed));
}

/// Kind of the LU factorization of unsymmetric matrices: 0 = faer chooses (default), 1 = supernodal, 2 = simplicial (measurements).
#[no_mangle]
pub extern "C" fn fea_faer_set_lu_mode(mode: c_int) {
    let _ = catch_unwind(|| LU_MODE.store(mode.clamp(0, 2) as usize, Ordering::Relaxed));
}

/// Version line of the library, a static NUL terminated string.
#[no_mangle]
pub extern "C" fn fea_faer_version() -> *const c_char {
    // never NULL (the caller prints the string): a static fallback if the guard ever fires
    catch_unwind(|| c"faer 0.24.4 (MIT), glue fea_faer 0.1.0".as_ptr()).unwrap_or(c"faer".as_ptr())
}

/// Symbolic analysis. `col_ptr` (n + 1 entries) and `row_idx` (col_ptr[n] entries) are the column compressed
/// pattern: the lower triangle with the diagonal for `symmetric != 0`, the whole matrix otherwise (the rows of a column
/// need not be sorted, every entry once). `perm` = new to old ordering of the n unknowns (symmetric only; NULL = the
/// approximate minimum degree ordering of faer). Returns NULL and sets `*status` on failure.
///
/// # Safety
/// the pointers must be valid for the sizes above.
#[no_mangle]
pub unsafe extern "C" fn fea_faer_new(
    n: c_uint,
    col_ptr: *const u32,
    row_idx: *const u32,
    perm: *const u32,
    symmetric: c_int,
    status: *mut c_int,
) -> *mut Ctx {
    let mut st = FEA_FAER_OK;
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<Box<Ctx>, c_int> {
        let n = n as usize;
        if col_ptr.is_null() || row_idx.is_null() || n == 0 {
            return Err(FEA_FAER_BAD_ARG);
        }
        let cp = copy_slice(col_ptr, n + 1)?;
        let nnz = cp[n] as usize;
        let ri = copy_slice(row_idx, nnz)?;
        let a = SymbolicSparseColMatRef::<Ix>::new_unsorted_checked(n, n, &cp, None, &ri);
        let sym = if symmetric != 0 {
            let pv;
            let pinv;
            let ord = if perm.is_null() {
                SymmetricOrdering::Amd
            } else {
                pv = copy_slice(perm, n)?;
                let mut inv = try_vec(n, 0 as Ix)?;
                for (i, &o) in pv.iter().enumerate() {
                    if (o as usize) >= n {
                        return Err(FEA_FAER_BAD_ARG);
                    }
                    inv[o as usize] = i as Ix;
                }
                pinv = inv;
                SymmetricOrdering::Custom(PermRef::new_checked(&pv, &pinv, n))
            };
            let s = chol::factorize_symbolic_cholesky(
                a,
                Side::Lower,
                ord,
                CholeskySymbolicParams::default(),
            )
            .map_err(|e| map_faer(&e))?;
            Sym::Chol(s)
        } else {
            let mut params = LuSymbolicParams::default();
            match LU_MODE.load(Ordering::Relaxed) {
                1 => params.supernodal_flop_ratio_threshold = SupernodalThreshold::FORCE_SUPERNODAL,
                2 => params.supernodal_flop_ratio_threshold = SupernodalThreshold::FORCE_SIMPLICIAL,
                _ => {}
            }
            let s = lu::factorize_symbolic_lu(a, params).map_err(|e| map_faer(&e))?;
            Sym::Lu(s)
        };
        Ok(Box::new(Ctx { n, col_ptr: cp, row_idx: ri, sym, num: Num::None, scratch: None }))
    }));
    let out = match res {
        Ok(Ok(b)) => Box::into_raw(b),
        Ok(Err(e)) => {
            st = e;
            std::ptr::null_mut()
        }
        Err(_) => {
            st = FEA_FAER_PANIC;
            std::ptr::null_mut()
        }
    };
    if !status.is_null() {
        // SAFETY: status is a valid out pointer when not NULL
        unsafe { *status = st };
    }
    out
}

/// Numeric factorization with the values of the pattern given to fea_faer_new (same order). `method` for a symmetric
/// matrix: 0 = LL^T, 1 = LDL^T (no pivoting), 2 = LBL^T (Bunch-Kaufman inside the supernodes); ignored for LU.
///
/// # Safety
/// `f` from fea_faer_new, `values` valid for col_ptr[n] doubles.
#[no_mangle]
pub unsafe extern "C" fn fea_faer_factor(f: *mut Ctx, values: *const c_double, method: c_int) -> c_int {
    if f.is_null() || values.is_null() {
        return FEA_FAER_BAD_ARG;
    }
    // SAFETY: f comes from fea_faer_new and is not used concurrently (single threaded caller)
    let ctx = unsafe { &mut *f };
    guard(AssertUnwindSafe(|| factor_impl(ctx, values, method)))
}

fn factor_impl(ctx: &mut Ctx, values: *const f64, method: c_int) -> c_int {
    ctx.num = Num::None; // the old factors go first (memory)
    ctx.scratch = None;
    let n = ctx.n;
    let nnz = ctx.col_ptr[n] as usize;
    // SAFETY: values has nnz entries (contract above)
    let vals: &[f64] = unsafe { std::slice::from_raw_parts(values, nnz) };
    let sp = SymbolicSparseColMatRef::<Ix>::new_unsorted_checked(n, n, &ctx.col_ptr, None, &ctx.row_idx);
    let a = SparseColMatRef::<Ix, f64>::new(sp, vals);
    let par = par_of(NTHREADS.load(Ordering::Relaxed));
    match &ctx.sym {
        Sym::Chol(s) => {
            let len = s.len_val();
            match method {
                M_LLT | M_LDLT => {
                    let mut l = match try_vec(len, 0.0f64) {
                        Ok(v) => v,
                        Err(e) => return e,
                    };
                    let scratch = if method == M_LLT {
                        s.factorize_numeric_llt_scratch::<f64>(par, Default::default())
                    } else {
                        s.factorize_numeric_ldlt_scratch::<f64>(par, Default::default())
                    };
                    let mut mem = match MemBuffer::try_new(scratch) {
                        Ok(m) => m,
                        Err(_) => return FEA_FAER_NOMEM,
                    };
                    if method == M_LLT {
                        match s.factorize_numeric_llt::<f64>(
                            &mut l,
                            a,
                            Side::Lower,
                            LltRegularization::default(),
                            par,
                            MemStack::new(&mut mem),
                            Default::default(),
                        ) {
                            Ok(_) => {}
                            Err(LltError::NonPositivePivot { .. }) => return FEA_FAER_NOT_PD,
                        }
                        ctx.num = Num::Llt(l);
                    } else {
                        match s.factorize_numeric_ldlt::<f64>(
                            &mut l,
                            a,
                            Side::Lower,
                            LdltRegularization::default(),
                            par,
                            MemStack::new(&mut mem),
                            Default::default(),
                        ) {
                            Ok(_) => {}
                            Err(LdltError::ZeroPivot { .. }) => return FEA_FAER_ZERO_PIVOT,
                        }
                        ctx.num = Num::Ldlt(l);
                    }
                    FEA_FAER_OK
                }
                M_LBLT => {
                    let mut l = match try_vec(len, 0.0f64) {
                        Ok(v) => v,
                        Err(e) => return e,
                    };
                    let mut sub = match try_vec(n, 0.0f64) {
                        Ok(v) => v,
                        Err(e) => return e,
                    };
                    let mut pf = match try_vec(n, 0 as Ix) {
                        Ok(v) => v,
                        Err(e) => return e,
                    };
                    let mut pi = match try_vec(n, 0 as Ix) {
                        Ok(v) => v,
                        Err(e) => return e,
                    };
                    let scratch = s.factorize_numeric_intranode_lblt_scratch::<f64>(par, Default::default());
                    let mut mem = match MemBuffer::try_new(scratch) {
                        Ok(m) => m,
                        Err(_) => return FEA_FAER_NOMEM,
                    };
                    let _ = s.factorize_numeric_intranode_lblt::<f64>(
                        &mut l,
                        &mut sub,
                        &mut pf,
                        &mut pi,
                        a,
                        Side::Lower,
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    );
                    ctx.num = Num::Lblt { l, sub, pf, pi };
                    FEA_FAER_OK
                }
                _ => FEA_FAER_BAD_ARG,
            }
        }
        Sym::Lu(s) => {
            let mut num = NumericLu::<Ix, f64>::new();
            let scratch = s.factorize_numeric_lu_scratch::<f64>(par, Default::default());
            let mut mem = match MemBuffer::try_new(scratch) {
                Ok(m) => m,
                Err(_) => return FEA_FAER_NOMEM,
            };
            match s.factorize_numeric_lu::<f64>(&mut num, a, par, MemStack::new(&mut mem), Default::default()) {
                Ok(_) => {}
                Err(LuError::SymbolicSingular { .. }) => return FEA_FAER_ZERO_PIVOT,
                Err(LuError::Generic(e)) => return map_faer(&e),
            }
            ctx.num = Num::Lu(num);
            FEA_FAER_OK
        }
    }
}

/// Solves in place for `nrhs` right hand sides stored one after the other (column major, n rows).
///
/// # Safety
/// `f` from fea_faer_new with a successful fea_faer_factor, `b` valid for n * nrhs doubles.
#[no_mangle]
pub unsafe extern "C" fn fea_faer_solve(f: *mut Ctx, b: *mut c_double, nrhs: c_uint) -> c_int {
    if f.is_null() || b.is_null() || nrhs == 0 {
        return FEA_FAER_BAD_ARG;
    }
    // SAFETY: see above
    let ctx = unsafe { &mut *f };
    guard(AssertUnwindSafe(|| solve_impl(ctx, b, nrhs as usize)))
}

fn solve_impl(ctx: &mut Ctx, b: *mut f64, nrhs: usize) -> c_int {
    let n = ctx.n;
    // SAFETY: b has n * nrhs entries (contract above)
    let rhs: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(b, n * nrhs) };
    let par = par_of(SOLVE_THREADS.load(Ordering::Relaxed));
    let need = match (&ctx.sym, &ctx.num) {
        (Sym::Chol(s), Num::Llt(_) | Num::Ldlt(_) | Num::Lblt { .. }) => s.solve_in_place_scratch::<f64>(nrhs, par),
        (Sym::Lu(s), Num::Lu(_)) => s.solve_in_place_scratch::<f64>(nrhs, par),
        _ => return FEA_FAER_BAD_ARG,
    };
    // the buffer is kept between the calls (an eigenvalue analysis solves thousands of times)
    let fits = ctx.scratch.as_ref().map(|m| m.len() >= need.size_bytes() + 4096).unwrap_or(false);
    if !fits {
        ctx.scratch = None;
        match MemBuffer::try_new(need.and(faer::dyn_stack::StackReq::new::<u8>(4096))) {
            Ok(m) => ctx.scratch = Some(m),
            Err(_) => return FEA_FAER_NOMEM,
        }
    }
    let mem = ctx.scratch.as_mut().unwrap();
    let x = MatMut::from_column_major_slice_mut(rhs, n, nrhs);
    match (&ctx.sym, &ctx.num) {
        (Sym::Chol(s), Num::Llt(l)) => LltRef::new(s, l).solve_in_place_with_conj(Conj::No, x, par, MemStack::new(mem)),
        (Sym::Chol(s), Num::Ldlt(l)) => LdltRef::new(s, l).solve_in_place_with_conj(Conj::No, x, par, MemStack::new(mem)),
        (Sym::Chol(s), Num::Lblt { l, sub, pf, pi }) => {
            let p = PermRef::new_checked(pf, pi, n);
            IntranodeLbltRef::new(s, l, sub, p).solve_in_place_with_conj(Conj::No, x, par, MemStack::new(mem))
        }
        (Sym::Lu(s), Num::Lu(num)) => {
            LuRef::new_unchecked(s, num).solve_in_place_with_conj(Conj::No, x, par, MemStack::new(mem))
        }
        _ => return FEA_FAER_BAD_ARG,
    }
    FEA_FAER_OK
}

/// Frees the numeric factors (the symbolic analysis stays for the next factorization of the same pattern).
///
/// # Safety
/// `f` from fea_faer_new or NULL.
#[no_mangle]
pub unsafe extern "C" fn fea_faer_drop_numeric(f: *mut Ctx) {
    if f.is_null() {
        return;
    }
    // SAFETY: see above
    let ctx = unsafe { &mut *f };
    let _ = catch_unwind(AssertUnwindSafe(|| {
        ctx.num = Num::None;
        ctx.scratch = None;
    }));
}

/// Frees everything.
///
/// # Safety
/// `f` from fea_faer_new or NULL; not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn fea_faer_free(f: *mut Ctx) {
    if f.is_null() {
        return;
    }
    // SAFETY: allocated by Box::into_raw in fea_faer_new
    let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(f) })));
}

/// Statistics of the analysis: out[0] = number of stored values of the factor, out[1] = 1 for a supernodal factor
/// (0 simplicial), out[2] = number of supernodes (0 for a simplicial factor or LU), out[3] = 1 for LU.
///
/// # Safety
/// `f` from fea_faer_new, `out` valid for 4 doubles.
#[no_mangle]
pub unsafe extern "C" fn fea_faer_info(f: *const Ctx, out: *mut c_double) {
    if f.is_null() || out.is_null() {
        return;
    }
    // SAFETY: see above
    let ctx = unsafe { &*f };
    let o = unsafe { std::slice::from_raw_parts_mut(out, 4) };
    o.fill(0.0);
    let _ = catch_unwind(AssertUnwindSafe(|| match &ctx.sym {
        Sym::Chol(s) => {
            o[0] = s.len_val() as f64;
            if let SymbolicCholeskyRaw::Supernodal(sn) = s.raw() {
                o[1] = 1.0;
                o[2] = sn.n_supernodes() as f64;
            }
        }
        Sym::Lu(_) => o[3] = 1.0,
    }));
}
