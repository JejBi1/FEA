/* C interface of libfea_faer (installer/faer_ccx, Rust crate "fea_faer"): faer sparse direct solvers for CalculiX.
 * Licence of this header and of the crate: MIT (faer and the other crates: MIT or dual licensed, the MIT terms are used). */
#ifndef FEA_FAER_H
#define FEA_FAER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FEA_FAER_OK          0
#define FEA_FAER_NOT_PD      1   /* LL^T: a non-positive pivot (the matrix is not positive definite) */
#define FEA_FAER_ZERO_PIVOT  2   /* LDL^T: zero pivot; LU: singular matrix */
#define FEA_FAER_NOMEM       3
#define FEA_FAER_TOO_BIG     4   /* does not fit the 32 bit indices of this interface */
#define FEA_FAER_BAD_ARG     5
#define FEA_FAER_PANIC       6   /* a panic of the Rust code was caught */
#define FEA_FAER_OTHER       7

#define FEA_FAER_LLT         0
#define FEA_FAER_LDLT        1   /* no pivoting */
#define FEA_FAER_LBLT        2   /* Bunch-Kaufman pivoting inside the supernodes */

typedef struct fea_faer_ctx fea_faer_ctx;

/* creates the thread pool (once); nthreads <= 1: everything on the calling thread */
int fea_faer_init(int nthreads);
/* threads of the triangular solves (default 1) */
void fea_faer_set_solve_threads(int nthreads);
/* kind of the LU factorization of unsymmetric matrices: 0 = faer chooses (default), 1 = supernodal, 2 = simplicial (measurements) */
void fea_faer_set_lu_mode(int mode);
const char *fea_faer_version(void);

/* symbolic analysis. col_ptr: n + 1 entries, row_idx: col_ptr[n] entries (column compressed pattern: the lower triangle with
   the diagonal when symmetric != 0, the whole matrix otherwise; every entry once, rows of a column in any order).
   perm: new to old ordering (symmetric only; NULL = approximate minimum degree of faer). NULL + *status on failure. */
fea_faer_ctx *fea_faer_new(unsigned n, const uint32_t *col_ptr, const uint32_t *row_idx, const uint32_t *perm,
                           int symmetric, int *status);
/* numeric factorization; values in the order of the pattern; method: FEA_FAER_LLT / LDLT / LBLT (symmetric), ignored for LU */
int fea_faer_factor(fea_faer_ctx *f, const double *values, int method);
/* solves in place, nrhs right hand sides one after the other (n entries each) */
int fea_faer_solve(fea_faer_ctx *f, double *b, unsigned nrhs);
void fea_faer_drop_numeric(fea_faer_ctx *f);
void fea_faer_free(fea_faer_ctx *f);
/* out[0] values of the factor, out[1] 1 = supernodal, out[2] supernodes, out[3] 1 = LU */
void fea_faer_info(const fea_faer_ctx *f, double *out);

#ifdef __cplusplus
}
#endif
#endif
