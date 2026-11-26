/*     CalculiX - A 3-dimensional finite element program                   */
/*              Copyright (C) 1998-2025 Guido Dhondt                          */

/*     This program is free software; you can redistribute it and/or     */
/*     modify it under the terms of the GNU General Public License as    */
/*     published by the Free Software Foundation(version 2);    */
/*                    */

/*     This program is distributed in the hope that it will be useful,   */
/*     but WITHOUT ANY WARRANTY; without even the implied warranty of    */
/*     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the      */
/*     GNU General Public License for more details.                      */

/*     You should have received a copy of the GNU General Public License */
/*     along with this program; if not, write to the Free Software       */
/*     Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.         */

/*
 * FEA Studio change (new file, 23. 11. 2026, revised 24. 11. 2026, GPL-2.0 like the rest of CalculiX 2.23): the faer sparse direct solver
 * (Rust library, MIT, linked statically through the C interface of installer/faer_ccx) behind the SPOOLES interface of
 * spooles.c. spooles_factor / spooles_solve / spooles_cleanup call the functions of this file first; when they return 0
 * the system of equations is factorized by SPOOLES exactly as before.
 *
 *   symmetric matrices (symmetryflag 0, inputformat 0): the fill reducing ordering is the nested dissection of SPOOLES
 *       (orderViaND with the parameters of spooles.c on the graph of the matrix: no ordering of our own is written, and the ordering that
 *       made SPOOLES fit 2 - 3 GB where AMD needs more is kept), the factorization is faer's supernodal Cholesky LL^T; a matrix that is not positive definite (a shifted
 *       eigenvalue problem, buckling) goes down a ladder: faer's LDL^T without pivoting, then its Bunch-Kaufman LBL^T inside
 *       the supernodes (3 times slower), then SPOOLES; the method that worked is remembered for the next matrix of the same pattern;
 *   unsymmetric matrices (symmetryflag 2, inputformat 1, 3, 4): faer's LU with partial pivoting (the column ordering of faer, COLAMD) - only when
 *       FEA_FAER_UNSYM=1, otherwise SPOOLES (the LU of faer is slower, see below);
 *   everything else (dense matrices of the radiation, small systems): SPOOLES.
 *
 * Every factorization is CHECKED like the multi-threaded SPOOLES factorization of spooles.c: one test solve with a pseudo
 * random right hand side and the row-wise backward error max_i |b_i - (A x)_i| / (sum_j |a_ij x_j| + |b_i|) of the matrix
 * that CalculiX handed over (not of the copy that was given to faer); above FEA_FAER_CHECK_TOL (1e-12) the factors are
 * dropped and SPOOLES factorizes the system, with a line in the log. A failed factorization (not positive definite
 * and the LBL^T fails too, zero pivot, no memory, a panic of the library) is the same: SPOOLES, with a line in the log.
 * A wrong answer without a message is not possible this way.
 *
 * A numerically singular system (a body that is held only by contacts that are still open, a free-free body with a tiny shift: the first iterations of
 * a Hertz contact, the verification case K2) has no solution that means anything; CalculiX goes on with whatever the factorization returns and finds its way
 * (contact elements are stabilized, rigid-body eigenvalues are noise) - a different vector in the null space of the matrix can make it diverge or change the
 * number of modes. So the same test solve gives a lower bound of the condition number, kappa >= |A|_inf |x|_inf / |b|_inf, and above FEA_FAER_COND_MAX
 * (default 1e12: fewer than 4 of the 16 digits are left) the system goes to SPOOLES as before ("faer: the system of equations is numerically singular"):
 * the behaviour of CalculiX for such matrices stays what it was. Measured on the 266 decks of the verification suite (23. 11. 2026): statics, shells, heat,
 * contact, plasticity, buckling 1e6 ... 2e11; above the limit only the first system of K2 (1e17), the orthotropic ring V16 (2e13), the free-free beam M2
 * (1.5e13) and the cyclic-symmetry modal problem CY2 (1e13) - a false alarm costs the speed of faer only, never a result.
 *
 * Review fixes of 24. 11. 2026: (C-3) a system declined by the condition guard is remembered per pattern (and per "shifted or not"): the following factorizations of the SAME
 * pattern go to SPOOLES at once, without the numeric factorization and the test solve of faer that would only be thrown away again (a free-free eigenvalue problem, a
 * contact analysis whose first iterations are singular); faer tries once more after FAER_RETRY_AFTER (20) such factorizations, and at once when the pattern changes;
 * (C-7) the column counts are checked against nzs BEFORE icol / irow are read with it (the digest of the pattern), and nzs is stored and compared when the analysis is
 * reused; (C-8) the number of stored entries must stay below 0x7fffff00 (the indices of CalculiX are signed 32 bit integers).
 *
 * The symbolic analysis (ordering, supernodes) is kept between the factorizations of a matrix with the same pattern
 * (Newton iterations, the shifted solves of one eigenvalue analysis): the pattern is recognised by a 64 bit digest of
 * icol / irow (contact changes the pattern: the analysis is made again).
 *
 * Environment (the program removes the variables marked * from the environment of ccx unless FEA_DEV=1):
 *   FEA_FAER=0           faer is not used at all (SPOOLES only)
 *   FEA_FAER_MIN_EQ=N    systems with fewer equations go to SPOOLES (default 2000)
 *   FEA_FAER_UNSYM=1     unsymmetric matrices go to faer's LU (default 0: SPOOLES; measured 23. 11. 2026: the LU of faer is single threaded and 2 - 3 times slower than the
 *                        SPOOLES factorization on the friction case K6, 35 k equations - it is correct (residuals 1e-14) but not faster, so it is off by default)
 * * FEA_FAER_CHECK_TOL   limit of the check (default 1e-12; 0 switches the check off: tests only)
 * * FEA_FAER_METHOD      llt | ldlt | lblt: force the method of the symmetric factorization (tests)
 * * FEA_FAER_VERBOSE=1   one line with the times and the condition number bound after every factorization
 * * FEA_FAER_COND_MAX   the condition number bound above which the system goes to SPOOLES (default 1e12; 0 switches the test off: tests only)
 * * FEA_FAER_ORDER       nd (default: the nested dissection of SPOOLES) | bestof (the better of the nested dissection and the multisection of SPOOLES, as spooles.c: 25 % more
 *                      time for the ordering and the same fill - measured 23. 11. 2026: 117 k equations 4.1 s against 2.8 s, 193 k equations 6.6 s against 4.8 s, factor 54 / 55 and 112 / 110
 *                      million entries) | amd (faer's own: 20 times faster to compute, but 1.3 times the fill at 66 k equations and 1.45 times at 193 k, the numeric factorization 2.5 times slower)
 * * FEA_FAER_MIN_EQ, FEA_FAER_UNSYM, FEA_FAER_LU (auto | supernodal | simplicial): see above (tests and measurements)
 * * FEA_FAER_INJECT=1    TEST ONLY: the factors are those of a matrix whose one entry is changed by 1e-4 - the check must catch it (then SPOOLES solves)
 *   FEA_FAER_SOLVE_THREADS=N  threads of a triangular solve (default 1)
 */

#ifdef FEA_FAER

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>
#include "CalculiX.h"
#include "spooles.h"
#include "fea_faer.h"
#include "faer.h"

#define DIAGBIT  0x80000000u        /* src entry of a diagonal value: DIAGBIT | index into ad */

/* the ordering of spooles.c (ssolve_creategraph) */
#define TUNE_MAXZEROS		1000
#define TUNE_MAXDOMAINSIZE	800
#define TUNE_MAXSIZE		64
#define RNDSEED		7892713

static struct {
    fea_faer_ctx *ctx;       /* symbolic analysis (and numeric factors when have_num) */
    int valid;               /* ctx belongs to the pattern with the digest key */
    uint64_t key;
    ITG n;
    ITG nzs;                 /* nzs of the pattern (compared again when the analysis is reused) */
    int sym;                 /* 1 symmetric (lower triangle), 0 unsymmetric */
    int cond_declined[2];    /* the condition guard declined a system of this pattern ([0] without a shift, [1] with one): the next ones go to SPOOLES at once */
    int cond_skips[2];       /* factorizations that were skipped since */
    uint32_t nnz;
    uint32_t *colptr;        /* n + 1 */
    uint32_t *src;           /* nnz: where the value of every stored entry comes from (ad / au) */
    int have_num;            /* numeric factors of the matrix of the last factorization */
    int method;
    int prefer[2];           /* the method that worked for the last matrix of this pattern: [0] without a shift, [1] with a shift (sigma != 0); positive definite until shown otherwise */
    int off;                 /* a hard failure of the library: not used again in this process */
    int inited;
    double flen;             /* entries of the factor */
    int nsolve;              /* solves with the current factors and their time (reported at the cleanup) */
    double tsolve;
} G;

static int env_int(const char *name, int dflt)
{
    char *e=getenv(name),*end;
    long v;
    if((e==NULL)||(*e=='\0')) return dflt;
    v=strtol(e,&end,10);
    if(*end!='\0') return dflt;
    return (int)v;
}

static double env_tol(void)
{
    char *e=getenv("FEA_FAER_CHECK_TOL"),*end;
    double t;
    if((e!=NULL)&&(*e!='\0')){
        t=strtod(e,&end);
        if((end!=e)&&(*end=='\0')&&(t>=0.)) return t;     /* a typo keeps the default */
    }
    return 1.e-12;
}

static double env_cond(void)
{
    char *e=getenv("FEA_FAER_COND_MAX"),*end;
    double t;
    if((e!=NULL)&&(*e!='\0')){
        t=strtod(e,&end);
        if((end!=e)&&(*end=='\0')&&(t>=0.)) return t;
    }
    return 1.e12;
}

static int faer_threads(void)
{
    /* the rules of spooles.c: CCX_NPROC_EQUATION_SOLVER (at most the processors), else OMP_NUM_THREADS, else all */
    int sys_cpus=0,n=0;
    char *e;
    e=getenv("NUMBER_OF_CPUS");
    if(e){sys_cpus=atoi(e);if(sys_cpus<0) sys_cpus=0;}
    if(sys_cpus==0){sys_cpus=(int)getSystemCPUs();if(sys_cpus<1) sys_cpus=1;}
    e=getenv("CCX_NPROC_EQUATION_SOLVER");
    if(e){n=atoi(e);if(n<0) n=0;else if(n>sys_cpus) n=sys_cpus;}
    if(n==0){
        e=getenv("OMP_NUM_THREADS");
        if(e) n=atoi(e);
        if(n<1) n=1;else if(n>sys_cpus) n=sys_cpus;
    }
    return n;
}

static uint64_t mix(uint64_t h,uint64_t w)
{
    h^=w;h*=0x9E3779B97F4A7C15ULL;h^=h>>29;
    return h;
}

static uint64_t hash_ints(uint64_t h,const ITG *a,size_t m)
{
    size_t i;
    for(i=0;i<m;i++) h=mix(h,(uint64_t)(uint32_t)a[i]);
    return h;
}

static int cmp_u64(const void *a,const void *b)
{
    uint64_t x=*(const uint64_t *)a,y=*(const uint64_t *)b;
    return (x>y)-(x<y);
}

/* HARNESS-BEGIN: pure helpers, compiled alone by tests/faer_guard_harness.c (no CalculiX, no SPOOLES) */

#define FAER_RETRY_AFTER 20          /* factorizations of a declined pattern that go to SPOOLES before faer tries it once more */

/* the sizes of the pattern must be consistent BEFORE anything reads icol / irow with them (the digest, build_pattern): 0 = ok and, if ptot is not NULL, *ptot = the
   number of off-diagonal entries counted; otherwise *why says what is wrong (the system goes to SPOOLES) */
static int check_counts(int fmt,ITG n,ITG nzs,const ITG *icol,const ITG *irow,int64_t *ptot,const char **why)
{
    int64_t tot=0;
    ITG j;
    *why=NULL;
    if(n<1){*why="no equations";return -1;}
    if(nzs<0){*why="negative nzs";return -1;}
    if(fmt==4){
        if((irow[0]!=0)||(irow[n]!=nzs)){*why="unexpected row pointers (input format 4)";return -1;}
        tot=nzs;
    }else{
        for(j=0;j<n;j++){if(icol[j]<0){*why="negative column count";return -1;}tot+=icol[j];}
        if(tot!=nzs){*why="the column counts do not add up to nzs";return -1;}
    }
    if(ptot!=NULL) *ptot=tot;
    return 0;
}

/* a pattern whose system the condition guard declined: 1 = skip faer for this factorization (SPOOLES), 0 = try it (not declined, or the retry after FAER_RETRY_AFTER skips
   - the flag is cleared and set again if the guard declines once more) */
static int declined_skip(int *declined,int *skips)
{
    if(!*declined) return 0;
    if(*skips<FAER_RETRY_AFTER){(*skips)++;return 1;}
    *declined=0;*skips=0;
    return 0;
}

/* HARNESS-END */

/* sorts the entries of one column by row (src follows); returns 1 if a row occurs twice */
static int sort_col(uint32_t *r,uint32_t *s,size_t len)
{
    size_t i,j;
    int dup=0,unsorted=0;
    if(len<2) return 0;
    for(i=1;i<len;i++){
        if(r[i]<r[i-1]) unsorted=1;
        else if(r[i]==r[i-1]) dup=1;
    }
    if(!unsorted) return dup;                        /* the usual case: CalculiX writes the rows of a column in ascending order */
    dup=0;
    if(len<=48){
        for(i=1;i<len;i++){
            uint32_t rr=r[i],ss=s[i];
            for(j=i;j>0&&r[j-1]>rr;j--){r[j]=r[j-1];s[j]=s[j-1];}
            r[j]=rr;s[j]=ss;
        }
    }else{
        uint64_t *k=(uint64_t *)malloc(len*sizeof(uint64_t));
        if(k==NULL) return 1;                        /* no memory: treated as a failure by the caller (dup) */
        for(i=0;i<len;i++) k[i]=((uint64_t)r[i]<<32)|s[i];
        qsort(k,len,sizeof(uint64_t),cmp_u64);
        for(i=0;i<len;i++){r[i]=(uint32_t)(k[i]>>32);s[i]=(uint32_t)(k[i]&0xffffffffu);}
        free(k);
    }
    for(i=1;i<len;i++) if(r[i]==r[i-1]) dup=1;
    return dup;
}

static void free_pattern(void)
{
    if(G.ctx){fea_faer_free(G.ctx);G.ctx=NULL;}
    free(G.colptr);G.colptr=NULL;
    free(G.src);G.src=NULL;
    G.valid=0;G.have_num=0;G.nnz=0;G.nzs=0;
    G.cond_declined[0]=G.cond_declined[1]=0;G.cond_skips[0]=G.cond_skips[1]=0;     /* a decline belongs to the pattern that is gone */
}

/* the column compressed pattern of the matrix and, for every stored entry, where its value is. 0 = ok, otherwise *why says
   what is not supported (the system goes to SPOOLES). The diagonal is part of the pattern (symmetric: the lower triangle). */
static int build_pattern(int fmt,ITG n,ITG nzs,ITG nzs3,ITG *icol,ITG *irow,uint32_t **pcolptr,uint32_t **prow,
                         uint32_t **psrc,uint32_t *pnnz,const char **why)
{
    uint32_t *colptr=NULL,*row=NULL,*src=NULL,*cnt=NULL;
    int64_t tot=0,nnz;
    ITG j,k,ipoint;
    size_t p;
    int unsorted=0;

    if(check_counts(fmt,n,nzs,icol,irow,&tot,why)) return -1;
    if(fmt==0) nnz=(int64_t)n+tot;
    else if(fmt==1) nnz=(int64_t)n+2*tot;
    else if(fmt==3) nnz=(int64_t)n+tot;
    else nnz=tot;
    if((nnz>=0x7fffff00LL)||(n>=0x7fffff00)){*why="the matrix does not fit 32 bit indices";return -1;}

    colptr=(uint32_t *)malloc(((size_t)n+1)*sizeof(uint32_t));
    row=(uint32_t *)malloc((size_t)nnz*sizeof(uint32_t));
    src=(uint32_t *)malloc((size_t)nnz*sizeof(uint32_t));
    if((colptr==NULL)||(row==NULL)||(src==NULL)){*why="no memory for the pattern";goto fail;}

    if(fmt==0){
        /* lower triangle by columns: the diagonal, then the rows irow-1 > column */
        p=0;ipoint=0;
        for(j=0;j<n;j++){
            colptr[j]=(uint32_t)p;
            row[p]=(uint32_t)j;src[p]=DIAGBIT|(uint32_t)j;p++;
            for(k=0;k<icol[j];k++){
                ITG r=irow[ipoint+k]-1;
                if((r<=j)||(r>=n)){*why="an entry on or above the diagonal in the lower triangle";goto fail;}
                if((k>0)&&((uint32_t)r<=row[p-1])) unsorted=1;
                row[p]=(uint32_t)r;src[p]=(uint32_t)(ipoint+k);p++;
            }
            if(unsorted){
                if(sort_col(row+colptr[j],src+colptr[j],p-colptr[j])){*why="a row occurs twice in a column";goto fail;}
                unsorted=0;
            }
            ipoint+=icol[j];
        }
        colptr[n]=(uint32_t)p;
    }else if(fmt==1){
        /* both triangles with the same structure: column c = the rows r < c whose list contains c (upper entries,
           au[ipo + nzs3]), the diagonal, the rows irow-1 > c (lower entries, au[ipo]) */
        cnt=(uint32_t *)calloc((size_t)n,sizeof(uint32_t));
        if(cnt==NULL){*why="no memory for the pattern";goto fail;}
        for(p=0;p<(size_t)nzs;p++){
            ITG r=irow[p]-1;
            if((r<0)||(r>=n)){*why="row index out of range";goto fail;}
            cnt[r]++;
        }
        {   /* cnt[c] = the number of upper entries of column c; colptr from it */
            size_t q=0;
            for(j=0;j<n;j++){
                colptr[j]=(uint32_t)q;
                q+=(size_t)cnt[j]+1+(size_t)icol[j];
            }
            colptr[n]=(uint32_t)q;
        }
        {   /* the upper entries: loop over the columns r ascending, so the rows of column c come out ascending */
            uint32_t *fill=(uint32_t *)calloc((size_t)n,sizeof(uint32_t));
            if(fill==NULL){*why="no memory for the pattern";goto fail;}
            ipoint=0;
            for(j=0;j<n;j++){
                for(k=0;k<icol[j];k++){
                    ITG c=irow[ipoint+k]-1;
                    if(c<=j){free(fill);*why="an entry on or above the diagonal in the lower triangle";goto fail;}
                    p=colptr[c]+fill[c]++;
                    row[p]=(uint32_t)j;src[p]=(uint32_t)(ipoint+k+nzs3);
                }
                ipoint+=icol[j];
            }
            ipoint=0;
            for(j=0;j<n;j++){
                p=colptr[j]+cnt[j];
                row[p]=(uint32_t)j;src[p]=DIAGBIT|(uint32_t)j;p++;
                for(k=0;k<icol[j];k++){
                    row[p]=(uint32_t)(irow[ipoint+k]-1);src[p]=(uint32_t)(ipoint+k);p++;
                }
                if(sort_col(row+colptr[j],src+colptr[j],colptr[j+1]-colptr[j])){free(fill);*why="a row occurs twice in a column";goto fail;}
                ipoint+=icol[j];
            }
            free(fill);
        }
    }else if(fmt==3){
        /* the diagonal in ad, the other entries of column j in au / irow (any rows) */
        p=0;ipoint=0;
        for(j=0;j<n;j++){
            colptr[j]=(uint32_t)p;
            for(k=0;k<icol[j];k++){
                ITG r=irow[ipoint+k]-1;
                if((r<0)||(r>=n)||(r==j)){*why="row index out of range or on the diagonal";goto fail;}
                row[p]=(uint32_t)r;src[p]=(uint32_t)(ipoint+k);p++;
            }
            row[p]=(uint32_t)j;src[p]=DIAGBIT|(uint32_t)j;p++;
            if(sort_col(row+colptr[j],src+colptr[j],p-colptr[j])){*why="a row occurs twice in a column";goto fail;}
            ipoint+=icol[j];
        }
        colptr[n]=(uint32_t)p;
    }else{
        /* format 4: the matrix is the transpose of the rows given: column `row` holds the entries col = icol-1 */
        for(j=0;j<n;j++){
            colptr[j]=(uint32_t)irow[j];
            if((irow[j+1]<irow[j])||(irow[j+1]>nzs)){*why="decreasing row pointers";goto fail;}
            for(k=irow[j];k<irow[j+1];k++){
                ITG c=icol[k]-1;
                if((c<0)||(c>=n)){*why="column index out of range";goto fail;}
                row[k]=(uint32_t)c;src[k]=(uint32_t)k;
            }
            if(sort_col(row+irow[j],src+irow[j],(size_t)(irow[j+1]-irow[j]))){*why="a column occurs twice in a row";goto fail;}
        }
        colptr[n]=(uint32_t)nzs;
    }
    free(cnt);
    *pcolptr=colptr;*prow=row;*psrc=src;*pnnz=(uint32_t)nnz;
    return 0;
fail:
    free(colptr);free(row);free(src);free(cnt);
    if(*why==NULL) *why="no memory for the pattern";
    return -1;
}

/* nested dissection of SPOOLES for the pattern of a symmetric matrix (lower triangle by columns, diagonal first):
   perm = new to old. 0 = ok */
static int order_symmetric(ITG n,const uint32_t *colptr,const uint32_t *row,uint32_t *perm,int mode)
{
    IVL *ivl;
    Graph *graph;
    ETree *etree;
    IV *n2o;
    int *sizes,*cur,**lists,sz,i,*ent;
    ITG j;
    size_t p;
    int nedges;

    sizes=(int *)calloc((size_t)n,sizeof(int));
    cur=(int *)calloc((size_t)n,sizeof(int));
    lists=(int **)malloc((size_t)n*sizeof(int *));
    if((sizes==NULL)||(cur==NULL)||(lists==NULL)){free(sizes);free(cur);free(lists);return -1;}
    for(j=0;j<n;j++){
        for(p=colptr[j]+1;p<colptr[j+1];p++){sizes[j]++;sizes[row[p]]++;}
    }
    ivl=IVL_new();
    IVL_init3(ivl,IVL_CHUNKED,n,sizes);
    for(j=0;j<n;j++) IVL_listAndSize(ivl,j,&sz,&lists[j]);
    for(j=0;j<n;j++){
        for(p=colptr[j]+1;p<colptr[j+1];p++){
            uint32_t r=row[p];
            lists[j][cur[j]++]=(int)r;
            lists[r][cur[r]++]=(int)j;
        }
    }
    free(sizes);free(cur);free(lists);
    nedges=IVL_tsize(ivl);
    graph=Graph_new();
    Graph_init2(graph,0,(int)n,0,nedges,(int)n,nedges,ivl,NULL,NULL);
    if(mode==1) etree=orderViaND(graph,TUNE_MAXDOMAINSIZE,RNDSEED,0,NULL);
    else etree=orderViaBestOfNDandMS(graph,TUNE_MAXDOMAINSIZE,TUNE_MAXZEROS,TUNE_MAXSIZE,RNDSEED,0,NULL);
    n2o=ETree_newToOldVtxPerm(etree);
    ent=IV_entries(n2o);
    for(i=0;i<(int)n;i++) perm[i]=(uint32_t)ent[i];
    IV_free(n2o);
    ETree_free(etree);
    Graph_free(graph);
    return 0;
}

static void fill_values(uint32_t nnz,const uint32_t *src,double *ad,double *au,double *adb,double *aub,double sig,double *vals)
{
    uint32_t q;
    if(sig==0.){
        for(q=0;q<nnz;q++){
            uint32_t s=src[q];
            vals[q]=(s&DIAGBIT)?ad[s&~DIAGBIT]:au[s];
        }
    }else{
        for(q=0;q<nnz;q++){
            uint32_t s=src[q];
            vals[q]=(s&DIAGBIT)?ad[s&~DIAGBIT]-sig*adb[s&~DIAGBIT]:au[s]-sig*aub[s];
        }
    }
}

/* y = A x and rs = sum_j |a_ij x_j| for the matrix as CalculiX handed it over (all formats that this file uses; the
   same entries as the InpMtx object of spooles_factor) */

#define FEA_ENTRY(R,C,V) do{v=(V);y[R]+=v*x[C];rs[R]+=fabs(v*x[C]);\
	if((sym)&&((R)!=(C))){y[C]+=v*x[R];rs[C]+=fabs(v*x[R]);}\
	}while(0)

static void faer_matvec(double *ad,double *au,double *adb,double *aub,double s,ITG *icol,ITG *irow,ITG n,ITG nzs,
                        int sym,int fmt,ITG nzs3,double *x,double *y,double *rs)
{
    ITG col,row,ipo,ipoint=0;
    double v;
    if(fmt==0){
        for(col=0;col<n;col++){
            if(s==0.){FEA_ENTRY(col,col,ad[col]);}
            else{FEA_ENTRY(col,col,ad[col]-s*adb[col]);}
            for(ipo=ipoint;ipo<ipoint+icol[col];ipo++){
                row=irow[ipo]-1;
                if(s==0.){FEA_ENTRY(col,row,au[ipo]);}
                else{FEA_ENTRY(col,row,au[ipo]-s*aub[ipo]);}
            }
            ipoint+=icol[col];
        }
    }else if(fmt==1){
        for(col=0;col<n;col++){
            if(s==0.){FEA_ENTRY(col,col,ad[col]);}
            else{FEA_ENTRY(col,col,ad[col]-s*adb[col]);}
            for(ipo=ipoint;ipo<ipoint+icol[col];ipo++){
                row=irow[ipo]-1;
                if(s==0.){
                    FEA_ENTRY(row,col,au[ipo]);
                    FEA_ENTRY(col,row,au[ipo+nzs3]);
                }else{
                    FEA_ENTRY(row,col,au[ipo]-s*aub[ipo]);
                    FEA_ENTRY(col,row,au[ipo+nzs3]-s*aub[ipo+nzs3]);
                }
            }
            ipoint+=icol[col];
        }
    }else if(fmt==3){
        for(col=0;col<n;col++){
            if(s==0.){FEA_ENTRY(col,col,ad[col]);}
            else{FEA_ENTRY(col,col,ad[col]-s*adb[col]);}
            for(ipo=ipoint;ipo<ipoint+icol[col];ipo++){
                row=irow[ipo]-1;
                if(s==0.){FEA_ENTRY(row,col,au[ipo]);}
                else{FEA_ENTRY(row,col,au[ipo]-s*aub[ipo]);}
            }
            ipoint+=icol[col];
        }
    }else{
        for(row=0;row<n;row++){
            for(ipo=irow[row];ipo<irow[row+1];ipo++){
                col=icol[ipo]-1;
                FEA_ENTRY(col,row,au[ipo]);
            }
        }
    }
    (void)nzs;
}

/* test solve with the factors in G.ctx: row-wise backward error for a fixed pseudo random right hand side (|b_i| between
   0.25 and 1); -1. if it could not be done (no memory, the solve failed); also primes the work space of the solves.
   *cond = |A|_inf |x|_inf / |b|_inf, a lower bound of the condition number of the matrix */
static double faer_check(double *ad,double *au,double *adb,double *aub,double s,ITG *icol,ITG *irow,ITG n,ITG nzs,
                         int sym,int fmt,ITG nzs3,double *cond)
{
    unsigned int seed=12345u;
    double *b0,*x0,*y,*rs,d,eta=0.,e,xmax=0.,bmax=0.,anorm=0.;
    ITG i;
    b0=(double *)malloc((size_t)n*sizeof(double));
    x0=(double *)malloc((size_t)n*sizeof(double));
    y=(double *)calloc((size_t)n,sizeof(double));
    rs=(double *)calloc((size_t)n,sizeof(double));
    if((b0==NULL)||(x0==NULL)||(y==NULL)||(rs==NULL)){free(b0);free(x0);free(y);free(rs);return -1.;}
    for(i=0;i<n;i++){
        seed=seed*1103515245u+12345u;
        d=0.25+0.75*(((seed>>8)&0xffff)/65536.);
        seed=seed*1103515245u+12345u;
        b0[i]=((seed>>16)&1u)?d:-d;
        x0[i]=b0[i];
    }
    if(fea_faer_solve(G.ctx,x0,1)!=FEA_FAER_OK){free(b0);free(x0);free(y);free(rs);return -1.;}
    faer_matvec(ad,au,adb,aub,s,icol,irow,n,nzs,sym,fmt,nzs3,x0,y,rs);
    for(i=0;i<n;i++){
        e=fabs(y[i]-b0[i])/(rs[i]+fabs(b0[i]));
        if(e!=e) e=1.e300;
        if(e>eta) eta=e;
        d=fabs(x0[i]);
        if((d>xmax)||(d!=d)) xmax=d;
        if(fabs(b0[i])>bmax) bmax=fabs(b0[i]);
    }
    *cond=0.;
    if((eta<1.e299)&&(xmax==xmax)){                    /* |A|_inf: the largest row sum of |a_ij| (the same pass with a vector of ones) */
        for(i=0;i<n;i++){x0[i]=1.;y[i]=0.;rs[i]=0.;}
        faer_matvec(ad,au,adb,aub,s,icol,irow,n,nzs,sym,fmt,nzs3,x0,y,rs);
        for(i=0;i<n;i++) if(rs[i]>anorm) anorm=rs[i];
        *cond=anorm*xmax/bmax;
    }else *cond=1.e300;
    free(b0);free(x0);free(y);free(rs);
    return eta;
}

static const char *status_text(int st)
{
    switch(st){
    case FEA_FAER_NOT_PD: return "the matrix is not positive definite";
    case FEA_FAER_ZERO_PIVOT: return "zero pivot (singular matrix)";
    case FEA_FAER_NOMEM: return "not enough memory";
    case FEA_FAER_TOO_BIG: return "the matrix does not fit the indices of the library";
    case FEA_FAER_PANIC: return "internal error of the library";
    default: return "the library refused the matrix";
    }
}

static const char *method_name(int sym,int method)
{
    if(!sym) return "LU";
    return method==FEA_FAER_LLT?"Cholesky":(method==FEA_FAER_LDLT?"LDL^T":"LBL^T");
}

static void drop_numeric(void)
{
    if(G.ctx) fea_faer_drop_numeric(G.ctx);
    G.have_num=0;
}

int faerc_active(void)
{
    return G.have_num;
}

int faerc_try_factor(double *ad,double *au,double *adb,double *aub,double *sigma,ITG *icol,ITG *irow,ITG *neq,ITG *nzs,
                     ITG *symmetryflag,ITG *inputformat,ITG *nzs3)
{
    ITG n=*neq,fmt=*inputformat;
    int sym,method,st,nth,forced=-1,ordmode=1,same=0,bad;
    const char *why=NULL,*fm;
    double t0,t1,t2,t3,t4,eta=0.,tol,sig,cond=0.,condmax;
    uint64_t key=0;
    uint32_t *row=NULL;
    double *vals=NULL,info[4];
    char *e;

    drop_numeric();                              /* the factors of an earlier factorization that was not cleaned up */
    if(G.off) return 0;
    if(env_int("FEA_FAER",1)==0) return 0;
    if((fmt==0)&&(*symmetryflag==0)) sym=1;
    else if(((fmt==1)||(fmt==3)||(fmt==4))&&(*symmetryflag==2)){
        if(env_int("FEA_FAER_UNSYM",0)==0) return 0;           /* the LU of faer is slower than SPOOLES (see the header): SPOOLES unless asked */
        sym=0;
    }else return 0;                              /* dense matrices of the radiation, anything unusual: SPOOLES */
    if(n<env_int("FEA_FAER_MIN_EQ",2000)) return 0;
    sig=(fmt==4)?0.:*sigma;

    e=getenv("FEA_FAER_ORDER");        /* nd (default: nested dissection of SPOOLES), bestof (nd or multisection, the better), amd (faer) */
    if((e!=NULL)&&(strcmp(e,"bestof")==0)) ordmode=0;
    else if((e!=NULL)&&(strcmp(e,"amd")==0)) ordmode=2;
    if(!G.inited){
        fea_faer_init(faer_threads());
        fea_faer_set_solve_threads(env_int("FEA_FAER_SOLVE_THREADS",1));
        e=getenv("FEA_FAER_LU");           /* auto (default) | supernodal | simplicial: the LU factorization of unsymmetric matrices (measurements) */
        if(e!=NULL) fea_faer_set_lu_mode(strcmp(e,"supernodal")==0?1:(strcmp(e,"simplicial")==0?2:0));
        G.inited=1;
    }
    nth=faer_threads();

    t0=omp_get_wtime();
    bad=check_counts((int)fmt,n,nzs[0],icol,irow,NULL,&why);        /* the sizes first: the digest below reads icol / irow with them */
    if(!bad){
        key=mix(mix(mix(mix(14695981039346656037ULL,(uint64_t)n),(uint64_t)nzs[0]),(uint64_t)fmt),(uint64_t)sym);
        if(fmt==4){
            key=hash_ints(key,icol,(size_t)*nzs);
            key=hash_ints(key,irow,(size_t)n+1);
        }else{
            key=hash_ints(key,icol,(size_t)n);
            key=hash_ints(key,irow,(size_t)*nzs);
        }
        same=(G.valid&&(G.key==key)&&(G.n==n)&&(G.nzs==nzs[0])&&(G.sym==sym));
        if(same&&declined_skip(&G.cond_declined[sig!=0.],&G.cond_skips[sig!=0.])) return 0;     /* the guard declined this pattern: SPOOLES at once, no banner of faer */
    }
    printf(" Factoring the system of equations using the %s faer solver\n",sym?"symmetric":"unsymmetric");
    printf(" Using up to %d cpu(s) for faer.\n\n",nth);
    fflush(stdout);
    if(bad){
        printf(" FEA faer: %s - the system of equations is solved with SPOOLES.\n\n",why);
        return 0;
    }
    if(!same){
        uint32_t *perm=NULL;
        free_pattern();
        G.prefer[0]=G.prefer[1]=FEA_FAER_LLT;    /* another matrix: positive definite until shown otherwise */
        if(build_pattern((int)fmt,n,*nzs,*nzs3,icol,irow,&G.colptr,&row,&G.src,&G.nnz,&why)){
            printf(" FEA faer: %s - the system of equations is solved with SPOOLES.\n\n",why);
            return 0;
        }
        t1=omp_get_wtime();
        if(sym&&(ordmode!=2)){
            perm=(uint32_t *)malloc((size_t)n*sizeof(uint32_t));
            if((perm==NULL)||order_symmetric(n,G.colptr,row,perm,ordmode)){
                free(perm);free(row);free_pattern();
                printf(" FEA faer: no memory for the ordering - the system of equations is solved with SPOOLES.\n\n");
                return 0;
            }
        }
        t2=omp_get_wtime();
        G.ctx=fea_faer_new((unsigned)n,G.colptr,row,perm,sym,&st);
        free(perm);free(row);
        if(G.ctx==NULL){
            free_pattern();
            printf(" FEA faer: the symbolic analysis failed (%s) - the system of equations is solved with SPOOLES.\n\n",status_text(st));
            return 0;
        }
        t3=omp_get_wtime();
        G.valid=1;G.key=key;G.n=n;G.nzs=nzs[0];G.sym=sym;
        fea_faer_info(G.ctx,info);
        G.flen=info[0];
        printf(" faer: %d equations, %u stored entries, ordering %.2f s, symbolic analysis %.2f s, factor %.0f million entries (%s%.0f supernodes)\n",
               (int)n,(unsigned)G.nnz,t2-t1,t3-t2,info[0]/1.e6,info[1]!=0.?"":"simplicial, ",info[2]);
        t0=t1;                                   /* the time of the analysis is reported above, not again */
    }

    vals=(double *)malloc((size_t)G.nnz*sizeof(double));
    if(vals==NULL){
        printf(" FEA faer: no memory for the values - the system of equations is solved with SPOOLES.\n\n");
        return 0;
    }
    fill_values(G.nnz,G.src,ad,au,adb,aub,sig,vals);
    e=getenv("FEA_FAER_INJECT");
    if((e!=NULL)&&(e[0]=='1')&&(G.nnz>0)) vals[G.colptr[n/2]]*=1.+1.e-4;   /* TEST ONLY (the program removes the variable unless FEA_DEV=1): the factors are those of another matrix - the check must catch it */

    e=getenv("FEA_FAER_METHOD");
    if(sym&&(e!=NULL)){
        if(strcmp(e,"llt")==0) forced=FEA_FAER_LLT;
        else if(strcmp(e,"ldlt")==0) forced=FEA_FAER_LDLT;
        else if(strcmp(e,"lblt")==0) forced=FEA_FAER_LBLT;
    }
    method=sym?(forced>=0?forced:G.prefer[sig!=0.]):FEA_FAER_LLT;
    tol=env_tol();
    condmax=env_cond();
    t3=omp_get_wtime();
    for(;;){
        st=fea_faer_factor(G.ctx,vals,method);
        if(((st==FEA_FAER_NOT_PD)||(st==FEA_FAER_ZERO_PIVOT))&&sym&&(forced<0)&&(method<FEA_FAER_LBLT)){
            /* the ladder of a symmetric matrix: Cholesky, then LDL^T without pivoting, then LBL^T with the pivoting inside the supernodes (slower), then SPOOLES */
            printf(" faer: %s fails (%s), factorized again as %s.\n",method_name(sym,method),status_text(st),method_name(sym,method+1));
            method++;
            continue;
        }
        if(st!=FEA_FAER_OK){
            printf(" FEA faer: the factorization failed (%s) - the system of equations is solved with SPOOLES.\n\n",status_text(st));
            if(st==FEA_FAER_PANIC) G.off=1;
            free(vals);drop_numeric();
            if(G.off) free_pattern();
            return 0;
        }
        t4=omp_get_wtime();
        G.have_num=1;G.method=method;
        if(tol>0.){
            eta=faer_check(ad,au,adb,aub,sig,icol,irow,n,*nzs,sym,(int)fmt,*nzs3,&cond);
            if(eta<0.){
                printf(" *WARNING in faer: the factors could not be checked (no memory for the test solve) - the system of equations is solved with SPOOLES.\n\n");
                free(vals);drop_numeric();
                return 0;
            }
            if(eta>tol){
                if(sym&&(forced<0)&&(method<FEA_FAER_LBLT)){
                    printf(" faer: the %s factors fail the check (relative residual of a test solve %.3e, limit %.1e), factorized again as %s.\n",method_name(sym,method),eta,tol,method_name(sym,method+1));
                    drop_numeric();method++;
                    continue;
                }
                printf(" *WARNING in faer: the factors fail the check of their result (relative residual of a test solve %.3e, limit %.1e) - the system of equations is solved with SPOOLES.\n\n",eta,tol);
                free(vals);drop_numeric();
                return 0;
            }
            if((condmax>0.)&&(cond>condmax)){
                printf(" faer: the system of equations is numerically singular (its condition number is at least %.1e, limit %.1e) - SPOOLES solves it as before"
                       " (and the next systems with this pattern, until it changes; faer tries again after %d of them).\n\n",cond,condmax,FAER_RETRY_AFTER);
                fflush(stdout);
                G.cond_declined[sig!=0.]=1;G.cond_skips[sig!=0.]=0;
                free(vals);drop_numeric();
                return 0;
            }
        }else{
            double *tb=(double *)calloc((size_t)n,sizeof(double));     /* primes the work space of the solves (no allocation at the first solve) */
            if(tb==NULL||fea_faer_solve(G.ctx,tb,1)!=FEA_FAER_OK){
                free(tb);free(vals);drop_numeric();
                printf(" FEA faer: no memory for the solves - the system of equations is solved with SPOOLES.\n\n");
                return 0;
            }
            free(tb);
        }
        break;
    }
    free(vals);
    if(sym) G.prefer[sig!=0.]=method;
    fm=method_name(sym,method);
    if(env_int("FEA_FAER_VERBOSE",0)!=0){
        printf(" faer: %s factorization %.2f s, check %.2f s (relative residual %.1e, condition number >= %.1e), total %.2f s\n",fm,t4-t3,omp_get_wtime()-t4,eta,cond,omp_get_wtime()-t0);
    }else{
        printf(" faer: %s factorization %.2f s, relative residual of the check %.1e\n",fm,t4-t3,eta);
    }
    printf("\n");
    fflush(stdout);
    return 1;
}

void faerc_solve(double *b,ITG *neq)
{
    int st;
    double t0=omp_get_wtime();
    if(!G.have_num||(G.n!=*neq)){
        fprintf(stderr," *ERROR in faer: solve without factors (internal error)\n");
        exit(-1);
    }
    st=fea_faer_solve(G.ctx,b,1);
    if(st!=FEA_FAER_OK){
        fprintf(stderr," *ERROR in faer: the solve failed (%s)\n",status_text(st));
        exit(-1);
    }
    G.nsolve++;
    G.tsolve+=omp_get_wtime()-t0;
}

void faerc_cleanup(void)
{
    if((G.nsolve>=5)||((G.nsolve>0)&&(env_int("FEA_FAER_VERBOSE",0)!=0))){
        printf(" faer: %d solves with these factors, %.2f s\n\n",G.nsolve,G.tsolve);
        fflush(stdout);
    }
    G.nsolve=0;G.tsolve=0.;
    drop_numeric();
}

void faerc_marker(void)
{
    printf("FEA_FAER=1 (%s: symmetric systems (Cholesky, LDL^T, LBL^T) with the nested dissection of SPOOLES, unsymmetric ones with SPOOLES (faer LU on request); every factorization is checked by a test solve, SPOOLES takes over when it fails)\n",
           fea_faer_version());
}

#endif
