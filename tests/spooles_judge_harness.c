/* Scenario test of fea_judge (the body is extracted from the patched spooles.c, the neighbours are stubs). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
typedef int ITG;
struct factorinfo { int dummy; };
static int fea_agg_violation = 0;
static int fea_check_off = 0;
static double g_tol = 1.e-12;          /* what fea_check_tol returns */
static double g_eta[3];                /* what the test solve of attempt 0, 1, 2 returns, INDEXED BY THE ATTEMPT (not by the number of the call: a skipped test solve must not shift it) */
static int g_attempt = 0;              /* the attempt that is being judged (set by the loop below before fea_judge) */
static int g_calls = 0, g_free = 0;
static double fea_check_tol(void) { return g_tol; }
static double fea_check_factor(struct factorinfo *p, double *ad, double *au, double *adb, double *aub, double *sigma,
                               ITG *icol, ITG *irow, ITG *neq, ITG *nzs, int s, ITG *f, ITG o)
{ g_calls++; return g_eta[g_attempt < 3 ? g_attempt : 2]; }
static void fea_free_factor(struct factorinfo *p) { g_free++; }

#include "judge_body.inc"

/* the choice of spooles_factor / spooles_factor_rad between the factorization on the threads and the one on ONE thread: FEA_THREADED_COND is the text of the
   condition of the patch itself (the test extracts it and writes the include file), it uses the variables num_cpus, fea_attempt and fea_check_off */
#include "threaded_cond.inc"
static int num_cpus = 4;
static int fea_attempt = 0;
static int threaded(void) { return FEA_THREADED_COND; }

/* simulates the loop of spooles_factor: attempts with bad[k] = bookkeeping flag of the factorization k, eta[k] = its test solve */
static int run(const char *title, double tol, int off, const int bad[3], const double eta[3], int expect_attempts, int expect_check_calls, int expect_off_after)
{
    struct factorinfo p;
    int attempt = 0, ok = 1;
    g_tol = tol;
    fea_check_off = off;
    g_calls = 0;
    g_free = 0;
    memcpy(g_eta, eta, sizeof g_eta);
    printf("---- %s\n", title);
    for (;;) {
        fea_agg_violation = (attempt < 2) ? bad[attempt] : 0;     /* factor_MT sets the flag; the one-thread factorization never */
        g_attempt = attempt;
        if (fea_judge(&p, attempt, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)) {
            attempt++;
            if (attempt > 2) { printf("TEST ERROR: more than 3 factorizations\n"); return 0; }
            continue;
        }
        break;
    }
    printf("[attempts=%d test solves=%d freed=%d off_after=%d]\n", attempt + 1, g_calls, g_free, fea_check_off);
    if (attempt + 1 != expect_attempts || g_calls != expect_check_calls || fea_check_off != expect_off_after) { printf("FAIL\n"); ok = 0; }
    return ok;
}

/* two systems of equations in ONE process (spooles_factor is called twice, fea_check_off is static): the first fails the check on every factorization (the matrix is the
   cause), so the check is switched off - the second system must then run on ONE thread (no unchecked factorization on the threads) and needs no test solve */
static int run_later(const char *title)
{
    struct factorinfo p;
    int ok = 1, sys, threads_used[2], attempts[2], solves[2];
    const double matrix[3] = {2e-9, 2e-9, 2e-9};
    g_tol = 1.e-12;
    fea_check_off = 0;
    g_calls = 0;
    g_free = 0;
    memcpy(g_eta, matrix, sizeof g_eta);
    printf("---- %s\n", title);
    for (sys = 0; sys < 2; sys++) {
        int before = g_calls;
        threads_used[sys] = 0;
        fea_attempt = 0;
        for (;;) {
            fea_agg_violation = 0;
            g_attempt = fea_attempt;
            if (threaded()) {
                threads_used[sys]++;
                fea_agg_violation = 0;                                 /* nothing violated in this scenario */
                if (fea_judge(&p, fea_attempt, num_cpus, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)) {
                    fea_attempt++;
                    if (fea_attempt > 2) { printf("TEST ERROR: more than 3 factorizations\n"); return 0; }
                    continue;
                }
            } else if (fea_attempt >= 2) {                             /* the factorization on one thread after two failed checks is judged too */
                fea_judge(&p, fea_attempt, num_cpus, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            }
            break;
        }
        attempts[sys] = fea_attempt + 1;
        solves[sys] = g_calls - before;
        printf("[system %d: attempts=%d threaded factorizations=%d test solves=%d off_after=%d]\n", sys + 1, attempts[sys], threads_used[sys], solves[sys], fea_check_off);
    }
    if (attempts[0] != 3 || threads_used[0] != 2 || solves[0] != 3 || fea_check_off != 1) { printf("FAIL: the first system must try 2 threaded factorizations and one on one thread, all checked\n"); ok = 0; }
    if (attempts[1] != 1 || threads_used[1] != 0 || solves[1] != 0) { printf("FAIL: the later system must run once, on ONE thread, without a test solve\n"); ok = 0; }
    return ok;
}

int main(void)
{
    int allok = 1;
    const int none[3] = {0, 0, 0};
    const double good[3] = {1e-16, 1e-16, 1e-16};
    /* 1. a clean factorization: one factorization, one test solve, no output */
    allok &= run("clean", 1e-12, 0, none, good, 1, 1, 0);
    /* 2. (a) the bookkeeping is violated and the static off-switch is set by an earlier matrix: it must still be repeated (before the fix: accepted) */
    {   const int bad[3] = {1, 0, 0};
        allok &= run("bad + static off-switch (earlier matrix)", 1e-12, 1, bad, good, 2, 0, 1); }
    /* 3. the same with the test limit 0 */
    {   const int bad[3] = {1, 1, 0};
        allok &= run("bad twice + test limit 0", 0., 0, bad, good, 3, 0, 0); }
    /* 4. a lost update found by the test solve, repeat passes */
    {   const double eta[3] = {3e-5, 2e-17, 2e-17};
        allok &= run("test solve fails, threaded repeat passes", 1e-12, 0, none, eta, 2, 2, 0); }
    /* 5. fails twice, one thread passes */
    {   const double eta[3] = {3e-5, 4e-5, 2e-17};
        allok &= run("fails twice, one thread passes", 1e-12, 0, none, eta, 3, 3, 0); }
    /* 6. the matrix: fails three times */
    {   const double eta[3] = {2e-9, 2e-9, 2e-9};
        allok &= run("the matrix is the cause", 1e-12, 0, none, eta, 3, 3, 1); }
    /* 7. (b) the first attempt fails, the repeat cannot be checked (eta < 0): no "confirmed" */
    {   const double eta[3] = {3e-5, -1., 0.};
        allok &= run("fails, repeat unchecked", 1e-12, 0, none, eta, 2, 2, 0); }
    /* 8. (b) fails twice, the one-thread factorization cannot be checked */
    {   const double eta[3] = {3e-5, 4e-5, -1.};
        allok &= run("fails twice, one thread unchecked", 1e-12, 0, none, eta, 3, 3, 0); }
    /* 9. the first factorization cannot be checked and nothing is wrong with the bookkeeping: accepted, one warning */
    {   const double eta[3] = {-1., 0., 0.};
        allok &= run("first unchecked", 1e-12, 0, none, eta, 1, 1, 0); }
    /* 10. the first cannot be checked, the bookkeeping is violated: repeat; the repeat passes */
    {   const int bad[3] = {1, 0, 0};
        const double eta[3] = {-1., 2e-17, 0.};
        allok &= run("bad + first unchecked, repeat passes", 1e-12, 0, bad, eta, 2, 2, 0); }
    /* 11. the bookkeeping is violated twice with passing test solves: ends on one thread, which passes */
    {   const int bad[3] = {1, 1, 0};
        allok &= run("bad twice with good test solves", 1e-12, 0, bad, good, 3, 3, 0); }
    /* 12. (A-1, 24. 11. 2026) after the verdict "the matrix" the LATER system of the process is factorized on ONE thread, unchecked only because nothing can be checked */
    allok &= run_later("later system after the matrix verdict");
    printf("\n%s\n", allok ? "ALL SCENARIOS AS EXPECTED" : "SOME SCENARIOS FAILED");
    return !allok;
}
