/* Scenario test of the pure helpers of installer/faer_ccx/ccx/faer.c (the text between HARNESS-BEGIN and HARNESS-END is included as faer_pure.inc by
   tests/test_faer.py; no CalculiX, no SPOOLES, no Rust): the check of the column counts that must come BEFORE the pattern is hashed (review C-7) and the memory of a
   system that the condition guard declined (review C-3). */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
typedef int ITG;

#include "faer_pure.inc"

static int fails = 0;

static void expect(int cond, const char *what)
{
    if (!cond) { printf("FAIL: %s\n", what); fails++; }
}

static void counts(const char *title, int fmt, ITG n, ITG nzs, const ITG *icol, const ITG *irow, int want_rc, const char *want_why, int64_t want_tot)
{
    const char *why = (const char *)1;
    int64_t tot = -1;
    int rc = check_counts(fmt, n, nzs, icol, irow, &tot, &why);
    int ok = (rc == want_rc);
    if (want_rc == 0) ok = ok && why == NULL && tot == want_tot;
    else ok = ok && why != NULL && strcmp(why, want_why) == 0 && tot == -1;
    printf("%s %s (rc=%d, %s)\n", ok ? "ok  " : "FAIL", title, rc, why ? why : "-");
    if (!ok) fails++;
}

int main(void)
{
    const ITG icol_ok[3] = {1, 1, 0}, icol_neg[3] = {1, -1, 2};
    const ITG rows4_ok[4] = {0, 1, 2, 2}, rows4_end[4] = {0, 1, 2, 5}, rows4_start[4] = {1, 1, 2, 2};
    int declined = 0, skips = 0, i, n_skipped;

    /* the sums must agree with nzs before irow is read with nzs (the digest) */
    counts("format 0, counts add up", 0, 3, 2, icol_ok, NULL, 0, NULL, 2);
    counts("format 1, counts add up", 1, 3, 2, icol_ok, NULL, 0, NULL, 2);
    counts("format 0, nzs larger than the counts", 0, 3, 3, icol_ok, NULL, -1, "the column counts do not add up to nzs", -1);
    counts("format 3, nzs smaller than the counts", 3, 3, 1, icol_ok, NULL, -1, "the column counts do not add up to nzs", -1);
    counts("negative column count", 0, 3, 2, icol_neg, NULL, -1, "negative column count", -1);
    counts("no equations", 0, 0, 0, icol_ok, NULL, -1, "no equations", -1);
    counts("negative nzs", 0, 3, -2, icol_ok, NULL, -1, "negative nzs", -1);
    counts("format 4, row pointers fit nzs", 4, 3, 2, NULL, rows4_ok, 0, NULL, 2);
    counts("format 4, last row pointer is not nzs", 4, 3, 2, NULL, rows4_end, -1, "unexpected row pointers (input format 4)", -1);
    counts("format 4, first row pointer is not 0", 4, 3, 2, NULL, rows4_start, -1, "unexpected row pointers (input format 4)", -1);
    {   const char *why;
        expect(check_counts(0, 3, 2, icol_ok, NULL, NULL, &why) == 0 && why == NULL, "ptot may be NULL"); }

    /* a declined pattern: skipped FAER_RETRY_AFTER times, then one retry (flag cleared), and a new decline starts the count again */
    expect(declined_skip(&declined, &skips) == 0 && skips == 0, "not declined: never skipped");
    declined = 1;
    skips = 0;
    n_skipped = 0;
    for (i = 0; i < 3 * FAER_RETRY_AFTER; i++) {
        if (declined_skip(&declined, &skips)) n_skipped++;
        else { printf("retry after %d skips\n", n_skipped); break; }
    }
    expect(n_skipped == FAER_RETRY_AFTER && FAER_RETRY_AFTER == 20, "exactly 20 factorizations are skipped, the 21st tries faer again");
    expect(declined == 0 && skips == 0, "the retry clears the flag");
    expect(declined_skip(&declined, &skips) == 0, "after the retry faer is tried until the guard declines again");
    declined = 1;
    skips = 0;
    for (n_skipped = 0; declined_skip(&declined, &skips); n_skipped++) ;
    expect(n_skipped == FAER_RETRY_AFTER, "a second decline skips another 20");
    printf("\n%s\n", fails ? "SOME CHECKS FAILED" : "ALL CHECKS OK");
    return fails != 0;
}
