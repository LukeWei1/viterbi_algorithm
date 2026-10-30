// benchmark.cpp -- runtime of Viterbi vs. brute force.
// Writes three CSVs into results/ (see scripts/plot_benchmark.py):
//   bench_vs_brute.csv : fixed N, T = 1..BRUTE_MAX_T       (exponential vs. linear)
//   bench_scale_T.csv  : fixed N, T doubling up to ~256k   (Viterbi is linear in T)
//   bench_scale_N.csv  : fixed T, N doubling up to 256     (Viterbi is quadratic in N)
// Timing covers the decoding DP only; emission matrices are built beforehand.
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include "viterbi.hpp"
#include "brute_force.hpp"

using namespace hmm;
using Clock = std::chrono::steady_clock;

static volatile double g_sink = 0;  // stops the optimiser deleting the work

static Vec random_log_dist(std::mt19937& rng, int n) {
    std::uniform_real_distribution<double> u(0.05, 1.0);
    Vec v(n);
    double s = 0;
    for (auto& x : v) { x = u(rng); s += x; }
    for (auto& x : v) x = std::log(x / s);
    return v;
}

struct Problem { Vec log_pi; Mat log_A, log_E; };

static Problem make_problem(int N, int T, unsigned seed) {
    std::mt19937 rng(seed);
    Problem p;
    p.log_pi = random_log_dist(rng, N);
    for (int i = 0; i < N; ++i) p.log_A.push_back(random_log_dist(rng, N));
    for (int t = 0; t < T; ++t) p.log_E.push_back(random_log_dist(rng, N));
    return p;
}

// Run f repeatedly until >= min_seconds have elapsed; return mean seconds per call.
template <class F>
static double time_it(F f, double min_seconds = 0.05) {
    int reps = 0;
    auto t0 = Clock::now();
    double elapsed = 0;
    do {
        f();
        ++reps;
        elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
    } while (elapsed < min_seconds);
    return elapsed / reps;
}

int main() {
    const int N_FIXED = 3;
    const int BRUTE_MAX_T = 14;  // 3^14 = 4.8M sequences

    // --- 1. Viterbi vs brute force, growing T ------------------------------
    {
        FILE* f = std::fopen("results/bench_vs_brute.csv", "w");
        std::fprintf(f, "T,viterbi_s,brute_s\n");
        std::printf("== Viterbi vs brute force (N=%d) ==\n%4s %14s %14s %10s\n", N_FIXED, "T", "viterbi(s)", "brute(s)", "speedup");
        for (int T = 1; T <= BRUTE_MAX_T; ++T) {
            Problem p = make_problem(N_FIXED, T, 100 + T);
            Result v = viterbi_decode(p.log_pi, p.log_A, p.log_E);
            Result b = brute_force_decode(p.log_pi, p.log_A, p.log_E);
            if (std::fabs(v.log_prob - b.log_prob) > 1e-9) { std::fprintf(stderr, "MISMATCH at T=%d\n", T); return 1; }
            double tv = time_it([&] { g_sink = viterbi_decode(p.log_pi, p.log_A, p.log_E).log_prob; });
            double tb = time_it([&] { g_sink = brute_force_decode(p.log_pi, p.log_A, p.log_E).log_prob; });
            std::fprintf(f, "%d,%.9g,%.9g\n", T, tv, tb);
            std::printf("%4d %14.3e %14.3e %9.0fx\n", T, tv, tb, tb / tv);
        }
        std::fclose(f);
    }

    // --- 2. Viterbi scaling in T --------------------------------------------
    {
        FILE* f = std::fopen("results/bench_scale_T.csv", "w");
        std::fprintf(f, "T,viterbi_s\n");
        std::printf("\n== Viterbi scaling in T (N=5) ==\n%10s %14s %16s\n", "T", "time(s)", "ns per step");
        for (int T = 1000; T <= 256000; T *= 2) {
            Problem p = make_problem(5, T, 7);
            double tv = time_it([&] { g_sink = viterbi_decode(p.log_pi, p.log_A, p.log_E).log_prob; });
            std::fprintf(f, "%d,%.9g\n", T, tv);
            std::printf("%10d %14.3e %16.1f\n", T, tv, 1e9 * tv / T);
        }
        std::fclose(f);
    }

    // --- 3. Viterbi scaling in N --------------------------------------------
    {
        FILE* f = std::fopen("results/bench_scale_N.csv", "w");
        std::fprintf(f, "N,viterbi_s\n");
        std::printf("\n== Viterbi scaling in N (T=500) ==\n%6s %14s %18s\n", "N", "time(s)", "ns per (t,i,j)");
        for (int N = 2; N <= 256; N *= 2) {
            Problem p = make_problem(N, 500, 9);
            double tv = time_it([&] { g_sink = viterbi_decode(p.log_pi, p.log_A, p.log_E).log_prob; });
            std::fprintf(f, "%d,%.9g\n", N, tv);
            std::printf("%6d %14.3e %18.3f\n", N, tv, 1e9 * tv / (500.0 * N * N));
        }
        std::fclose(f);
    }
    return 0;
}
