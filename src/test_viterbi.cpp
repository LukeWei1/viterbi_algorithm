// test_viterbi.cpp -- unit tests: Viterbi vs. brute force on toy and random HMMs.
#include <cstdio>
#include <random>
#include "viterbi.hpp"
#include "brute_force.hpp"
#include "models.hpp"

using namespace hmm;

static int g_checks = 0, g_fail = 0;
#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        ++g_checks;                                                               \
        if (!(cond)) { ++g_fail; std::printf("  FAIL (%s:%d): %s\n", __FILE__, __LINE__, msg); } \
    } while (0)

// Exact comparison: same path AND same score. Valid when the optimum is unique.
static bool same_result(const Result& a, const Result& b) {
    return a.path == b.path && std::fabs(a.log_prob - b.log_prob) < 1e-9;
}

// log P(path, obs) recomputed independently from a path.
static double score_path(const HMM& m, const std::vector<int>& obs, const std::vector<int>& path) {
    double s = m.log_pi[path[0]] + m.log_B[path[0]][obs[0]];
    for (size_t t = 1; t < obs.size(); ++t)
        s += m.log_A[path[t - 1]][path[t]] + m.log_B[path[t]][obs[t]];
    return s;
}

// Tie-tolerant comparison: two different paths can have exactly the same
// probability (e.g. same transitions and emissions in a different order), so we
// require (1) equal optimal score and (2) the path Viterbi returns really
// achieves that score.
static bool equivalent(const HMM& m, const std::vector<int>& obs, const Result& v, const Result& b) {
    if (obs.empty()) return v.path.empty() && b.path.empty();
    if (std::isinf(v.log_prob) || std::isinf(b.log_prob)) return v.log_prob == b.log_prob;
    return std::fabs(v.log_prob - b.log_prob) < 1e-9 &&
           std::fabs(score_path(m, obs, v.path) - v.log_prob) < 1e-9;
}

static void test_weather_textbook() {
    std::printf("[weather] textbook example\n");
    auto m = models::weather();
    std::vector<int> obs = {0, 1, 2};  // normal, cold, dizzy
    Result v = viterbi(m, obs);
    CHECK((v.path == std::vector<int>{0, 0, 1}), "path should be Healthy,Healthy,Fever");
    CHECK(std::fabs(std::exp(v.log_prob) - 0.01512) < 1e-12, "P(best path, obs) should be 0.01512");
    CHECK(same_result(v, brute_force(m, obs)), "matches brute force");
}

static void test_weather_exhaustive() {
    std::printf("[weather] every observation sequence of length 1..6\n");
    auto m = models::weather();
    int n_seq = 0;
    for (int T = 1; T <= 6; ++T) {
        int total = 1;
        for (int i = 0; i < T; ++i) total *= m.M;
        for (int code = 0; code < total; ++code) {
            std::vector<int> obs(T);
            for (int t = 0, c = code; t < T; ++t, c /= m.M) obs[t] = c % m.M;
            CHECK(same_result(viterbi(m, obs), brute_force(m, obs)), "weather mismatch");
            ++n_seq;
        }
    }
    std::printf("  compared %d sequences\n", n_seq);
}

static void test_casino() {
    std::printf("[casino] hand-made rolls (T=20) vs brute force\n");
    auto m = models::casino();
    // faces 1..6 -> 0..5.   a run of sixes in the middle should look "loaded"
    std::vector<int> rolls = {3, 1, 5, 2, 4, 3, 1, 2, 5, 4, 6, 6, 6, 6, 6, 6, 3, 1, 4, 2};
    std::vector<int> obs;
    for (int r : rolls) obs.push_back(r - 1);
    Result v = viterbi(m, obs), b = brute_force(m, obs);
    CHECK(same_result(v, b), "casino mismatch");
    std::printf("  rolls : ");
    for (int r : rolls) std::printf("%d", r);
    std::printf("\n  states: ");
    for (int s : v.path) std::printf("%c", s == 0 ? 'F' : 'L');
    std::printf("   (F=fair, L=loaded)\n");
    int loaded_in_run = 0;
    for (int t = 10; t <= 15; ++t) loaded_in_run += v.path[t];
    CHECK(loaded_in_run >= 5, "a run of six 6s should be decoded as (mostly) loaded");
    CHECK(v.path[0] == 0 && v.path[19] == 0, "ordinary-looking rolls at the ends should be Fair");
}

static Vec random_dist(std::mt19937& rng, int n) {
    std::uniform_real_distribution<double> u(0.05, 1.0);
    Vec v(n);
    double s = 0;
    for (auto& x : v) { x = u(rng); s += x; }
    for (auto& x : v) x /= s;
    return v;
}

static void test_random_hmms() {
    std::printf("[random] 500 random HMMs vs brute force (tie-tolerant)\n");
    std::mt19937 rng(12345);
    int bad = 0;
    for (int trial = 0; trial < 500; ++trial) {
        int N = 2 + rng() % 4;  // 2..5 states
        int M = 2 + rng() % 5;  // 2..6 symbols
        int T = 1 + rng() % 8;  // 1..8 steps
        Mat A, B;
        for (int i = 0; i < N; ++i) { A.push_back(random_dist(rng, N)); B.push_back(random_dist(rng, M)); }
        HMM m(random_dist(rng, N), A, B);
        std::vector<int> obs(T);
        for (auto& o : obs) o = rng() % M;
        if (!equivalent(m, obs, viterbi(m, obs), brute_force(m, obs))) ++bad;
    }
    CHECK(bad == 0, "random HMM mismatch");
    std::printf("  mismatches: %d\n", bad);
}

static void test_edge_cases() {
    std::printf("[edge] T=0, T=1, zero probabilities, impossible input\n");
    auto m = models::weather();
    CHECK(viterbi(m, {}).path.empty(), "empty observation -> empty path");

    Result one = viterbi(m, {2});  // dizzy at t=0: Healthy .6*.1=.06, Fever .4*.6=.24
    CHECK(one.path == std::vector<int>{1}, "T=1 picks Fever");

    // Forbidden transitions: state 0 can never go to 1 and vice versa.
    HMM stuck({0.5, 0.5}, {{1.0, 0.0}, {0.0, 1.0}}, {{0.5, 0.5}, {0.5, 0.5}});
    Result s = viterbi(stuck, {0, 1, 0, 1});
    bool constant = true;
    for (int x : s.path) constant &= (x == s.path[0]);
    CHECK(constant, "path must not use zero-probability transitions");

    // Observation impossible under every state -> log-prob must be -inf, no NaN.
    HMM det({1.0, 0.0}, {{0.5, 0.5}, {0.5, 0.5}}, {{1.0, 0.0}, {1.0, 0.0}});
    Result imp = viterbi(det, {0, 1});
    CHECK(std::isinf(imp.log_prob) && imp.log_prob < 0, "impossible sequence -> -inf");
    CHECK(!std::isnan(imp.log_prob), "no NaN");
}

static void test_underflow() {
    std::printf("[underflow] long casino sequence (T=100000)\n");
    auto m = models::casino();
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0, 1);
    const int T = 100000;
    std::vector<int> truth(T), obs(T);
    int s = 0;
    for (int t = 0; t < T; ++t) {
        s = (u(rng) < m.A[s][1]) ? 1 : 0;
        truth[t] = s;
        double r = u(rng), acc = 0;
        int sym = m.M - 1;
        for (int k = 0; k < m.M; ++k) { acc += m.B[s][k]; if (r < acc) { sym = k; break; } }
        obs[t] = sym;
    }
    Result v = viterbi(m, obs);
    CHECK(std::isfinite(v.log_prob), "log-prob finite");
    CHECK(std::exp(v.log_prob) == 0.0, "(sanity) plain probabilities WOULD have underflowed to 0");
    int agree = 0, always_fair = 0;
    for (int t = 0; t < T; ++t) { agree += (v.path[t] == truth[t]); always_fair += (truth[t] == 0); }
    std::printf("  log P = %.1f  (e^that underflows to 0 in double precision)\n", v.log_prob);
    std::printf("  state accuracy vs. truth: %.1f%%   (always-guess-Fair: %.1f%%)\n",
                100.0 * agree / T, 100.0 * always_fair / T);
    CHECK(agree > always_fair, "Viterbi should beat the always-Fair guess");
}

int main() {
    test_weather_textbook();
    test_weather_exhaustive();
    test_casino();
    test_random_hmms();
    test_edge_cases();
    test_underflow();
    std::printf("\n%d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
