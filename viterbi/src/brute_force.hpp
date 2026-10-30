// brute_force.hpp -- exhaustive decoder used as a ground-truth oracle.
// Scores all N^T state sequences: O(T * N^T). Only usable for tiny T.
#pragma once
#include "viterbi.hpp"

namespace hmm {

inline Result brute_force_decode(const Vec& log_pi, const Mat& log_A, const Mat& log_E) {
    const size_t T = log_E.size();
    const int N = (int)log_pi.size();
    if (T == 0) return {{}, 0.0};

    std::vector<int> seq(T, 0), best_seq(T, 0);
    double best = NEG_INF;
    bool have_best = false;

    while (true) {
        // Same left-to-right summation order as Viterbi, so scores are
        // bit-identical and can be compared exactly.
        double s = log_pi[seq[0]] + log_E[0][seq[0]];
        for (size_t t = 1; t < T; ++t) s = (s + log_A[seq[t - 1]][seq[t]]) + log_E[t][seq[t]];

        if (!have_best || s > best) { best = s; best_seq = seq; have_best = true; }

        // odometer increment (last position changes fastest => lexicographic order)
        size_t pos = T;
        while (pos > 0) {
            --pos;
            if (++seq[pos] < N) break;
            seq[pos] = 0;
            if (pos == 0) return {best_seq, best};
        }
    }
}

inline Result brute_force(const HMM& m, const std::vector<int>& obs) {
    return brute_force_decode(m.log_pi, m.log_A, emission_matrix(m, obs));
}

}  // namespace hmm
