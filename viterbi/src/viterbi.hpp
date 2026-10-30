// viterbi.hpp -- log-space Viterbi decoder for Hidden Markov Models.
//
// Header-only so every program (tests, benchmark, POS tagger) shares exactly
// one implementation.
#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hmm {

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

// log(0) = -inf is exactly what we want: an impossible event can never win a max.
// (-inf + x = -inf, and we never add +inf, so no NaNs can appear.)
inline double safe_log(double p) { return p > 0.0 ? std::log(p) : NEG_INF; }

struct Result {
    std::vector<int> path;  // most likely hidden-state sequence
    double log_prob;        // log P(path, observations); -inf if impossible
};

// ---------------------------------------------------------------------------
// A discrete HMM  lambda = (pi, A, B)
//   pi[i]   = P(state_0 = i)
//   A[i][j] = P(state_{t+1} = j | state_t = i)
//   B[i][k] = P(obs_t = k | state_t = i)
// Log versions are precomputed once at construction.
// ---------------------------------------------------------------------------
struct HMM {
    int N = 0;  // number of hidden states
    int M = 0;  // number of observation symbols
    Vec pi;
    Mat A, B;
    Vec log_pi;
    Mat log_A, log_B;

    HMM() = default;
    HMM(Vec pi_, Mat A_, Mat B_)
        : N((int)pi_.size()), M(B_.empty() ? 0 : (int)B_[0].size()),
          pi(std::move(pi_)), A(std::move(A_)), B(std::move(B_)) {
        validate();
        log_pi.resize(N);
        log_A.assign(N, Vec(N));
        log_B.assign(N, Vec(M));
        for (int i = 0; i < N; ++i) {
            log_pi[i] = safe_log(pi[i]);
            for (int j = 0; j < N; ++j) log_A[i][j] = safe_log(A[i][j]);
            for (int k = 0; k < M; ++k) log_B[i][k] = safe_log(B[i][k]);
        }
    }

private:
    static void check_dist(const Vec& v, const char* what) {
        double s = 0;
        for (double x : v) {
            if (x < 0) throw std::invalid_argument(std::string(what) + ": negative probability");
            s += x;
        }
        if (std::fabs(s - 1.0) > 1e-6)
            throw std::invalid_argument(std::string(what) + ": does not sum to 1");
    }
    void validate() const {
        if (N == 0) throw std::invalid_argument("HMM needs at least one state");
        if ((int)A.size() != N || (int)B.size() != N)
            throw std::invalid_argument("A and B must have N rows");
        check_dist(pi, "pi");
        for (const auto& r : A) {
            if ((int)r.size() != N) throw std::invalid_argument("A must be N x N");
            check_dist(r, "row of A");
        }
        for (const auto& r : B) {
            if ((int)r.size() != M) throw std::invalid_argument("B rows must all have M entries");
            check_dist(r, "row of B");
        }
    }
};

// T x N matrix of per-step emission log-probs: out[t][j] = log P(o_t | state j).
// Decoupling "how emissions are scored" from the DP lets the POS tagger reuse
// the very same decoder with word-dependent emission models.
inline Mat emission_matrix(const HMM& m, const std::vector<int>& obs) {
    Mat E(obs.size(), Vec(m.N));
    for (size_t t = 0; t < obs.size(); ++t) {
        if (obs[t] < 0 || obs[t] >= m.M) throw std::out_of_range("observation symbol out of range");
        for (int j = 0; j < m.N; ++j) E[t][j] = m.log_B[j][obs[t]];
    }
    return E;
}

// ---------------------------------------------------------------------------
// Core Viterbi.  O(T * N^2) time, O(T * N) memory (back-pointers); the delta
// table itself only needs two rows.
//
//   delta_t(j) = max_i [ delta_{t-1}(i) + log A[i][j] ] + log E[t][j]
//   psi_t(j)   = argmax_i of the same bracket
//
// Ties are broken toward the lowest state index (strict '>').
// ---------------------------------------------------------------------------
inline Result viterbi_decode(const Vec& log_pi, const Mat& log_A, const Mat& log_E) {
    const size_t T = log_E.size();
    const int N = (int)log_pi.size();
    if (T == 0) return {{}, 0.0};

    Vec prev(N), cur(N);
    std::vector<int> psi(T * (size_t)N, 0);  // psi[t*N + j]

    for (int j = 0; j < N; ++j) prev[j] = log_pi[j] + log_E[0][j];

    for (size_t t = 1; t < T; ++t) {
        for (int j = 0; j < N; ++j) {
            double best = NEG_INF;
            int arg = 0;
            for (int i = 0; i < N; ++i) {
                double s = prev[i] + log_A[i][j];
                if (s > best) { best = s; arg = i; }
            }
            cur[j] = best + log_E[t][j];
            psi[t * N + j] = arg;
        }
        prev.swap(cur);
    }

    // termination: best final state
    int last = 0;
    double best = prev[0];
    for (int j = 1; j < N; ++j)
        if (prev[j] > best) { best = prev[j]; last = j; }

    // backtrack
    std::vector<int> path(T);
    path[T - 1] = last;
    for (size_t t = T - 1; t > 0; --t) path[t - 1] = psi[t * N + path[t]];
    return {std::move(path), best};
}

// Convenience wrapper for a discrete HMM and an observation sequence.
inline Result viterbi(const HMM& m, const std::vector<int>& obs) {
    return viterbi_decode(m.log_pi, m.log_A, emission_matrix(m, obs));
}

}  // namespace hmm
