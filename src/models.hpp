// models.hpp -- classic toy HMMs used by the tests and examples.
#pragma once
#include "viterbi.hpp"

namespace models {

// Healthy/Fever example (Wikipedia). States: 0=Healthy, 1=Fever.
// Observations: 0=normal, 1=cold, 2=dizzy.
inline hmm::HMM weather() {
    return hmm::HMM({0.6, 0.4},
                    {{0.7, 0.3}, {0.4, 0.6}},
                    {{0.5, 0.4, 0.1}, {0.1, 0.3, 0.6}});
}

// Dishonest casino (Durbin et al., "Biological Sequence Analysis").
// States: 0=Fair, 1=Loaded. Observations: die faces 1..6 stored as 0..5.
// Fair die: uniform. Loaded die: 6 with prob 1/2, others 1/10.
// Switching: Fair->Loaded 0.05, Loaded->Fair 0.10.
inline hmm::HMM casino() {
    const double f = 1.0 / 6.0;
    return hmm::HMM({0.5, 0.5},
                    {{0.95, 0.05}, {0.10, 0.90}},
                    {{f, f, f, f, f, f}, {0.1, 0.1, 0.1, 0.1, 0.1, 0.5}});
}

}  // namespace models
