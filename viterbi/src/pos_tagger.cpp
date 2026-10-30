// pos_tagger.cpp -- bigram HMM part-of-speech tagger decoded with Viterbi,
// compared against a most-common-tag baseline.
//
// Corpus format: one sentence per line, tokens "word/TAG" separated by spaces
// (the Brown corpus' native format; see scripts/export_brown.py).
//
// Usage: ./pos_tagger corpus.txt [--sentence "the dog barks ."]
//
// Model (all counts from the training split):
//   hidden states   = tags
//   pi[t]           = P(first tag of a sentence = t)
//   A[s][t]         = P(tag_i = t | tag_{i-1} = s)            add-k smoothed
//   P(word | tag)   = count(word,tag) / count(tag)            for known words
//   unknown words   -> mapped to a "signature" class (capitalised? digits?
//                      suffix -ing/-ed/-ly/...) whose emission probabilities
//                      are estimated from words seen only ONCE in training
//                      (hapax legomena stand in for unseen words).
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include "viterbi.hpp"

using namespace hmm;

struct Token { std::string word; std::string tag; };
using Sentence = std::vector<Token>;

// ---------------------------------------------------------------- corpus I/O
static std::vector<Sentence> read_corpus(const std::string& path) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    std::vector<Sentence> out;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tok;
        Sentence s;
        while (ss >> tok) {
            size_t slash = tok.rfind('/');  // last '/', since words like "1/2" contain slashes
            if (slash == std::string::npos || slash == 0 || slash + 1 == tok.size()) continue;
            s.push_back({tok.substr(0, slash), tok.substr(slash + 1)});
        }
        if (!s.empty()) out.push_back(std::move(s));
    }
    return out;
}

static std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static bool ends_with(const std::string& s, const char* suf) {
    size_t n = std::char_traits<char>::length(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// Word "shape" used for words never seen in training.
static std::string signature(const std::string& w) {
    bool has_digit = false, has_alpha = false, all_upper = true, has_hyphen = false;
    for (unsigned char c : w) {
        if (std::isdigit(c)) has_digit = true;
        if (std::isalpha(c)) { has_alpha = true; if (!std::isupper(c)) all_upper = false; }
        if (c == '-') has_hyphen = true;
    }
    if (!has_alpha && !has_digit) return "<PUNCT>";
    if (has_digit) return "<NUM>";
    if (has_alpha && all_upper && w.size() > 1) return "<ALLCAP>";
    if (std::isupper((unsigned char)w[0])) return "<CAP>";
    if (has_hyphen) return "<HYPHEN>";
    std::string l = lower(w);
    if (ends_with(l, "ing")) return "<ING>";
    if (ends_with(l, "ed")) return "<ED>";
    if (ends_with(l, "ly")) return "<LY>";
    if (ends_with(l, "tion") || ends_with(l, "ness") || ends_with(l, "ment") || ends_with(l, "ity")) return "<NOUNSUF>";
    if (ends_with(l, "able") || ends_with(l, "ible") || ends_with(l, "ous") || ends_with(l, "ful") || ends_with(l, "al")) return "<ADJSUF>";
    if (ends_with(l, "s")) return "<S>";
    return "<UNK>";
}

// ---------------------------------------------------------------- the model
struct Tagger {
    std::vector<std::string> tags;
    std::unordered_map<std::string, int> tag_id;
    Vec log_pi;
    Mat log_A;
    // emission log P(w | tag) for known words, and for signature classes
    std::unordered_map<std::string, Vec> log_emit_word;
    std::unordered_map<std::string, Vec> log_emit_sig;
    Vec log_emit_fallback;  // for a signature never seen among hapaxes

    int N() const { return (int)tags.size(); }

    void train(const std::vector<Sentence>& train, double k_trans = 0.001, double k_unk = 0.001) {
        for (auto& s : train)
            for (auto& t : s)
                if (!tag_id.count(t.tag)) { tag_id[t.tag] = (int)tags.size(); tags.push_back(t.tag); }
        const int n = N();

        std::vector<double> start(n, 0), tag_count(n, 0);
        Mat trans(n, Vec(n, 0));
        std::unordered_map<std::string, Vec> wt;  // word -> counts per tag
        std::unordered_map<std::string, double> word_total;

        for (auto& s : train) {
            for (size_t i = 0; i < s.size(); ++i) {
                int t = tag_id[s[i].tag];
                tag_count[t] += 1;
                std::string w = lower(s[i].word);
                auto it = wt.find(w);
                if (it == wt.end()) it = wt.emplace(w, Vec(n, 0)).first;
                it->second[t] += 1;
                word_total[w] += 1;
                if (i == 0) start[t] += 1;
                else trans[tag_id[s[i - 1].tag]][t] += 1;
            }
        }

        // transitions with add-k smoothing
        double start_sum = 0;
        for (double x : start) start_sum += x;
        log_pi.assign(n, 0);
        for (int j = 0; j < n; ++j) log_pi[j] = std::log((start[j] + k_trans) / (start_sum + k_trans * n));
        log_A.assign(n, Vec(n, 0));
        for (int i = 0; i < n; ++i) {
            double row = 0;
            for (double x : trans[i]) row += x;
            for (int j = 0; j < n; ++j) log_A[i][j] = std::log((trans[i][j] + k_trans) / (row + k_trans * n));
        }

        // emissions for known words (zero stays -inf: this word was never seen with this tag)
        for (auto& [w, cnt] : wt) {
            Vec lp(n);
            for (int t = 0; t < n; ++t) lp[t] = safe_log(cnt[t] / tag_count[t]);
            log_emit_word[w] = std::move(lp);
        }

        // emissions for signature classes, estimated from hapax words
        std::unordered_map<std::string, Vec> sig_counts;
        // (re-scan the corpus so the signature sees the original casing)
        for (auto& s : train)
            for (auto& tk : s) {
                if (word_total[lower(tk.word)] != 1) continue;
                std::string sg = signature(tk.word);
                auto it = sig_counts.find(sg);
                if (it == sig_counts.end()) it = sig_counts.emplace(sg, Vec(n, 0)).first;
                it->second[tag_id[tk.tag]] += 1;
            }
        const double n_sigs = (double)sig_counts.size() + 1;
        for (auto& [sg, cnt] : sig_counts) {
            Vec lp(n);
            for (int t = 0; t < n; ++t) lp[t] = std::log((cnt[t] + k_unk) / (tag_count[t] + k_unk * n_sigs));
            log_emit_sig[sg] = std::move(lp);
        }
        // A signature never seen among hapaxes carries no evidence about the tag:
        // use a constant so that only the tag transitions decide.
        log_emit_fallback.assign(n, std::log(k_unk));
    }

    bool is_known(const std::string& w) const { return log_emit_word.count(lower(w)) > 0; }

    const Vec& emission(const std::string& w) const {
        auto it = log_emit_word.find(lower(w));
        if (it != log_emit_word.end()) return it->second;
        auto s = log_emit_sig.find(signature(w));
        return s != log_emit_sig.end() ? s->second : log_emit_fallback;
    }

    std::vector<std::string> tag(const std::vector<std::string>& words) const {
        Mat E;
        E.reserve(words.size());
        for (auto& w : words) E.push_back(emission(w));
        Result r = viterbi_decode(log_pi, log_A, E);
        std::vector<std::string> out;
        for (int t : r.path) out.push_back(tags[t]);
        return out;
    }
};

// ---------------------------------------------------------------- baseline
struct Baseline {
    std::unordered_map<std::string, std::string> best;  // word -> most common tag
    std::string overall;                                // most common tag overall

    void train(const std::vector<Sentence>& train) {
        std::unordered_map<std::string, std::map<std::string, int>> c;
        std::map<std::string, int> all;
        for (auto& s : train)
            for (auto& t : s) { c[lower(t.word)][t.tag]++; all[t.tag]++; }
        for (auto& [w, m] : c) {
            auto b = std::max_element(m.begin(), m.end(), [](auto& a, auto& b) { return a.second < b.second; });
            best[w] = b->first;
        }
        overall = std::max_element(all.begin(), all.end(), [](auto& a, auto& b) { return a.second < b.second; })->first;
    }
    std::string tag(const std::string& w) const {
        auto it = best.find(lower(w));
        return it != best.end() ? it->second : overall;
    }
};

struct Score { long correct = 0, total = 0; double pct() const { return total ? 100.0 * correct / total : 0; } };

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s corpus.txt [--sentence \"...\"]\n", argv[0]); return 1; }
    std::string custom;
    for (int i = 2; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--sentence") custom = argv[i + 1];

    auto corpus = read_corpus(argv[1]);
    std::vector<Sentence> train, test;  // every 10th sentence -> test (deterministic 90/10 split)
    for (size_t i = 0; i < corpus.size(); ++i) (i % 10 == 9 ? test : train).push_back(corpus[i]);

    Tagger hmm_tagger;
    auto t0 = std::chrono::steady_clock::now();
    hmm_tagger.train(train);
    Baseline base;
    base.train(train);
    double train_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("corpus: %zu sentences  (train %zu / test %zu),  %d tags,  %zu distinct training words\n",
                corpus.size(), train.size(), test.size(), hmm_tagger.N(), hmm_tagger.log_emit_word.size());
    std::printf("training time: %.3f s\n\n", train_s);

    Score hv_all, hv_known, hv_unk, bl_all, bl_known, bl_unk;
    std::map<std::pair<std::string, std::string>, int> confusion;  // (gold, predicted)
    double decode_s = 0;
    long n_tokens = 0;
    for (auto& s : test) {
        std::vector<std::string> words;
        for (auto& t : s) words.push_back(t.word);
        auto t1 = std::chrono::steady_clock::now();
        auto pred = hmm_tagger.tag(words);
        decode_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        for (size_t i = 0; i < s.size(); ++i) {
            bool known = hmm_tagger.is_known(s[i].word);
            bool hv_ok = pred[i] == s[i].tag;
            bool bl_ok = base.tag(s[i].word) == s[i].tag;
            hv_all.total++; hv_all.correct += hv_ok;
            bl_all.total++; bl_all.correct += bl_ok;
            Score& hk = known ? hv_known : hv_unk;
            Score& bk = known ? bl_known : bl_unk;
            hk.total++; hk.correct += hv_ok;
            bk.total++; bk.correct += bl_ok;
            if (!hv_ok) confusion[{s[i].tag, pred[i]}]++;
            ++n_tokens;
        }
    }

    std::printf("%-26s %10s %10s %10s\n", "token accuracy (%)", "overall", "known", "unknown");
    std::printf("%-26s %10.2f %10.2f %10.2f\n", "most-common-tag baseline", bl_all.pct(), bl_known.pct(), bl_unk.pct());
    std::printf("%-26s %10.2f %10.2f %10.2f\n", "HMM + Viterbi", hv_all.pct(), hv_known.pct(), hv_unk.pct());
    std::printf("(test tokens: %ld total, %ld known, %ld unknown = %.1f%% OOV)\n", hv_all.total, hv_known.total,
                hv_unk.total, 100.0 * hv_unk.total / std::max(1L, hv_all.total));
    std::printf("decoding speed: %.0f tokens/s\n\n", n_tokens / std::max(decode_s, 1e-9));

    std::vector<std::pair<int, std::pair<std::string, std::string>>> errs;
    for (auto& [k, v] : confusion) errs.push_back({v, k});
    std::sort(errs.rbegin(), errs.rend());
    std::printf("most frequent HMM errors (gold -> predicted):\n");
    for (size_t i = 0; i < errs.size() && i < 8; ++i)
        std::printf("  %-8s -> %-8s %6d\n", errs[i].second.first.c_str(), errs[i].second.second.c_str(), errs[i].first);

    // Show one test sentence with gold / baseline / HMM tags side by side.
    if (!test.empty()) {
        const Sentence& s = test[test.size() / 2];
        std::vector<std::string> words;
        for (auto& t : s) words.push_back(t.word);
        auto pred = hmm_tagger.tag(words);
        std::printf("\nexample test sentence:\n  %-14s %-8s %-8s %-8s\n", "word", "gold", "baseline", "HMM");
        for (size_t i = 0; i < s.size() && i < 20; ++i)
            std::printf("  %-14s %-8s %-8s %-8s%s\n", s[i].word.c_str(), s[i].tag.c_str(), base.tag(s[i].word).c_str(),
                        pred[i].c_str(), pred[i] != s[i].tag ? "  <-- HMM wrong" : "");
    }

    if (!custom.empty()) {
        std::istringstream ss(custom);
        std::vector<std::string> words;
        std::string w;
        while (ss >> w) words.push_back(w);
        auto pred = hmm_tagger.tag(words);
        std::printf("\ncustom sentence:\n");
        for (size_t i = 0; i < words.size(); ++i) std::printf("  %-14s %s\n", words[i].c_str(), pred[i].c_str());
    }
    return 0;
}
