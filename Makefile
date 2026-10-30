CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

all: test_viterbi benchmark pos_tagger

test_viterbi: src/test_viterbi.cpp src/viterbi.hpp src/brute_force.hpp src/models.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

benchmark: src/benchmark.cpp src/viterbi.hpp src/brute_force.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

pos_tagger: src/pos_tagger.cpp src/viterbi.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test: test_viterbi
	./test_viterbi

bench: benchmark
	mkdir -p results && ./benchmark && python3 scripts/plot_benchmark.py

pos-sample: pos_tagger
	python3 scripts/make_sample_corpus.py && ./pos_tagger data/sample_tagged.txt

pos: pos_tagger
	./pos_tagger data/brown_tagged.txt

clean:
	rm -f test_viterbi benchmark pos_tagger

.PHONY: all test bench pos-sample pos clean
