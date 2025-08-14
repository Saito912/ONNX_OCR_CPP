// ctc_decode_optimized.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

std::string encode_sequence(const std::vector<int>& seq) {
    std::string s;
    s.reserve(seq.size());
    for (int token : seq) {
        s += static_cast<char>(token);
    }
    return s;
}

std::vector<std::string> ctc_beam_search_decode_optimized(
    const std::vector<float>& output_tensor,
    const std::vector<int64_t>& shape,
    int beam_size = 10)
{
    const std::string charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ  ";
    const int blank_id = 0;
    const int num_classes = static_cast<int>(charset.size()) + 1; // 63

    if (shape.size() != 3) {
        throw std::invalid_argument("Shape must be [B, T, C]");
    }

    int64_t B = shape[0];
    int64_t T = shape[1];
    int64_t C = shape[2];


    std::vector<std::string> results(B);

    #pragma omp parallel for
    for (int64_t b = 0; b < B; ++b) {
        using Hypothesis = std::pair<std::string, double>;
        std::vector<Hypothesis> beam;
        beam.reserve(beam_size);
        beam.push_back({"", 0.0});

        std::vector<double> log_probs(C);

        for (int64_t t = 0; t < T; ++t) {
            int64_t offset = b * T * C + t * C;
            const float* t_probs_ptr = output_tensor.data() + offset;

            // Log Softmax
            double max_logit = -INFINITY;
            for (int c = 0; c < C; ++c) if (t_probs_ptr[c] > max_logit) max_logit = t_probs_ptr[c];

            double log_sum = 0.0;
            for (int c = 0; c < C; ++c) {
                log_sum += std::exp(t_probs_ptr[c] - max_logit);
            }
            log_sum = std::log(log_sum) + max_logit;
            for (int c = 0; c < C; ++c) {
                log_probs[c] = t_probs_ptr[c] - log_sum;
            }
            std::map<std::string, double> candidates;

            for (const auto& hyp : beam) {
                const auto& prefix_str = hyp.first;
                double prev_log_prob = hyp.second;

                for (int c = 0; c < C; ++c) {
                    double new_log_prob = prev_log_prob + log_probs[c];

                    if (c == blank_id) {
                        auto it = candidates.find(prefix_str);
                        if (it == candidates.end()) {
                            candidates[prefix_str] = new_log_prob;
                        } else {
                            double max_p = std::max(it->second, new_log_prob);
                            it->second = max_p + std::log1p(std::exp(std::min(it->second, new_log_prob) - max_p));
                        }
                    } else {
                        // 避免在循环内创建 vector，直接操作 string
                        std::string new_prefix_str = prefix_str;
                        if (prefix_str.empty() || static_cast<int>(prefix_str.back()) != c) {
                           new_prefix_str += static_cast<char>(c);
                        }

                        auto it = candidates.find(new_prefix_str);
                        if (it == candidates.end()) {
                            candidates[new_prefix_str] = new_log_prob;
                        } else {
                            double max_p = std::max(it->second, new_log_prob);
                            it->second = max_p + std::log1p(std::exp(std::min(it->second, new_log_prob) - max_p));
                        }
                    }
                }
            }

            std::vector<Hypothesis> new_beam;
            new_beam.reserve(candidates.size());
            for(const auto& pair : candidates) {
                new_beam.push_back({pair.first, pair.second});
            }

            std::sort(new_beam.begin(), new_beam.end(),
                      [](const Hypothesis& a, const Hypothesis& b) { return a.second > b.second; });

            if (new_beam.size() > beam_size) {
                new_beam.resize(beam_size);
            }
            beam = std::move(new_beam);
        }

        // 解码最终结果
        if (beam.empty() || beam[0].first.empty()) {
            results[b] = "";
        } else {
            std::string decoded;
            const std::string& best_prefix_str = beam[0].first;
            decoded.reserve(best_prefix_str.length());
            for (char token_char : best_prefix_str) {
                int label_id = static_cast<int>(token_char);
                int char_idx = label_id - 1;
                if (char_idx >= 0 && char_idx < static_cast<int>(charset.size())) {
                    decoded += charset[char_idx];
                }
            }
            results[b] = decoded;
        }
    }

    return results;
}

// 绑定模块
namespace py = pybind11;

PYBIND11_MODULE(ctc_decoder, m) {
    m.doc() = "Optimized C++ CTC Beam Search Decoder for Python";
    m.def("ctc_beam_search", &ctc_beam_search_decode_optimized,
          py::arg("output_tensor"),
          py::arg("shape"),
          py::arg("beam_size") = 10,
          "Performs a fast, parallelized CTC beam search decoding.\n"
          "Optimizations:\n"
          "- Uses std::string as map key for efficient path merging.\n"
          "- Parallelized over the batch dimension using OpenMP."
          );
}