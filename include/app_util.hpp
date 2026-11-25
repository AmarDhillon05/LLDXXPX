// Tiny helpers shared by the apps: argument parsing, default parameters,
// device info, timing statistics.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// Parameters used everywhere unless overridden on the command line.
struct Params {
    int blur_radius = 7;        // 15x15 Gaussian
    float blur_sigma = 2.5f;
    float sobel_threshold = 40.f;
    int bil_radius = 5;         // 11x11 circular window (81 taps)
    float bil_sigma_color = 25.f;
    float bil_sigma_space = 3.f;
    float canny_low = 20.f, canny_high = 60.f;
};

// --key value / --flag parser.
class Args {
public:
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) != 0) continue;
            a = a.substr(2);
            const auto eq = a.find('=');
            if (eq != std::string::npos) {
                kv_[a.substr(0, eq)] = a.substr(eq + 1);
            } else if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                kv_[a] = argv[++i];
            } else {
                kv_[a] = "1";
            }
        }
    }
    bool has(const std::string& k) const { return kv_.count(k) != 0; }
    std::string str(const std::string& k, const std::string& def) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? def : it->second;
    }
    int i(const std::string& k, int def) const { return has(k) ? std::atoi(kv_.at(k).c_str()) : def; }
    float f(const std::string& k, float def) const { return has(k) ? float(std::atof(kv_.at(k).c_str())) : def; }

    Params params() const {
        Params p;
        p.blur_radius = i("radius", p.blur_radius);
        p.blur_sigma = f("sigma", p.blur_sigma);
        p.sobel_threshold = f("threshold", p.sobel_threshold);
        p.bil_radius = i("bil-radius", p.bil_radius);
        p.bil_sigma_color = f("bil-sigma-color", p.bil_sigma_color);
        p.bil_sigma_space = f("bil-sigma-space", p.bil_sigma_space);
        return p;
    }

private:
    std::map<std::string, std::string> kv_;
};

struct Resolution {
    std::string name;
    int w, h;
};

// "1080p,4k,8k" or "1920x1080,..." -> list
inline std::vector<Resolution> parse_sizes(const std::string& s) {
    std::vector<Resolution> out;
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t end = s.find(',', pos);
        if (end == std::string::npos) end = s.size();
        std::string t = s.substr(pos, end - pos);
        if (t == "1080p") out.push_back({"1080p", 1920, 1080});
        else if (t == "4k") out.push_back({"4k", 3840, 2160});
        else if (t == "8k") out.push_back({"8k", 7680, 4320});
        else if (!t.empty()) {
            int w = 0, h = 0;
            if (std::sscanf(t.c_str(), "%dx%d", &w, &h) == 2) out.push_back({t, w, h});
            else std::fprintf(stderr, "ignoring size '%s'\n", t.c_str());
        }
        pos = end + 1;
    }
    return out;
}

// Wall-clock timing for CPU baselines.
template <typename F>
double time_cpu_ms(F&& fn, int reps) {
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        auto a = std::chrono::steady_clock::now();
        fn();
        auto b = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];  // median
}
