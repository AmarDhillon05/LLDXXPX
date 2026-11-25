// OpenCV implementations of every operation, returning this project's image
// types, used as the external ground truth. Header-only; requires HAVE_OPENCV.
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>

#include "image.hpp"

namespace cvref {

inline cv::Mat wrap(const ImageF32& img) {
    return cv::Mat(img.height, img.width, CV_32F, const_cast<float*>(img.data.data()));
}
inline cv::Mat wrap(const ImageU8& img) {
    return cv::Mat(img.height, img.width, CV_8U, const_cast<uint8_t*>(img.data.data()));
}
inline cv::Mat wrap(const ImageRGBA8& img) {
    return cv::Mat(img.height, img.width, CV_8UC4, const_cast<uint8_t*>(img.data.data()));
}
inline ImageF32 to_f32(const cv::Mat& m) {
    ImageF32 out(m.cols, m.rows);
    m.copyTo(cv::Mat(m.rows, m.cols, CV_32F, out.data.data()));
    return out;
}
inline ImageU8 to_u8(const cv::Mat& m) {
    ImageU8 out(m.cols, m.rows);
    m.copyTo(cv::Mat(m.rows, m.cols, CV_8U, out.data.data()));
    return out;
}

inline ImageF32 gray(const ImageRGBA8& in) {
    cv::Mat f, g;
    wrap(in).convertTo(f, CV_32FC4);
    cv::cvtColor(f, g, cv::COLOR_RGBA2GRAY);  // float path: 0.299 R + 0.587 G + 0.114 B
    return to_f32(g);
}

inline ImageF32 gaussian(const ImageF32& in, int radius, float sigma) {
    cv::Mat out;
    const int k = 2 * radius + 1;
    cv::GaussianBlur(wrap(in), out, cv::Size(k, k), sigma, sigma, cv::BORDER_REPLICATE);
    return to_f32(out);
}

inline ImageF32 sobel_mag(const ImageF32& in) {
    cv::Mat gx, gy, mag;
    cv::Sobel(wrap(in), gx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(wrap(in), gy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::magnitude(gx, gy, mag);
    return to_f32(mag);
}

inline ImageU8 threshold(const ImageF32& mag, float thr) {
    cv::Mat out, out8;
    cv::threshold(wrap(mag), out, thr, 255.0, cv::THRESH_BINARY);  // > thr -> 255
    out.convertTo(out8, CV_8U);
    return to_u8(out8);
}

inline ImageF32 bilateral(const ImageF32& in, int radius, float sigma_color, float sigma_space) {
    cv::Mat out;
    cv::bilateralFilter(wrap(in), out, 2 * radius + 1, sigma_color, sigma_space, cv::BORDER_REPLICATE);
    return to_f32(out);
}

inline std::array<uint32_t, 256> histogram(const ImageU8& in) {
    cv::Mat src = wrap(in), hist;
    const int channels[] = {0};
    const int size[] = {256};
    const float range[] = {0.f, 256.f};
    const float* ranges[] = {range};
    cv::calcHist(&src, 1, channels, cv::Mat(), hist, 1, size, ranges);
    std::array<uint32_t, 256> h{};
    for (int i = 0; i < 256; ++i) h[i] = uint32_t(hist.at<float>(i));
    return h;
}

inline ImageU8 equalize(const ImageU8& in) {
    cv::Mat out;
    cv::equalizeHist(wrap(in), out);
    return to_u8(out);
}

inline ImageU8 edge_pipeline(const ImageRGBA8& in, int radius, float sigma, float thr) {
    return threshold(sobel_mag(gaussian(gray(in), radius, sigma)), thr);
}

}  // namespace cvref
