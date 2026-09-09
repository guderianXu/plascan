#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace metashape_texture {

template <class T>
class Image {
public:
    Image() = default;
    Image(int width, int height, int channels = 1, T value = T{})
        : width_(width), height_(height), channels_(channels),
          pixels_(checked_size(width, height, channels), value) {}

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] int channels() const noexcept { return channels_; }
    [[nodiscard]] bool empty() const noexcept { return pixels_.empty(); }

    T &operator()(int x, int y, int channel = 0) {
        return pixels_.at(index(x, y, channel));
    }
    const T &operator()(int x, int y, int channel = 0) const {
        return pixels_.at(index(x, y, channel));
    }

    [[nodiscard]] const std::vector<T> &pixels() const noexcept { return pixels_; }
    [[nodiscard]] std::vector<T> &pixels() noexcept { return pixels_; }

private:
    static std::size_t checked_size(int width, int height, int channels) {
        if (width < 0 || height < 0 || channels <= 0) {
            throw std::invalid_argument("invalid image dimensions");
        }
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
               static_cast<std::size_t>(channels);
    }

    [[nodiscard]] std::size_t index(int x, int y, int channel) const {
        if (x < 0 || y < 0 || channel < 0 || x >= width_ || y >= height_ ||
            channel >= channels_) {
            throw std::out_of_range("image coordinate out of range");
        }
        return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) +
                static_cast<std::size_t>(x)) * static_cast<std::size_t>(channels_) +
               static_cast<std::size_t>(channel);
    }

    int width_ = 0;
    int height_ = 0;
    int channels_ = 1;
    std::vector<T> pixels_;
};

}  // namespace metashape_texture
