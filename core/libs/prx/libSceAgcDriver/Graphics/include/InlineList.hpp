#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_INLINELIST_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_INLINELIST_HPP

#include <array>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

// A list whose first N elements live inline: the per-draw lists hold a handful of entries, and a
// heap allocation each cost more than filling them. Past N the elements move to a vector.
template <typename T, std::size_t N>
class InlineList {
public:
    InlineList() = default;
    InlineList(const InlineList&) = default;
    InlineList& operator=(const InlineList&) = default;
    // A moved-from list is empty, as a moved-from vector is.
    InlineList(InlineList&& other) noexcept : local(std::move(other.local)), heap(std::move(other.heap)), count(std::exchange(other.count, 0)) {}
    InlineList& operator=(InlineList&& other) noexcept {
        local = std::move(other.local);
        heap = std::move(other.heap);
        count = std::exchange(other.count, 0);
        return *this;
    }
    void push_back(const T& value) { add(value); }
    void push_back(T&& value) { add(std::move(value)); }
    void assign(std::size_t size, const T& value) {
        clear();
        for (std::size_t i = 0; i < size; ++i) push_back(value);
    }
    void clear() {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (std::size_t i = 0; i < count && i < N; ++i) local[i] = T{};
        }
        heap.clear();
        count = 0;
    }
    T* data() { return count <= N ? local.data() : heap.data(); }
    const T* data() const { return count <= N ? local.data() : heap.data(); }
    std::size_t size() const { return count; }
    bool empty() const { return count == 0; }
    T& operator[](std::size_t index) { return data()[index]; }
    const T& operator[](std::size_t index) const { return data()[index]; }
    T& front() { return data()[0]; }
    const T& front() const { return data()[0]; }
    T& back() { return data()[count - 1]; }
    T* begin() { return data(); }
    T* end() { return data() + count; }
    const T* begin() const { return data(); }
    const T* end() const { return data() + count; }

private:
    template <typename U>
    void add(U&& value) {
        if (count < N) {
            local[count] = std::forward<U>(value);
        } else {
            if (heap.empty()) {
                // Room for as many again: the spill is one allocation, not one for the inline
                // elements and another as the first push grows it.
                heap.reserve(2 * N);
                heap.assign(std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
            }
            heap.push_back(std::forward<U>(value));
        }
        ++count;
    }

    // Default-initialized: nothing past `count` is read, so trivial elements are left unwritten (a
    // draw builds kilobytes of these lists) while a class type such as shared_ptr starts empty.
    std::array<T, N> local;
    std::vector<T> heap;
    std::size_t count = 0;
};

}

#endif
