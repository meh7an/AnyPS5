#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HASHSLOTMAP_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HASHSLOTMAP_HPP

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace AgcDriver {

// A map from 64-bit hashes to values, by open addressing in a power-of-two table kept at most half
// full: a lookup masks the key (a hash already) and scans a compact key array, where
// std::unordered_map divided by its prime bucket count and followed the bucket's pointer to the node
// (the stage memo's lookup was about 2% of the worker on S3K). Key 0 marks a free slot, so a key of 0
// is kept as 1: callers verify what a lookup finds. Erasing shifts the entries after it back (no
// tombstones), so a pointer into the map holds only until the next insert or erase.
template<typename TValue>
class HashSlotMap {
public:
    explicit HashSlotMap(std::size_t capacity = 64) { reset(capacity); }

    std::size_t size() const { return count; }

    TValue* find(std::uint64_t key) {
        key = stored(key);
        for (auto index = key & mask; keys[index] != 0; index = (index + 1) & mask) {
            if (keys[index] == key) return &values[index];
        }
        return nullptr;
    }

    // The value at `key`, default-constructed when new, and whether it was.
    std::pair<TValue*, bool> try_emplace(std::uint64_t key) {
        if (auto* found = find(key)) return {found, false};
        if ((count + 1) * 2 > keys.size()) grow();
        key = stored(key);
        auto index = key & mask;
        while (keys[index] != 0) index = (index + 1) & mask;
        keys[index] = key;
        values[index] = TValue{};
        ++count;
        return {&values[index], true};
    }

    bool erase(std::uint64_t key) {
        key = stored(key);
        auto hole = key & mask;
        while (keys[hole] != key) {
            if (keys[hole] == 0) return false;
            hole = (hole + 1) & mask;
        }
        // An entry after the hole stays where it is when its home slot lies cyclically in
        // (hole, entry]; any other moves into the hole, which moves to where it was.
        for (auto next = (hole + 1) & mask; keys[next] != 0; next = (next + 1) & mask) {
            const auto home = keys[next] & mask;
            const bool stays = hole <= next ? hole < home && home <= next : hole < home || home <= next;
            if (stays) continue;
            keys[hole] = keys[next];
            values[hole] = std::move(values[next]);
            hole = next;
        }
        keys[hole] = 0;
        values[hole] = TValue{};
        --count;
        return true;
    }

private:
    static std::uint64_t stored(std::uint64_t key) { return key != 0 ? key : 1; }

    void reset(std::size_t capacity) {
        std::size_t slots = 16;
        while (slots < capacity) slots *= 2;
        keys.assign(slots, 0);
        values.clear();
        values.resize(slots);
        mask = slots - 1;
        count = 0;
    }

    void grow() {
        auto oldKeys = std::move(keys);
        auto oldValues = std::move(values);
        reset(oldKeys.size() * 2);
        for (std::size_t old = 0; old < oldKeys.size(); ++old) {
            if (oldKeys[old] == 0) continue;
            auto index = oldKeys[old] & mask;
            while (keys[index] != 0) index = (index + 1) & mask;
            keys[index] = oldKeys[old];
            values[index] = std::move(oldValues[old]);
            ++count;
        }
    }

    std::vector<std::uint64_t> keys;
    std::vector<TValue> values;
    std::size_t mask = 0;
    std::size_t count = 0;
};

}

#endif
