#pragma once
#include <memory>   // std::make_unique
#include <sstream>  // std::stringstream
#include <string>
#include <vector>

// This function allows us to define custom print strings
template<typename... Args>
inline std::string fmtstr(const std::string &format, Args... args) {
    // This function is adapted from a code snippet on StackOverflow under cc-by-1.0
    // https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf

    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1);  // We don't want the '\0' inside
}

// Converts elements in a vector to a string
template<typename T>
inline std::string vec2str(const std::vector<T> &vec) {
    std::stringstream ss;
    ss << "(";
    if (!vec.empty()) {
        for (auto &vec : vec) {
            ss << vec << ", ";
        }
        ss << vec.back();
    }
    ss << ")";
    return ss.str();
}

// Converts elements in an array to a string
template<typename T>
inline std::string arr2str(const T * const &arr, size_t size) {
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < size - 1; ++i) {
        ss << arr[i] << ", ";
    }
    if (size > 0) {
        ss << arr[size - 1];
    }
    ss << ")";
    return ss.str();
}