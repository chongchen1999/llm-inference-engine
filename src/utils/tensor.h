#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iostream>
#include <cuda_fp16.h>
#include "string_utils.h"
#include "macro.h"

#include <type_traits>
#include <cstdint>  // for int8_t

enum class Device {
    CPU_PINNED,
    CPU,
    GPU
};

enum class DataType {
    FP32,
    FP16,
    INT8,
    INT32,
    BOOL,
    BYTES,
    UNSUPPORTED
};

template<typename T>
inline DataType getTensorType() {
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
        return DataType::FP32;
    } else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
        return DataType::FP16;
    } else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
        return DataType::INT8;
    } else if (std::is_same<T, int>::value || std::is_same<T, const int>::value) {
        return DataType::INT32;
    } else if (std::is_same<T, bool>::value || std::is_same<T, const bool>::value) {
        return DataType::BOOL;
    } else if (std::is_same<T, char>::value || std::is_same<T, const char>::value) {
        return DataType::BYTES;
    } else {
        return DataType::UNSUPPORTED;
    }
}

template<typename T>
class TensorWrapper;

class Tensor {
public:
    Device device;
    DataType dtype;
    std::vector<int> shape;

    Tensor() = default;

    Tensor(
        const Device &device,
        const DataType &dtype,
        const std::vector<int> &shape
    ) : device(device), dtype(dtype), shape(shape) {}

    virtual int size() const {
        if (shape.empty()) {
            return 0;
        }
        return std::accumulate(
            shape.begin(),
            shape.end(),
            1,
            std::multiplies<int>()
        );
    }

    template<typename T>
    TensorWrapper<T> *wrap() {
        return static_cast<TensorWrapper<T> *>(this);
    }

    std::string deviceString() const {
        static const std::unordered_map<Device, std::string> device_string {
            {Device::CPU, "CPU"},
            {Device::CPU_PINNED, "CPU_PINNED"},
            {Device::GPU, "GPU"}
        };
        return device_string.at(device);
    }

    virtual std::string toString() const {
        static const std::unordered_map<DataType, std::string> type_to_string {
            {DataType::INT8, "INT8"},
            {DataType::INT32, "INT32"},
            {DataType::FP16, "FP16"},
            {DataType::FP32, "FP32"}
        };
        return fmtstr(
            "Tensor[device = %s, type = %s, shape = %s]",
            deviceString().c_str(),
            type_to_string.at(dtype).c_str(),
            vec2str(shape).c_str()
        );
    }
};

template<typename T>
class TensorWrapper : public Tensor {
public:
    T *data;

    TensorWrapper(
        const Device &device,
        const DataType &dtype,
        const std::vector<int> &shape
    ) : Tensor(device, dtype, shape) {}

    TensorWrapper(
        const Device &device,
        const DataType &dtype,
        const std::vector<int> &shape,
        T * const &data
    ) : Tensor(device, dtype, shape), data(data) {
        DataType in_dtype = getTensorType<T>();
        LLM_CHECK_WITH_INFO(
            getTensorType<T>() == dtype,
            "Passed in data type should be same as dtype in params"
        );
    }

    virtual int size() const override {
        if (data == nullptr || shape.empty()) {
            return 0;
        }
        return std::accumulate(
            shape.begin(),
            shape.end(),
            1,
            std::multiplies<int>()
        );
    }

    inline T getVal(const int &id) const {
        LLM_CHECK(device == Device::CPU);
        return data[id];
    }

    inline T getVal() const {
        LLM_CHECK(device == Device::CPU);
        return getVal(0);
    }

    inline T *getPtr() const {
        return data;
    }

    inline T *getPtrByOffset(const int &offset) const {
        return data + offset;
    }

    virtual std::string toString() const override {
        static const std::unordered_map<DataType, std::string> type_to_string {
            {DataType::INT8, "INT8"},
            {DataType::FP16, "FP16"},
            {DataType::FP32, "FP32"}
        };
        return fmtstr(
            "Tensor[device = %s, type = %s, shape = %s, data = %p]",
            deviceString().c_str(),
            type_to_string.at(dtype).c_str(),
            vec2str(shape).c_str(),
            data
        );
    }
};

class TensorMap {
public:
    std::unordered_map<std::string, Tensor *> tensor_map;

    TensorMap() = default;

    TensorMap(std::initializer_list<std::pair<std::string, Tensor *>> tensor_map) {
        for (const auto &kv : tensor_map) {
            if (isValid(kv.second)) {
                insert(kv.first, kv.second);
            } else {
                LLM_CHECK_WITH_INFO(
                    isValid(kv.second),
                    fmtstr(
                        "%s is not a valid tensor, skipping insert into TensorMap",
                        kv.first.c_str()
                    )
                );
            }
        }
    }

    TensorMap(const std::unordered_map<std::string, Tensor*> &tensor_map) {
        for (const auto &kv : tensor_map) {
            if (isValid(kv.second)) {
                insert(kv.first, kv.second);
            }
        }
    }

    virtual ~TensorMap() {
        tensor_map.clear();
    }

    inline size_t size() const {
        return tensor_map.size();
    }

    inline bool isExist(const std::string &key) const {
        return tensor_map.find(key) != tensor_map.end();
    }

    inline bool isValid(const Tensor *tensor) const {
        return tensor->size() > 0;
    }

    inline void insert(const std::string &key, Tensor *value) {
        tensor_map[key] = value;
    }

    inline void insert(const std::pair<std::string, Tensor *> &kv) {
        tensor_map.insert(kv);
    }

    inline Tensor *at(const std::string &key) const {
        LLM_CHECK_WITH_INFO(
            isExist(key),
            fmtstr(
                "Cannot find a tensor of name %s in the tensor map (keys: %s)",
                key.c_str(),
                vec2str(keys()).c_str()
            )
        );
        return tensor_map.at(key);
    }

    inline Tensor *operator[](const std::string &key) const {
        LLM_CHECK_WITH_INFO(
            isExist(key),
            fmtstr(
                "Cannot find a tensor of name %s in the tensor map (keys: %s)",
                key.c_str(),
                vec2str(keys()).c_str()
            )
        );
        return tensor_map.at(key);
    }

    std::vector<std::string> keys() const {
        std::vector<std::string> key_names;
        for (const auto &kv : tensor_map) {
            key_names.push_back(kv.first);
        }
        return key_names;
    }

    std::string toString() const {
        std::stringstream ss;
        ss << "{";
        auto key_names = keys();
        const auto size = tensor_map.size();
        for (size_t i = 0; i < size; ++i) {
            ss << key_names[i] << ": " << at(key_names[i])->toString();
            if (i < size - 1) {
                ss << ", ";
            }
        }
        ss << "}";
        return ss.str();
    }
};
