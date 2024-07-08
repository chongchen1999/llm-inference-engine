#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iostream>
#include <cuda_fp16.h>
#include "src/utils/string_utils.h"
#include "src/utils/macro.h"

enum Device {
    CPU_PINNED,
    CPU,
    GPU
};

enum DataType {
    FP32,
    FP16,
    INT8,
    INT32,
    BOOL,
    BYTES,
    UNSUPPORTED
};

template<typename T>
DataType getTensorType() {
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
        return FP32;
    } else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
        return FP16;
    } else if (std::is_same<T, int>::value || std::is_same<T, const int>::value) {
        return INT32;
    } else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
        return INT8;
    } else if (std::is_same<T, bool>::value || std::is_same<T, const bool>::value) {
        return BOOL;
    } else if (std::is_same<T, char>::value || std::is_same<T, const char>::value) {
        return BYTES;
    } else {
        return UNSUPPORTED;
    }
}

template<typename T>
class TensorWrapper;

class Tensor {
    Device location;
    DataType dtype;
    std::vector<int> shape;

    Tensor() = default;

    Tensor(Device location_, DataType dtype_, const std::vector<int>& shape_)
        : location(location_), dtype(dtype_), shape(shape_) {}

    virtual int size() const {
        if (shape.empty()) {
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }

    template<typename T>
    TensorWrapper<T>* as() {
        return static_cast<TensorWrapper<T>*>(this);
    }

    std::string DeviceString() const {
        static const std::unordered_map<Device, std::string> device_string{
            {CPU, "CPU"}, {CPU_PINNED, "CPU_PINNED"}, {GPU, "GPU"}
        };
        return device_string.at(location);
    }

    virtual std::string toString() const {
        static const std::unordered_map<DataType, std::string> type_to_string{
            {INT8, "INT8"}, {INT32, "INT32"}, {FP16, "FP16"}, {FP32, "FP32"}
        };
        return fmtstr("Tensor[where=%s, type=%s, shape=%s]",
            DeviceString().c_str(), type_to_string.at(dtype).c_str(), vec2str(shape).c_str());
    }  
};

template<typename T>
class TensorWrapper : public Tensor {
public:
    T* data;

    TensorWrapper(Device location, DataType dtype, const std::vector<int>& shape)
        : Tensor(location, dtype, shape) {}

    TensorWrapper(Device location, DataType dtype, const std::vector<int>& shape, T* data)
        : Tensor(location, dtype, shape), data(data) {
        DataType in_dtype = getTensorType<T>();
        LLM_CHECK_WITH_INFO(in_dtype == dtype, "Passed in data type should be same as dtype in params");
    }

    virtual int size() const override {
        if (data == nullptr || shape.empty()) {
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }

    inline T getVal(int id) const {
        LLM_CHECK(location == CPU);
        return data[id];
    }

    inline T getVal() const {
        LLM_CHECK(location == CPU);
        return getVal(0);
    }

    inline T* getPtr() const {
        return data;
    }

    inline T* getPtrByOffset(int offset) const {
        return data + offset;
    }

    virtual std::string toString() const override {
        static const std::unordered_map<DataType, std::string> type_to_string{
            {INT8, "INT8"}, {FP16, "FP16"}, {FP32, "FP32"}
        };
        return fmtstr("Tensor[where=%s, type=%s, shape=%s, data=%p]",
            DeviceString().c_str(), type_to_string.at(dtype).c_str(), vec2str(shape).c_str(), data);
    }
};

class TensorMap {
    std::unordered_map<std::string, Tensor*> tensor_map_;

    TensorMap() = default;

    TensorMap(std::initializer_list<std::pair<std::string, Tensor*>> tensor_map) {
        for (const auto& pair : tensor_map) {
            if (isValid(pair.second)) {
                insert(pair.first, pair.second);
            } else {
                LLM_CHECK_WITH_INFO(isValid(pair.second), fmtstr("%s is not a valid tensor, skipping insert into TensorMap", pair.first.c_str()));
            }
        }
    }

    TensorMap(const std::unordered_map<std::string, Tensor*>& tensor_map) {
        for (const auto& kv : tensor_map) {
            if (isValid(kv.second)) {
                insert(kv.first, kv.second);
            }
        }
    }

    virtual ~TensorMap() {
        tensor_map_.clear();
    }

    inline size_t size() const {
        return tensor_map_.size();
    }

    inline bool isExist(const std::string& key) const {
        return tensor_map_.find(key) != tensor_map_.end();
    }

    inline bool isValid(const Tensor* tensor) const {
        return tensor->size() > 0;
    }

    inline void insert(const std::string& key, Tensor* value) {
        tensor_map_[key] = value;
    }

    inline void insert(std::pair<std::string, Tensor*> p) {
        tensor_map_.insert(p);
    }

    inline Tensor* at(const std::string& key) {
        LLM_CHECK_WITH_INFO(isExist(key), fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                              key.c_str(), vec2str(keys()).c_str()));
        return tensor_map_.at(key);
    }

    inline Tensor* operator[] (const std::string& key) {
        LLM_CHECK_WITH_INFO(isExist(key), fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                              key.c_str(), vec2str(keys()).c_str()));
        return tensor_map_.at(key);
    }

    std::vector<std::string> keys() const {
        std::vector<std::string> key_names;
        for (const auto& kv : tensor_map_) {
            key_names.push_back(kv.first);
        }
        return key_names;
    }

    std::string toString() const {
        std::stringstream ss;
        ss << "{";
        std::vector<std::string> key_names = keys();
        for (size_t i = 0; i < tensor_map_.size(); ++i) {
            ss << key_names[i] << ": " << at(key_names[i])->toString();
            if (i < tensor_map_.size() - 1) {
                ss << ", ";
            }
        }
        ss << "}";
        return ss.str();
    }
};