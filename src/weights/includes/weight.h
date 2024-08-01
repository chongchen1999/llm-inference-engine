#pragma once

#include <string>

class Weight {
public:
    virtual void loadWeightsFromFile(const std::string &weight_path) = 0;
};