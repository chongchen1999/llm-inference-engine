#include <iostream>
#include <memory>

int main() {
    std::unique_ptr<int[]> arrayPtr = std::make_unique<int[]>(10);
    arrayPtr[0] = 5;

    // Using get() to access the raw pointer
    auto rawPtr = arrayPtr.get(1);
    std::cout << typeid(rawPtr).name() << std::endl;

    // Using &arrayPtr[0] to access the raw pointer
    auto firstElementPtr = &arrayPtr[1];
    std::cout << typeid(firstElementPtr).name() << std::endl;

    std::cout << (rawPtr == firstElementPtr) << std::endl;
    return 0;
}
