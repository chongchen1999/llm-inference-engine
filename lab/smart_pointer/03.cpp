#include <memory>
#include <iostream>

int main() {
    /*int *rawPtr = new int(10); // Raw pointer to an integer
    std::cout << rawPtr << std::endl;
    std::cout << (rawPtr == nullptr) << std::endl;

    std::unique_ptr<int> smartPtr = std::unique_ptr<int>(rawPtr); // Transfer ownership to smart pointer
    std::cout << smartPtr.get() << std::endl;
    std::cout << rawPtr << std::endl;
    std::cout << (rawPtr == nullptr) << std::endl;

    std::cout << std::endl;

    auto smart_ptr = std::make_unique<int>(10); // Create a smart pointer
    std::cout << smart_ptr.get() << std::endl;
    auto smart_ptr2 = std::move(smartPtr); // Transfer ownership to another smart pointer

    std::cout << (smartPtr.get() == nullptr) << std::endl;
    std::cout << smart_ptr2.get() << std::endl;
    std::cout << std::endl;*/

    auto smartPtr = std::make_unique<int>(10); // Create a smart pointer
    std::cout << smartPtr.get() << std::endl;
    std::cout << &*smartPtr << std::endl;
    return 0;
}
