#include <iostream>
#include <memory>

class MyClass {
public:
    void display() {
        std::cout << "MyClass instance" << std::endl;
    }
};

int main() {
    std::unique_ptr<MyClass> ptr1 = std::make_unique<MyClass>();
    
    // Use the pointer
    if (ptr1) {
        ptr1->display();
    }

    // Move the unique pointer to create a nullptr in ptr1
    std::unique_ptr<MyClass> ptr2 = std::move(ptr1);

    // Check if ptr1 is now nullptr
    if (!ptr1) {
        std::cout << "ptr1 is now nullptr" << std::endl;
        std::cout << &ptr1 << std::endl;
    }

    // Use the moved pointer
    if (ptr2) {
        ptr2->display();
    }

    int *a;

    return 0;
}
