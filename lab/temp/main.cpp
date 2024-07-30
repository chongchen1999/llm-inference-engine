#include <iostream>
#include <string>

class MyClass {
public:
    static int count;
    int id;

    MyClass() = default;

    MyClass(const int &id) : id(id) {
        ++count;
        std::cout << "Constructor" << std::endl;
    }

    MyClass(const MyClass &other) : id(other.id) {
        ++count;
        std::cout << "Copy constructor" << std::endl;
    }

    MyClass &operator=(const MyClass &other) {
        id = other.id;
        std::cout << "Assignment operator" << std::endl;
        return *this;
    }
};

int main() {
    auto *a = &MyClass(1);

    return 0;
}