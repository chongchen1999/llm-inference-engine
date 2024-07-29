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

int MyClass::count = 0;

MyClass get(const int &id) {
    MyClass a = MyClass(id);
    // std::cout << MyClass::count << std::endl;
    std::cout<< &a<< std::endl;
    return a;
}

std::string get_str() {
    std::string s = "Hello";
    std::cout<< &s << std::endl;
    return s;
}

int main() {
    MyClass a = get(10);
    std::cout << MyClass::count << std::endl;
    std::cout<< &a << std::endl;

    MyClass b = a;
    std::cout << MyClass::count << std::endl;
    std::cout<< &b << std::endl;

    MyClass c;
    c = a;
    std::cout << MyClass::count << std::endl;
    std::cout<< &b << std::endl;
    return 0;
}