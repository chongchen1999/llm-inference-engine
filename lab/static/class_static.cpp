#include <iostream>

class foo {
private:
    static const int y = 20;
public:
    static const int x = 10;
    static int w;

    void bar() {
        static int z = 10;
    }
};

class MyClass {
public:
    static int staticVar;
    MyClass() {
        std::cout << "Constructor" << std::endl;
    }
};

// Definition and initialization of the static member variable
int MyClass::staticVar = 10;

int main() {
    // Accessing the static member variable without instantiating the class
    MyClass *myClass;
    std::cout << reinterpret_cast<int *>(myClass) << std::endl;
    std::cout << MyClass::staticVar << std::endl;
    return 0;
}