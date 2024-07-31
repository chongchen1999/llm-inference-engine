#include <iostream>
#include <memory>
#include <vector>

class MyClass {
private:
    std::vector<int> shape;
public:
    MyClass() {
        std::cout << "MyClass constructor" << std::endl;
    }

    MyClass(const MyClass& other) {
        std::cout << "MyClass copy constructor" << std::endl;
    }

    MyClass(const std::vector<int> &shape) : shape(shape) {}

    ~MyClass() {
        std::cout << "MyClass destructor" << std::endl;
    }

    void doSomething() {
        std::cout << "Doing something" << std::endl;
    }
};

int main() {
    // Define a unique_ptr to a MyClass object with shape {1, 2, 3, 4}
    auto myObject = std::make_unique<MyClass>(std::vector<int>{1, 2, 3, 4});
    auto myObject2 = new MyClass({1, 2, 3, 4});

    std::unique_ptr<MyClass> obj3;
    obj3 = std::unique_ptr<MyClass>(new MyClass({1, 2, 3, 4}));
    
    // Use the object
    myObject->doSomething();

    delete myObject2;
    return 0;
}
