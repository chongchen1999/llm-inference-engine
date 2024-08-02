#include <iostream>
#include <memory>

class Base {
public:
    virtual ~Base() {
        std::cout << "Base class destructor" << std::endl;
    }
    virtual void display() const {
        std::cout << "Base class display function" << std::endl;
    }
};

class Derived : public Base {
public:
    void display() const override {
        std::cout << "Derived class display function" << std::endl;
    }

    ~Derived() override {
        std::cout << "Derived class destructor" << std::endl;
    }
};

void function(std::unique_ptr<Base> ptr) {
    if (ptr) {
        ptr->display();
    }
}

int main() {
    std::unique_ptr<Derived> ptr = std::make_unique<Derived>();
    function(std::move(ptr));

    std::cout<< "moved!" << std::endl;

    std::cout<< (ptr == nullptr) << std::endl;
    if (ptr) {
        ptr->display();
    }
    return 0;
}
