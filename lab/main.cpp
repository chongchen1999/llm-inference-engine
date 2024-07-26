#include <iostream>
#include <typeinfo>

struct A {
    int a;
};
int main() {
    auto address = &A::a;
    std::cout << "Address: " << address << std::endl;
    std::cout << "Datatype: " << typeid(decltype(address)).name() << std::endl;
    return 0;
}