#include <iostream>

template <typename T>
T add(T a, T b);

template<>
int add(int a, int b) {
    std::cout << "int" << std::endl;
    return a + b;
}

template<>
double add(double a, double b) {
    std::cout << "double" << std::endl; 
    return a + b;
}
int main() {
    add<int>(1, 2);
    add<double>(1, 2);
    return 0;
}