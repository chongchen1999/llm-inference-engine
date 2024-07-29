#include <iostream>

void print_int(void *address) {
    int value = *(int *)address;
    std::cout << "Integer: " << value << std::endl;
}

void print_float(void *address) {
    float value = *(float *)address;
    std::cout << "Float: " << value << std::endl;
}

void print_char(void *address) {
    char value = *(char *)address;
    std::cout << "Character: " << value << std::endl;
}

void print_string(void *address) {
    char *value = (char *)address;
    std::cout << "String: " << value << std::endl;
}

int main() {
    int example_int = 123456;
    float example_float = 123.456f;
    char example_char = 'A';
    char example_string[] = "Hello, World!";

    print_int(&example_int);
    print_float(&example_float);
    print_char(&example_char);
    print_string(example_string);

    return 0;
}
