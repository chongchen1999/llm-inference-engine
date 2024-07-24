#include <iostream>

int main() {
    const char *code = R"(#include <iostream>

int main() {
    const char *code = R"(%s)";
    std::cout << code << std::endl;
    printf(code, code);
    return 0;
}
)";
    std::cout << code << std::endl;
    printf(code, code);
    return 0;
}
