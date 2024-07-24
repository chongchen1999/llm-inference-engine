#include <map>
#include <string>
#include <iostream>

int main() {
    // Define a map where keys are strings and values are integers
    std::map<std::string, int> myMap;

    // Example usage
    myMap["apple"] = 1;
    myMap["banana"] = 2;
    myMap["cherry"] = 3;

    auto *ptr = &myMap;

    // Accessing elements
    int appleValue = myMap["apple"];  // appleValue will be 1

    // Iterating over the map
    for (const auto& pair : myMap) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    return 0;
}
