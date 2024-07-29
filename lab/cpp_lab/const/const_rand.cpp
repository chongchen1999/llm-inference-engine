#include <iostream>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()

int main() {
    // Seed the random number generator with the current time
    std::srand(static_cast<unsigned int>(std::time(0)));

    // Generate a random integer
    const int randomNumber = std::rand();

    // Output the random number
    std::cout << "Random Number: " << randomNumber << std::endl;

    return 0;
}
