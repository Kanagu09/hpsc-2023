/**
 * @file print.hpp
 * @brief print tool
 * @author Masaki Otsubo
 * @date 2023-06-06
 */

class print {
  public:
    static void null() { std::cout << std::endl; }
    static void value(std::string name, double value) {
        std::cout << name << ": " << value << std::endl;
    }
    static void array(std::string name, std::vector<double> array) {
        std::cout << name << ": [";
        for(double value : array) {
            std::cout << value << ",";
        }
        std::cout << "]" << std::endl;
    }
    static void array2(std::string name,
                       std::vector<std::vector<double>> array2) {
        std::cout << name << ": [" << std::endl;
        for(std::vector<double> array1 : array2) {
            std::cout << "[";
            for(double value : array1) {
                std::cout << value << ",";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }
};
