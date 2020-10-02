#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <random>
#include <ctime>
#include <algorithm>
#include <iterator>
#include "reduce_api.hpp"




int main(int argc, char* argv[]){	
    unsigned int length = std::stoi(argv[1]);
    
    int init = 100;

    std::vector<unsigned int> array(length, 0);

    gen_array(array);
    
    unsigned int my_sum = chiu::reduce(
        array.begin(), 
        array.end(), 
        init, 
        [=] __host__ __device__ (unsigned a, unsigned b){ 
            return a+b; 
        }
    ); 
    
    assert(my_sum == verify(array,init, [](unsigned a, unsigned b){ return a+b;}));

    std::cout << "array of " << length << " elements passed.\n";
    
    return 0;
} 
