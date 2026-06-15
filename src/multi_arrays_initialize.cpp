#include <iostream>
#include "multi_arrays.h"
// #include "globals.h"

void initialize_rank2_array(rank2_array tensor){
    // Initializes a rank 2 array with zeros 
    // Input is a predefined tensor of type rank_2_array 
    // Changes the tensor to be initialized with zeros 
    std::fill( tensor.data() , tensor.data() + tensor.num_elements() , 0.0);
}

void initialize_rank3_array(rank3_array tensor){
    // Initializes a rank 2 array with zeros 
    // Input is a predefined tensor of type rank_3_array 
    // Changes the tensor to be initialized with zeros 
    std::fill(tensor.data() , tensor.data() + tensor.num_elements() , 0.0);
}

void initialize_rank4_array(rank4_array tensor){
    // Initializes a rank 2 array with zeros 
    // Input is a predefined tensor of type rank_4_array 
    // Changes the tensor to be initialized with zeros 
    std::fill(tensor.data() , tensor.data() + tensor.num_elements() , 0.0);
}

