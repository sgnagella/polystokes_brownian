// Header file to instantiate necessary multidimensional arrays for this calculation

#include "boost/multi_array.hpp"

// Define type for a 3x3 array
typedef boost::multi_array<long double, 2> rank2_array; 

// Define type for a 3x3x3 array
typedef boost::multi_array<long double, 3> rank3_array;

// Define type a 3x3x3x3 array 
typedef boost::multi_array<long double, 4> rank4_array;

// Declare method for initializing any of these multiarrays
void initialize_rank2_array(rank2_array tensor); 
void initialize_rank3_array(rank3_array tensor); 
void initialize_rank4_array(rank4_array tensor);
