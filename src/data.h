// Declare a struct to neighborlist data
#ifndef DATA_H
#define DATA_H
#include <vector>
struct Data {
    std::vector<float> box;     // Box vector {Lx, Ly, Lz}
    std::vector<float> rcuts;   // Interaction cutoff distances for force calcs
    std::vector<float> rverls;   // Verlet cutoff distances
    std::vector<float> sigmas; 
    std::vector<int> nlist;
    std::vector<int> headofList; // Head of chain for each cell
    std::vector<int> linkedList; // Linked list for each particle
};

#endif 