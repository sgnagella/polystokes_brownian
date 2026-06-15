// Routine to compute the cell given the particle configuration 

#include "data.h"
#include "CellList.h"
#include "globals.h"
#include "arrays.h"
#include <iostream>
#include <cmath>

using namespace arrays;

void compute_cell_list(){
    std::cout << "Computing cell list" << std::endl;
    
    int i, j, k, ii, ii3, cellIndex, nCellsx, nCellsy, nCellsz, nCells; 
    float rc, cellSizex, cellSizey, cellSizez;

    std::vector<float> &box = dataStruct->box;
    // int N = dataStruct->N;
    rc = dataStruct->rcut;

    // Choose number of cells based on the cut-off radius
    // round up to ensure that the cut-off radius is less than the cell size
    nCellsx = std::ceil(box[0] / rc);
    nCellsy = std::ceil(box[1] / rc);
    nCellsz = std::ceil(box[2] / rc);

    // Compute the cell size
    cellSizex = box[0] / nCellsx;
    cellSizey = box[1] / nCellsy;
    cellSizez = box[2] / nCellsz;

    // Initialize the head of chain for each cell
    // std::vector<int> headofChain;
    nCells = nCellsx * nCellsy * nCellsz;
    dataStruct->headofList.resize(nCells, 0);
    std::cout << "Number of cells: " << nCells << std::endl;

    // Initialize the linked list for each particle
    // std::vector<int> linkedList;
    dataStruct->linkedList.resize(Np);
    std::cout << "Number of particles: " << Np << std::endl;

    // std::vector<float> &x = dataStruct->x;
    // std::vector<float> &y = dataStruct->y;
    // std::vector<float> &z = dataStruct->z;

    for( ii = 0; ii < Np; ii++ ){
        // Compute the cell index for each particle
        // Compute indices along each dimensions 
        ii3 = ndim * ii; 
        i = std::floor((x[ii3] + box[0]/2) / cellSizex);
        j = std::floor((x[ii3+1] + box[1]/2) / cellSizey);
        k = std::floor((x[ii3+2] + box[2]/2) / cellSizez);

        // Compute the linear cell index
        cellIndex = i + j * nCellsx + k * nCellsx * nCellsy;
        // std::cout << "Particle: " << ii << " Cell index: " << cellIndex << std::endl;

        dataStruct->linkedList[ii] = dataStruct->headofList[cellIndex];
        dataStruct->headofList[cellIndex] = ii;

        // Output the result
        // std::cout << "Particle: " << ii << " List: " << dataStruct->linkedList[ii] << std::endl;

        // // Update the head of chain
        // if (headofChain[cellIndex] == 0){
        //     headofChain[cellIndex] = ii;
        // }
        // else{
        //     // Find the last particle in the chain
        //     int jj = headofChain[cellIndex];
        //     while (jj != 0){
        //         jj = headofChain[jj];
        //     }
        //     headofChain[jj] = ii;
        // }
    }

    // Print out elements of linkedlist and head of chain
    for( ii = 0; ii < nCells; ii++){
        std::cout << "Cell index: " << ii << " Head of chain: " << dataStruct->headofList[ii] << std::endl;
    }

    // Print linked list
    for( ii = 0; ii < Np; ii++){
        std::cout << "Particle: " << ii << " List: " << dataStruct->linkedList[ii] << std::endl;
    }

    return;
}