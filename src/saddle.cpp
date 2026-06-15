#include "saddle.h"
#include "arrays.h"
#include "globals.h"
#include <iostream>

using namespace arrays;

void saddle(){
    // This routine constructs the saddle point matrix
    // from the grand mobility matrix and the projection operator

    // Generate saddle point matrix
    PetscErrorCode ierr;
    PetscInt i, ishift;
    // Mat A; 
    // MatCreate(PETSC_COMM_WORLD, &A);
    // MatSetSizes(A, n17p, n17p, PETSC_DECIDE, PETSC_DECIDE);
    // MatSetType(A, MATDENSE); 
    // MatSetUp(A);
    // MatZeroEntries(A);
    std::cout << "Creating saddle point..." << std::endl;
    PetscScalar array4[nm3nc11];
    PetscScalar array4T[nm3nc11*nm3nc11];
    PetscScalar arrays5T[nm3nc11*nm3nc6];
    PetscInt rowsUF[nm3nc11], colsUF[nm3nc6], colsUFshift[nm3nc6];
    std::iota(rowsUF, rowsUF + nm3nc11, 0);
    std::iota(colsUF, colsUF + nm3nc6, 0);
    std::iota(colsUFshift, colsUFshift + nm3nc6, nm3nc11);
    // for(i = 0; i < n11p; i++){
    //     rowsUF[i] = i;  
    // }

    // for(i = 0; i < n6p; i++){
    //     colsUF[i] = i;
    //     colsUFshift[i] = i + n11p;
    // }

    // Insert M into A
    ierr = MatGetValues(M, nm3nc11, rowsUF, nm3nc11, rowsUF, array4T); CHKERRV(ierr);
    ierr = MatSetValues(A, nm3nc11, rowsUF, nm3nc11, rowsUF, array4T, INSERT_VALUES); CHKERRV(ierr);

    // // Insert B into A
    // MatGetValues(B, n11p, rowsUF, n6p, colsUF, arrays5T);
    // MatSetValues(A, n11p, rowsUF, n6p, colsUFshift, arrays5T, INSERT_VALUES);

    // // Insert B^T into A
    // for( i = 0; i < n6p; i++){
    //     ishift = i + n11p;
    //     MatGetValues(B, n11p, rowsUF, 1, &i, array4);
    //     MatSetValues(A, 1, &ishift, n11p, rowsUF, array4, INSERT_VALUES);
    // }

    // TODO: add lubrication to lower right block
    // Assemble the saddle point matrix
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRV(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRV(ierr);

    std::cout << "Saddle point matrix created." << std::endl;
    // Is matrix symmetric
    PetscBool isSymmetric;
    PetscReal tol = 1e-10;
    MatIsSymmetric(A, tol, &isSymmetric);
    if (isSymmetric == PETSC_TRUE) {
        PetscPrintf(PETSC_COMM_WORLD, "The matrix A is symmetric.\n\n");
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "The matrix A is not symmetric.\n\n");
    }

    // Save matrix to file
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "matrix.csv", &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_DENSE); // Forces dense output
    MatView(A, viewer);
    PetscViewerPopFormat(viewer);
    PetscViewerDestroy(&viewer);


    
    return;
}

PetscErrorCode PreconditionerApply(PC pc, Vec x, Vec y) {
    Mat P_inv;

    // Retrieve the P^{-1} matrix from the PC context
    PCShellGetContext(pc, (void **)&P_inv);

    // Perform the matrix-vector multiplication y = P^{-1} * x
    MatMult(P_inv, x, y);

    return 0;
}