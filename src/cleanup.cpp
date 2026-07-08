#include "arrays.h"
#include "data.h"
#include "Stokes.h"
#include <iostream>
using namespace arrays;

void PolyStokes::cleanup(){
    // Clean up
    // MatDestroy(&ZMUF);
    // MatDestroy(&ZMUS);
    // MatDestroy(&ZMES);
    // MatDestroy(&Pinv);
    
    // Free the pointer
    // delete dataStruct;
    // dataStruct = nullptr;

    PetscErrorCode ierr; 
    // ierr = MatDestroy(&M); CHKERRV(ierr);
    if (A) { ierr = MatDestroy(&A); CHKERRV(ierr); }
    if (B) { ierr = MatDestroy(&B); CHKERRV(ierr); }
    if (rhs) { ierr = VecDestroy(&rhs); CHKERRV(ierr); }
    if (X) { ierr = VecDestroy(&X); CHKERRV(ierr); }
    if (ksp) { ierr = KSPDestroy(&ksp); CHKERRV(ierr); }

    // Schur-complement workspace (only built when kT > 0 && !mm_HI)
    if (Smat) { ierr = MatDestroy(&Smat); CHKERRV(ierr); }

    // Arrowhead mobility pieces
    if (Mcm_block) { ierr = MatDestroy(&Mcm_block); CHKERRV(ierr); }
    if (Mcc_block) { ierr = MatDestroy(&Mcc_block); CHKERRV(ierr); }

    PetscBool petsc_initialized, petsc_finalized;
    PetscInitialized(&petsc_initialized);
    PetscFinalized(&petsc_finalized);

    // if (petsc_initialized && !petsc_finalized) {
    //     PetscFinalize();
    // }
    std::cout << "Cleanup complete" << std::endl;
    return;
}