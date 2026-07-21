// Matrix-free saddle operator A for the single-colloid / no-monomer-HI regime.
//
// The grand mobility is an "arrowhead" matrix:
//     Mob = [[ beta_inv*I , M^mc ],
//            [ M^cm       , M^cc ]]
// with M^mm = beta_inv*I diagonal (monomer self), M^cm = Mcm_block (nc11 x nm3,
// colloid<->monomer coupling, rebuilt each step), M^mc = M^cm^T, and M^cc = Mcc_block
// (nc11 x nc11, constant colloid self). The saddle matrix is
//     A = [[ Mob , B   ],
//          [ B^T , 0   ]]
// where B is the constant constraint coupling (arrays::B). None of these is dense in N,
// so A applies in O(N). This shell replaces the dense A when mm_HI == false.
//
// NOTE: assumes a single colloid (Mcc is the constant self-block; there is no
// colloid-colloid coupling). With Nc > 1 and no monomer HI, the BB pair block would be
// missing -- a warning is emitted in that case.

#include "arrays.h"
#include "params.h"
#include <petscmat.h>
#include <iostream>

using namespace arrays;

namespace {

struct ArrowheadCtx {
    PetscReal beta_inv;
    PetscInt  nm3, nc11, nm3nc11, nm3nc6;
    // reusable work vectors (all PETSC_COMM_SELF)
    Vec u_m, u_c, l, u_full, ym, yc, yl, btop;
};

// y = A x,  x = [u ; l]  (u: nm3nc11 mobility DOFs, l: nm3nc6 constraint DOFs)
//   u = [u_m ; u_c],
//   y_top = [ beta_inv*u_m + M^mc*u_c ; M^cm*u_m + M^cc*u_c ] + B*l
//   y_bot = B^T * u
PetscErrorCode ArrowheadMult(Mat A, Vec x, Vec y)
{
    ArrowheadCtx *c;
    PetscErrorCode ierr;
    ierr = MatShellGetContext(A, &c); CHKERRQ(ierr);

    const PetscScalar *xa;
    PetscScalar *p;
    ierr = VecGetArrayRead(x, &xa); CHKERRQ(ierr);
    ierr = VecGetArray(c->u_m, &p);    CHKERRQ(ierr); for (PetscInt i = 0; i < c->nm3;     i++) p[i] = xa[i];                 ierr = VecRestoreArray(c->u_m, &p);    CHKERRQ(ierr);
    ierr = VecGetArray(c->u_c, &p);    CHKERRQ(ierr); for (PetscInt i = 0; i < c->nc11;    i++) p[i] = xa[c->nm3 + i];        ierr = VecRestoreArray(c->u_c, &p);    CHKERRQ(ierr);
    ierr = VecGetArray(c->u_full, &p); CHKERRQ(ierr); for (PetscInt i = 0; i < c->nm3nc11; i++) p[i] = xa[i];                 ierr = VecRestoreArray(c->u_full, &p); CHKERRQ(ierr);
    ierr = VecGetArray(c->l, &p);      CHKERRQ(ierr); for (PetscInt i = 0; i < c->nm3nc6;  i++) p[i] = xa[c->nm3nc11 + i];    ierr = VecRestoreArray(c->l, &p);      CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(x, &xa); CHKERRQ(ierr);

    // Mobility block action.
    ierr = MatMultTranspose(Mcm_block, c->u_c, c->ym); CHKERRQ(ierr);   // ym = M^mc u_c
    ierr = VecAXPY(c->ym, c->beta_inv, c->u_m);        CHKERRQ(ierr);   // ym += beta_inv u_m
    ierr = MatMult(Mcm_block, c->u_m, c->yc);          CHKERRQ(ierr);   // yc = M^cm u_m
    ierr = MatMultAdd(Mcc_block, c->u_c, c->yc, c->yc); CHKERRQ(ierr);  // yc += M^cc u_c

    // Constraint coupling.
    ierr = MatMult(B, c->l, c->btop);                  CHKERRQ(ierr);   // btop = B l  (nm3nc11)
    ierr = MatMultTranspose(B, c->u_full, c->yl);      CHKERRQ(ierr);   // yl   = B^T u (nm3nc6)

    // Assemble y = [ ym + btop[0:nm3] ; yc + btop[nm3:nm3nc11] ; yl ].
    PetscScalar *ya;
    const PetscScalar *bt, *ymv, *ycv, *ylv;
    ierr = VecGetArray(y, &ya);            CHKERRQ(ierr);
    ierr = VecGetArrayRead(c->btop, &bt);  CHKERRQ(ierr);
    ierr = VecGetArrayRead(c->ym, &ymv);   CHKERRQ(ierr);
    ierr = VecGetArrayRead(c->yc, &ycv);   CHKERRQ(ierr);
    ierr = VecGetArrayRead(c->yl, &ylv);   CHKERRQ(ierr);
    for (PetscInt i = 0; i < c->nm3;    i++) ya[i]                = ymv[i] + bt[i];
    for (PetscInt i = 0; i < c->nc11;   i++) ya[c->nm3 + i]       = ycv[i] + bt[c->nm3 + i];
    for (PetscInt i = 0; i < c->nm3nc6; i++) ya[c->nm3nc11 + i]   = ylv[i];
    ierr = VecRestoreArrayRead(c->yl, &ylv);   CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(c->yc, &ycv);   CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(c->ym, &ymv);   CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(c->btop, &bt);  CHKERRQ(ierr);
    ierr = VecRestoreArray(y, &ya);            CHKERRQ(ierr);
    return 0;
}

// diag(A) = [ beta_inv (x nm3) ; diag(M^cc) ; 0 (x nm3nc6) ].
// The zero constraint-block diagonal is handled by PCJacobiSetFixDiagonal.
PetscErrorCode ArrowheadGetDiagonal(Mat A, Vec d)
{
    ArrowheadCtx *c;
    PetscErrorCode ierr;
    ierr = MatShellGetContext(A, &c); CHKERRQ(ierr);

    ierr = VecSet(d, 0.0); CHKERRQ(ierr);
    ierr = MatGetDiagonal(Mcc_block, c->u_c); CHKERRQ(ierr);   // reuse u_c as the nc11 diagonal

    PetscScalar *da;
    const PetscScalar *dc;
    ierr = VecGetArray(d, &da);           CHKERRQ(ierr);
    ierr = VecGetArrayRead(c->u_c, &dc);  CHKERRQ(ierr);
    for (PetscInt i = 0; i < c->nm3;  i++) da[i]          = c->beta_inv;
    for (PetscInt i = 0; i < c->nc11; i++) da[c->nm3 + i] = dc[i];
    ierr = VecRestoreArrayRead(c->u_c, &dc); CHKERRQ(ierr);
    ierr = VecRestoreArray(d, &da);          CHKERRQ(ierr);
    return 0;
}

PetscErrorCode ArrowheadDestroy(Mat A)
{
    ArrowheadCtx *c;
    PetscErrorCode ierr;
    ierr = MatShellGetContext(A, &c); CHKERRQ(ierr);
    if (c) {
        VecDestroy(&c->u_m);    VecDestroy(&c->u_c);  VecDestroy(&c->l);   VecDestroy(&c->u_full);
        VecDestroy(&c->ym);     VecDestroy(&c->yc);   VecDestroy(&c->yl);  VecDestroy(&c->btop);
        delete c;
    }
    return 0;
}

} // anonymous namespace

// Create the global arrays::A as a matrix-free shell (arrowhead mobility + B constraint).
namespace arrays {
void initialize_A_shell(PetscInt nm6nc17, PetscInt nm3nc11, PetscInt nm3nc6,
                        PetscInt nm3, PetscInt nc11, PetscInt Nc, PetscReal beta_inv)
{
    if (Nc != 1) {
        std::cout << "[matfree_A] WARNING: Nc=" << Nc << " but the arrowhead shell assumes a "
                  << "single colloid (no colloid-colloid coupling)." << std::endl;
    }

    ArrowheadCtx *c = new ArrowheadCtx();
    c->beta_inv = beta_inv;
    c->nm3 = nm3;  c->nc11 = nc11;  c->nm3nc11 = nm3nc11;  c->nm3nc6 = nm3nc6;

    VecCreateSeq(PETSC_COMM_SELF, nm3,     &c->u_m);
    VecCreateSeq(PETSC_COMM_SELF, nc11,    &c->u_c);
    VecCreateSeq(PETSC_COMM_SELF, nm3nc6,  &c->l);
    VecCreateSeq(PETSC_COMM_SELF, nm3nc11, &c->u_full);
    VecCreateSeq(PETSC_COMM_SELF, nm3,     &c->ym);
    VecCreateSeq(PETSC_COMM_SELF, nc11,    &c->yc);
    VecCreateSeq(PETSC_COMM_SELF, nm3nc6,  &c->yl);
    VecCreateSeq(PETSC_COMM_SELF, nm3nc11, &c->btop);

    // rep_comm is PETSC_COMM_WORLD in serial and PETSC_COMM_SELF under MPI (the distributed
    // solve uses its own COMM_WORLD shell in matfree_A_mpi.cpp). Creating this serial arrowhead
    // shell on WORLD with full local==global size is invalid on >1 rank (PetscSplitOwnership),
    // so use rep_comm to get a valid per-rank SELF shell that the replicated serial path can use.
    MatCreateShell(rep_comm, nm6nc17, nm6nc17, nm6nc17, nm6nc17, (void*)c, &A);
    MatShellSetOperation(A, MATOP_MULT,         (void(*)(void))ArrowheadMult);
    MatShellSetOperation(A, MATOP_GET_DIAGONAL, (void(*)(void))ArrowheadGetDiagonal);
    MatShellSetOperation(A, MATOP_DESTROY,      (void(*)(void))ArrowheadDestroy);
}
} // namespace arrays
