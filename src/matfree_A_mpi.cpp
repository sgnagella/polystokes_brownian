// Stage-2b, milestone 1: verify the DISTRIBUTED arrowhead matvec against the serial one.
//
// The arrowhead saddle operator (mm_HI=False, single colloid) acts on x = [u_m | u_c | l]:
//     y_m = beta_inv*u_m + M^mc*u_c - l_m           (nm3, monomer velocity)
//     y_c = M^cm*u_m + M^cc*u_c + [-l_c ; 0]        (nc11, colloid)
//     y_l = [-u_m ; -u_c[0:nc6]]                    (nm3nc6, constraint)
// where M^cm = Mcm_block (nc11 x nm3), M^mc = (M^cm)^T, M^cc = Mcc_block (nc11 x nc11), and the
// constraint block B = [-I_{nm3nc6}; 0] is a trivial signed injection (l_m = l[0:nm3],
// l_c = l[nm3:nm3nc6], size nc6). See ArrowheadMult in matfree_A.cpp.
//
// Under an MPI monomer partition (rank r owns monomers [m0,m1)), the ONLY cross-rank coupling
// is the colloid term M^cm*u_m: each rank forms its local nc11 partial and MPI_Allreduce-sums
// it. Everything else is local (u_c, Mcc replicated; M^mc*u_c, beta_inv*u_m, the -l/-u copies).
// This routine builds a deterministic x, computes y both ways, and reports the max discrepancy;
// on a single rank it also cross-checks the reference formula against the real ArrowheadMult so
// a formula error cannot hide. Physics is unchanged from the serial code — this verifies only
// the partition + reduction mechanics.

#include <petscmat.h>
#include <cmath>
#include <vector>
#include <iostream>
#include "arrays.h"
#include "Stokes.h"

using namespace arrays;

bool PolyStokes::verify_distributed_matvec()
{
    const PetscInt nm3     = consts.nm3;
    const PetscInt nc11    = consts.nc11;
    const PetscInt nc6     = consts.nc6;
    const PetscInt nm3nc6  = consts.nm3nc6;
    const PetscInt nm3nc11 = consts.nm3nc11;
    const PetscInt nm6nc17 = consts.nm6nc17;
    const PetscInt Nm      = pinfo.Nm;
    const PetscReal beta_inv = pinfo.beta_inv;

    // Deterministic, rank-independent input x (full replicated vector).
    std::vector<PetscScalar> x(nm6nc17);
    for (PetscInt i = 0; i < nm6nc17; i++) x[i] = std::sin(0.1 * (i + 1));
    const PetscScalar *u_m = &x[0];
    const PetscScalar *u_c = &x[nm3];
    const PetscScalar *l   = &x[nm3nc11];          // l_m = l[0:nm3], l_c = l[nm3:nm3nc6]

    // Replicated dense blocks (column-major, lda = nc11).
    const PetscScalar *cm, *cc;
    MatDenseGetArrayRead(Mcm_block, &cm);
    MatDenseGetArrayRead(Mcc_block, &cc);
    auto Mcm = [&](PetscInt k, PetscInt i){ return cm[k + i * nc11]; };   // M^cm[k,i]
    auto Mcc = [&](PetscInt k, PetscInt kk){ return cc[k + kk * nc11]; }; // M^cc[k,kk]

    // ---- Reference: serial matvec (every rank computes it redundantly) ----
    std::vector<PetscScalar> yref(nm6nc17, 0.0);
    PetscScalar *yr_m = &yref[0], *yr_c = &yref[nm3], *yr_l = &yref[nm3nc11];
    for (PetscInt k = 0; k < nc11; k++) {          // colloid: M^cm u_m + M^cc u_c + bc
        PetscScalar s = 0.0;
        for (PetscInt i = 0; i < nm3; i++) s += Mcm(k, i) * u_m[i];
        for (PetscInt kk = 0; kk < nc11; kk++) s += Mcc(k, kk) * u_c[kk];
        if (k < nc6) s += -l[nm3 + k];             // bc = [-l_c ; 0]
        yr_c[k] = s;
    }
    for (PetscInt i = 0; i < nm3; i++) {           // monomer velocity: beta_inv u_m + M^mc u_c - l_m
        PetscScalar mmc = 0.0;
        for (PetscInt k = 0; k < nc11; k++) mmc += Mcm(k, i) * u_c[k];   // (M^cm)^T u_c
        yr_m[i] = beta_inv * u_m[i] + mmc - l[i];
    }
    for (PetscInt j = 0; j < nm3;    j++) yr_l[j]       = -u_m[j];       // constraint: -u_full[0:nm3nc6]
    for (PetscInt j = 0; j < nc6;    j++) yr_l[nm3 + j] = -u_c[j];

    // ---- Distributed: partition monomers [m0,m1) across ranks (even, contiguous) ----
    const PetscInt base = Nm / mpi_size, rem = Nm % mpi_size;
    const PetscInt m0 = mpi_rank * base + std::min((PetscInt)mpi_rank, rem);
    const PetscInt nm_loc = base + (mpi_rank < rem ? 1 : 0);
    const PetscInt m1 = m0 + nm_loc;
    const PetscInt d0 = 3 * m0, dloc = 3 * nm_loc;                       // local DOF range [d0, d0+dloc)

    // Colloid coupling M^cm u_m: local partial over this rank's monomer DOFs, then Allreduce.
    std::vector<PetscScalar> pcm(nc11, 0.0);
    for (PetscInt k = 0; k < nc11; k++)
        for (PetscInt i = d0; i < d0 + dloc; i++) pcm[k] += Mcm(k, i) * u_m[i];
    std::vector<PetscScalar> cm_full(nc11, 0.0);
    MPI_Allreduce(pcm.data(), cm_full.data(), nc11, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);

    // Local monomer outputs (velocity + constraint) over [d0, d0+dloc).
    std::vector<PetscScalar> ym_loc(dloc), ylm_loc(dloc);
    for (PetscInt i = d0; i < d0 + dloc; i++) {
        PetscScalar mmc = 0.0;
        for (PetscInt k = 0; k < nc11; k++) mmc += Mcm(k, i) * u_c[k];
        ym_loc[i - d0]  = beta_inv * u_m[i] + mmc - l[i];
        ylm_loc[i - d0] = -u_m[i];
    }
    // Colloid outputs (replicated on every rank via the Allreduce above).
    std::vector<PetscScalar> yc(nc11), ylc(nc6);
    for (PetscInt k = 0; k < nc11; k++) {
        PetscScalar s = cm_full[k];
        for (PetscInt kk = 0; kk < nc11; kk++) s += Mcc(k, kk) * u_c[kk];
        if (k < nc6) s += -l[nm3 + k];
        yc[k] = s;
    }
    for (PetscInt j = 0; j < nc6; j++) ylc[j] = -u_c[j];

    MatDenseRestoreArrayRead(Mcm_block, &cm);
    MatDenseRestoreArrayRead(Mcc_block, &cc);

    // Gather the distributed monomer outputs back to full replicated arrays for comparison.
    std::vector<int> cnts(mpi_size), disp(mpi_size);
    for (int r = 0; r < mpi_size; r++) {
        PetscInt rm0 = r * base + std::min((PetscInt)r, rem);
        PetscInt rnm = base + (r < rem ? 1 : 0);
        cnts[r] = 3 * rnm;  disp[r] = 3 * rm0;
    }
    std::vector<PetscScalar> ym_full(nm3), ylm_full(nm3);
    MPI_Allgatherv(ym_loc.data(),  dloc, MPIU_SCALAR, ym_full.data(),  cnts.data(), disp.data(), MPIU_SCALAR, PETSC_COMM_WORLD);
    MPI_Allgatherv(ylm_loc.data(), dloc, MPIU_SCALAR, ylm_full.data(), cnts.data(), disp.data(), MPIU_SCALAR, PETSC_COMM_WORLD);

    // Assemble ydist = [y_m | y_c | y_l_m | y_l_c] and diff against yref.
    PetscReal maxdiff = 0.0;
    for (PetscInt i = 0; i < nm3;  i++) maxdiff = std::max(maxdiff, PetscAbsScalar(ym_full[i]  - yref[i]));
    for (PetscInt k = 0; k < nc11; k++) maxdiff = std::max(maxdiff, PetscAbsScalar(yc[k]        - yref[nm3 + k]));
    for (PetscInt j = 0; j < nm3;  j++) maxdiff = std::max(maxdiff, PetscAbsScalar(ylm_full[j]  - yref[nm3nc11 + j]));
    for (PetscInt j = 0; j < nc6;  j++) maxdiff = std::max(maxdiff, PetscAbsScalar(ylc[j]        - yref[nm3nc11 + nm3 + j]));

    // On a single rank, also cross-check the reference formula against the real ArrowheadMult.
    PetscReal refdiff = -1.0;
    if (mpi_size == 1) {
        Vec xv, yv;
        VecCreateSeq(PETSC_COMM_SELF, nm6nc17, &xv);
        VecDuplicate(xv, &yv);
        PetscScalar *xa; VecGetArray(xv, &xa);
        for (PetscInt i = 0; i < nm6nc17; i++) xa[i] = x[i];
        VecRestoreArray(xv, &xa);
        MatMult(A, xv, yv);                        // the real serial arrowhead shell
        const PetscScalar *ya; VecGetArrayRead(yv, &ya);
        refdiff = 0.0;
        for (PetscInt i = 0; i < nm6nc17; i++) refdiff = std::max(refdiff, PetscAbsScalar(ya[i] - yref[i]));
        VecRestoreArrayRead(yv, &ya);
        VecDestroy(&xv); VecDestroy(&yv);
    }

    if (mpi_rank == 0) {
        std::cout << "[mpi-selftest] ranks=" << mpi_size << " Nm=" << Nm
                  << "  max|y_dist - y_serial| = " << (double)maxdiff;
        if (refdiff >= 0.0) std::cout << "   (formula vs real ArrowheadMult = " << (double)refdiff << ")";
        std::cout << std::endl;
    }
    return maxdiff < 1e-10;
}
