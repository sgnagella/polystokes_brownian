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
#include <petscksp.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
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

// ---------------------------------------------------------------------------------------------
// Stage-2b m2: a distributed MINRES solve of the arrowhead saddle system.
//
// Monomer partition: rank r owns monomers [m0,m1). The distributed solve vector uses a
// per-rank-contiguous layout  [ u_m_loc | l_m_loc | (rank0 only: u_c, l_c) ]  (a fixed
// permutation of the serial order, so the replicated assembly is bridged by plain array copies
// -- no VecScatter needed). The shell MatMult is the verified m1 kernel; the colloid DOFs live
// on rank 0 and are broadcast into each matvec, with M^cm*u_m reduced by MPI_Allreduce.
namespace {

struct DistCtx {
    PetscMPIInt rank, size;
    PetscInt nm3, nc11, nc6, nm3nc6, nm3nc11, nm6nc17;
    PetscInt d0, dloc;                    // local monomer DOF range [d0, d0+dloc)
    PetscReal beta_inv;
    std::vector<PetscScalar> Mcm_loc;     // nc11 x dloc, col-major: Mcm_loc[k + jl*nc11] = M^cm[k, d0+jl]
    const PetscScalar *Mcc = nullptr;     // nc11 x nc11, col-major (borrowed from Mcc_block)
};

// y = A x on the distributed layout (see file header for the block formulas).
PetscErrorCode DistMult(Mat A, Vec x, Vec y)
{
    DistCtx *c; MatShellGetContext(A, &c);
    const PetscInt nc11 = c->nc11, nc6 = c->nc6, dloc = c->dloc;
    const PetscScalar *xa; PetscScalar *ya;
    VecGetArrayRead(x, &xa); VecGetArray(y, &ya);
    const PetscScalar *um = xa, *lm = xa + dloc;

    // Colloid DOFs live on rank 0; broadcast to all for the local M^mc/M^cc products.
    std::vector<PetscScalar> uc(nc11, 0.0), lc(nc6, 0.0);
    if (c->rank == 0) {
        for (PetscInt k = 0; k < nc11; k++) uc[k] = xa[2*dloc + k];
        for (PetscInt j = 0; j < nc6;  j++) lc[j] = xa[2*dloc + nc11 + j];
    }
    MPI_Bcast(uc.data(), nc11, MPIU_SCALAR, 0, PETSC_COMM_WORLD);
    MPI_Bcast(lc.data(), nc6,  MPIU_SCALAR, 0, PETSC_COMM_WORLD);

    // Colloid coupling M^cm u_m: local partial over this rank's monomers, then Allreduce.
    std::vector<PetscScalar> pcm(nc11, 0.0), cmf(nc11, 0.0);
    for (PetscInt k = 0; k < nc11; k++)
        for (PetscInt jl = 0; jl < dloc; jl++) pcm[k] += c->Mcm_loc[k + jl*nc11] * um[jl];
    MPI_Allreduce(pcm.data(), cmf.data(), nc11, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);

    // Local monomer outputs.
    PetscScalar *ym = ya, *ylm = ya + dloc;
    for (PetscInt jl = 0; jl < dloc; jl++) {
        PetscScalar mmc = 0.0;
        for (PetscInt k = 0; k < nc11; k++) mmc += c->Mcm_loc[k + jl*nc11] * uc[k];
        ym[jl]  = c->beta_inv * um[jl] + mmc - lm[jl];
        ylm[jl] = -um[jl];
    }
    // Colloid outputs on rank 0.
    if (c->rank == 0) {
        PetscScalar *yc = ya + 2*dloc, *ylc = ya + 2*dloc + nc11;
        for (PetscInt k = 0; k < nc11; k++) {
            PetscScalar s = cmf[k];
            for (PetscInt kk = 0; kk < nc11; kk++) s += c->Mcc[k + kk*nc11] * uc[kk];
            if (k < nc6) s += -lc[k];
            yc[k] = s;
        }
        for (PetscInt j = 0; j < nc6; j++) ylc[j] = -uc[j];
    }
    VecRestoreArrayRead(x, &xa); VecRestoreArray(y, &ya);
    return 0;
}

// diag(A): beta_inv on u_m, diag(M^cc) on u_c, 0 on the constraint DOFs (Jacobi fix-diagonal
// replaces the zeros).
PetscErrorCode DistDiag(Mat A, Vec d)
{
    DistCtx *c; MatShellGetContext(A, &c);
    VecSet(d, 0.0);
    PetscScalar *da; VecGetArray(d, &da);
    for (PetscInt jl = 0; jl < c->dloc; jl++) da[jl] = c->beta_inv;      // u_m
    if (c->rank == 0)
        for (PetscInt k = 0; k < c->nc11; k++) da[2*c->dloc + k] = c->Mcc[k + k*c->nc11];  // u_c
    VecRestoreArray(d, &da);
    return 0;
}

} // anonymous namespace

bool PolyStokes::verify_distributed_solve()
{
    const PetscInt nm3 = consts.nm3, nc11 = consts.nc11, nc6 = consts.nc6;
    const PetscInt nm3nc6 = consts.nm3nc6, nm3nc11 = consts.nm3nc11, nm6nc17 = consts.nm6nc17;
    const PetscInt Nm = pinfo.Nm;

    // Monomer partition (even, contiguous).
    const PetscInt base = Nm / mpi_size, rem = Nm % mpi_size;
    const PetscInt m0 = mpi_rank * base + std::min((PetscInt)mpi_rank, rem);
    const PetscInt nm_loc = base + (mpi_rank < rem ? 1 : 0);
    const PetscInt d0 = 3 * m0, dloc = 3 * nm_loc;
    const PetscInt local_size = 2*dloc + (mpi_rank == 0 ? nc11 + nc6 : 0);

    // Replicate the current rhs onto every rank.
    VecScatter toall; Vec rhs_all;
    VecScatterCreateToAll(rhs, &toall, &rhs_all);
    VecScatterBegin(toall, rhs, rhs_all, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(toall, rhs, rhs_all, INSERT_VALUES, SCATTER_FORWARD);
    const PetscScalar *ra; VecGetArrayRead(rhs_all, &ra);
    std::vector<PetscScalar> rhs_full(ra, ra + nm6nc17);
    VecRestoreArrayRead(rhs_all, &ra);
    VecScatterDestroy(&toall); VecDestroy(&rhs_all);

    // Build the distributed context (extract this rank's Mcm columns).
    DistCtx c;
    c.rank = mpi_rank; c.size = mpi_size;
    c.nm3 = nm3; c.nc11 = nc11; c.nc6 = nc6; c.nm3nc6 = nm3nc6; c.nm3nc11 = nm3nc11; c.nm6nc17 = nm6nc17;
    c.d0 = d0; c.dloc = dloc; c.beta_inv = pinfo.beta_inv;
    const PetscScalar *cm; MatDenseGetArrayRead(Mcm_block, &cm);
    c.Mcm_loc.resize((size_t)nc11 * dloc);
    for (PetscInt jl = 0; jl < dloc; jl++)
        for (PetscInt k = 0; k < nc11; k++) c.Mcm_loc[k + jl*nc11] = cm[k + (d0 + jl)*nc11];
    MatDenseRestoreArrayRead(Mcm_block, &cm);
    const PetscScalar *cc; MatDenseGetArrayRead(Mcc_block, &cc);
    std::vector<PetscScalar> Mcc_copy(cc, cc + (size_t)nc11*nc11);
    MatDenseRestoreArrayRead(Mcc_block, &cc);
    c.Mcc = Mcc_copy.data();

    // Distributed shell + vectors.
    Mat Ampi; MatCreateShell(PETSC_COMM_WORLD, local_size, local_size, nm6nc17, nm6nc17, &c, &Ampi);
    MatShellSetOperation(Ampi, MATOP_MULT,         (void(*)(void))DistMult);
    MatShellSetOperation(Ampi, MATOP_GET_DIAGONAL, (void(*)(void))DistDiag);
    Vec bmpi, xmpi; VecCreateMPI(PETSC_COMM_WORLD, local_size, nm6nc17, &bmpi); VecDuplicate(bmpi, &xmpi);

    // Fill b from the replicated rhs (per-rank layout) and solve.
    PetscScalar *ba; VecGetArray(bmpi, &ba);
    for (PetscInt jl = 0; jl < dloc; jl++) { ba[jl] = rhs_full[d0 + jl]; ba[dloc + jl] = rhs_full[nm3nc11 + d0 + jl]; }
    if (mpi_rank == 0) {
        for (PetscInt k = 0; k < nc11; k++) ba[2*dloc + k]        = rhs_full[nm3 + k];
        for (PetscInt j = 0; j < nc6;  j++) ba[2*dloc + nc11 + j] = rhs_full[nm3nc11 + nm3 + j];
    }
    VecRestoreArray(bmpi, &ba);
    VecSet(xmpi, 0.0);

    KSP kmpi; PC pmpi;
    KSPCreate(PETSC_COMM_WORLD, &kmpi);
    KSPSetOperators(kmpi, Ampi, Ampi);
    KSPSetType(kmpi, KSPMINRES);
    KSPGetPC(kmpi, &pmpi); PCSetType(pmpi, PCJACOBI); PCJacobiSetFixDiagonal(pmpi, PETSC_TRUE);
    KSPSetTolerances(kmpi, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSolve(kmpi, bmpi, xmpi);
    PetscInt its; KSPGetIterationNumber(kmpi, &its);

    // Gather the distributed solution back to a full replicated array.
    std::vector<PetscScalar> x_full(nm6nc17, 0.0);
    const PetscScalar *xa; VecGetArrayRead(xmpi, &xa);
    std::vector<PetscScalar> um_loc(dloc), lm_loc(dloc);
    for (PetscInt jl = 0; jl < dloc; jl++) { um_loc[jl] = xa[jl]; lm_loc[jl] = xa[dloc + jl]; }
    std::vector<PetscScalar> uc(nc11, 0.0), lc(nc6, 0.0);
    if (mpi_rank == 0) {
        for (PetscInt k = 0; k < nc11; k++) uc[k] = xa[2*dloc + k];
        for (PetscInt j = 0; j < nc6;  j++) lc[j] = xa[2*dloc + nc11 + j];
    }
    VecRestoreArrayRead(xmpi, &xa);
    MPI_Bcast(uc.data(), nc11, MPIU_SCALAR, 0, PETSC_COMM_WORLD);
    MPI_Bcast(lc.data(), nc6,  MPIU_SCALAR, 0, PETSC_COMM_WORLD);
    std::vector<int> cnts(mpi_size), disp(mpi_size);
    for (int r = 0; r < mpi_size; r++) {
        PetscInt rm0 = r * base + std::min((PetscInt)r, rem), rnm = base + (r < rem ? 1 : 0);
        cnts[r] = 3 * rnm; disp[r] = 3 * rm0;
    }
    std::vector<PetscScalar> um_full(nm3), lm_full(nm3);
    MPI_Allgatherv(um_loc.data(), dloc, MPIU_SCALAR, um_full.data(), cnts.data(), disp.data(), MPIU_SCALAR, PETSC_COMM_WORLD);
    MPI_Allgatherv(lm_loc.data(), dloc, MPIU_SCALAR, lm_full.data(), cnts.data(), disp.data(), MPIU_SCALAR, PETSC_COMM_WORLD);
    for (PetscInt i = 0; i < nm3; i++) { x_full[i] = um_full[i]; x_full[nm3nc11 + i] = lm_full[i]; }
    for (PetscInt k = 0; k < nc11; k++) x_full[nm3 + k] = uc[k];
    for (PetscInt j = 0; j < nc6;  j++) x_full[nm3nc11 + nm3 + j] = lc[j];

    // The integrator reads the velocity block X[nm3nc11 : nm6nc17]; report its norm (comparable
    // across rank counts and to serial).
    PetscReal vnorm = 0.0;
    for (PetscInt j = nm3nc11; j < nm6nc17; j++) vnorm += PetscRealPart(x_full[j])*PetscRealPart(x_full[j]);
    vnorm = std::sqrt(vnorm);

    // On a single rank, compare directly to the serial solve of the same rhs.
    PetscReal serdiff = -1.0;
    if (mpi_size == 1) {
        solve_saddle(Xdet, false);
        const PetscScalar *xs; VecGetArrayRead(Xdet, &xs);
        serdiff = 0.0;
        for (PetscInt i = 0; i < nm6nc17; i++) serdiff = std::max(serdiff, PetscAbsScalar(xs[i] - x_full[i]));
        VecRestoreArrayRead(Xdet, &xs);
    }

    if (mpi_rank == 0) {
        std::cout << "[mpi-solve] ranks=" << mpi_size << " Nm=" << Nm << " its=" << its
                  << "  ||v_block|| = " << std::scientific << (double)vnorm;
        if (serdiff >= 0.0) std::cout << "   max|x_mpi - x_serial| = " << (double)serdiff;
        std::cout << std::endl;
    }

    KSPDestroy(&kmpi); MatDestroy(&Ampi); VecDestroy(&bmpi); VecDestroy(&xmpi);
    return (mpi_size > 1) || (serdiff >= 0.0 && serdiff < 1e-6);
}
