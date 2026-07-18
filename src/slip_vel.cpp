#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <petscksp.h>
#include <petscblaslapack.h>
#include "arrays.h"
#include "Stokes.h"
using namespace arrays;

void PolyStokes::sample_slip_vel(){
    // This routine samples the random slip velocities
    // needed to compute the far-field Brownian forces
    int& nm3nc11 = consts.nm3nc11;
    for(int i = 0; i < nm3nc11; i++){ fb[i] = normal_dist(rng);}
    return;
}

void PolyStokes::solve_slip_vel(){
    // This routine computes the 11N Brownian slip velocities
    // by computing the action of sqrt(M)*\Psi

    PetscErrorCode ierr;
    VecZeroEntries(W);
    sample_slip_vel();

    if( mm_HI ){
        // Full grand-mobility path: W <- sqrt(M) * noise via SLEPc MFN, where M is the
        // mobility block of A (top-left [0, nm3nc11)). Bind that view as the operator.
        Mat Mview;
        ierr = MatDenseGetSubMatrix(A, 0, consts.nm3nc11, 0, consts.nm3nc11, &Mview); CHKERRV(ierr);
        ierr = MFNSetOperator(mfn, Mview); CHKERRV(ierr);

        PetscScalar *arr;
        ierr = VecGetArray(W, &arr); CHKERRV(ierr);
        std::copy(fb.begin(), fb.end(), arr);
        ierr = VecRestoreArray(W, &arr); CHKERRV(ierr);

        // Compute
        ierr = MFNSolve(mfn, W, W); CHKERRV(ierr);

        MFNConvergedReason reason;
        MFNGetConvergedReason(mfn, &reason);
        ierr = MatDenseRestoreSubMatrix(A, &Mview); CHKERRV(ierr);
        if (reason < 0) {
            PetscPrintf(PETSC_COMM_WORLD, "MFN solver did not converge\n");
            return;
        }
    }
    else{
        // Monomer-monomer HI disabled: sample W with the block-Cholesky factor,
        // squaring the root only over the small colloid Schur complement.
        build_slip_vel_schur();
    }

    // Add to rhs
    PetscReal scale = PetscSqrtReal(2 * pinfo.kT / timeinfo.dt);
    VecScale(W, scale);

    // Create an index set defining where in the large vector to add W
    IS is;
    PetscInt start = 0;  // starting index in the large vector
    PetscInt size;
    VecGetLocalSize(W, &size);
    ISCreateStride(PETSC_COMM_WORLD, size, start, 1, &is);

    // Create scatter context
    VecScatter scatter;
    VecScatterCreate(W, NULL, rhs, is, &scatter);

    // Scatter with ADD_VALUES to add W into large_vec
    VecScatterBegin(scatter, W, rhs, ADD_VALUES, SCATTER_FORWARD);
    VecScatterEnd(scatter, W, rhs, ADD_VALUES, SCATTER_FORWARD);

    VecScatterDestroy(&scatter);
    ISDestroy(&is);

    return;
}

void PolyStokes::sync_mcc_schur_correction(){
    // Refresh Mcc_block = Mcc_base + adaptive eigenvalue-floor correction of the colloid
    // Schur complement S = M^cc - beta*M^cm*(M^cm)^T, using the CURRENT Mcm_block (just
    // rebuilt this stage by mob()). Mcc_block otherwise persists whatever correction was
    // last applied -- possibly on a PREVIOUS step, at a different position -- since
    // fill_self() only resets it once, at t=0. Called once per stage, right after mob()
    // and before any solve reads Mcc_block, so solve_deterministic_vel() and the
    // Brownian noise sample (build_slip_vel_schur(), which reuses the schur_Q /
    // schur_lambda_corrected cache filled here) agree on M^cc within that stage.
    //
    // No-op when mm_HI (the dense grand-mobility path doesn't use this arrowhead block)
    // or when nc11 exceeds the Lanczos crossover (that path samples noise without
    // correcting Mcc_block -- see schur_sqrt_lanczos's note).
    if (mm_HI) return;

    PetscInt nc11 = consts.nc11;
    PetscInt lanczos_threshold = 64;
    { const char *e = std::getenv("POLYSTOKES_LANCZOS_NC"); if (e) lanczos_threshold = atoi(e); }
    if (nc11 > lanczos_threshold) return;

    PetscErrorCode ierr;
    double beta = pinfo.beta;

    // S = M^cc - beta M^cm M^mc = Mcc_base - beta Bcm Bcm^T, built into the persistent
    // workspace Smat: S = Q diag(lambda) Q^T.
    Mat P;
    ierr = MatMatTransposeMult(Mcm_block, Mcm_block, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &P); CHKERRV(ierr);
    // Stage-2c: when the Mcm columns are distributed (mpi_size>1), each rank's Mcm holds only
    // its local monomers' columns, so this product is the LOCAL partial of M^cm(M^cm)^T
    // (= sum over local monomers of the outer products). Sum the nc11 x nc11 partials across
    // ranks to get the full contraction on every rank.
    if (mpi_size > 1) {
        PetscScalar *Parr; ierr = MatDenseGetArray(P, &Parr); CHKERRV(ierr);
        std::vector<PetscScalar> tmp((size_t)nc11 * nc11);
        MPI_Allreduce(Parr, tmp.data(), nc11 * nc11, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);
        std::copy(tmp.begin(), tmp.end(), Parr);
        ierr = MatDenseRestoreArray(P, &Parr); CHKERRV(ierr);
    }
    ierr = MatScale(P, -beta); CHKERRV(ierr);
    ierr = MatAXPY(P, 1.0, Mcc_base, SAME_NONZERO_PATTERN); CHKERRV(ierr);
    ierr = MatCopy(P, Smat, SAME_NONZERO_PATTERN); CHKERRV(ierr);
    ierr = MatDestroy(&P); CHKERRV(ierr);

    PetscScalar *Sarr;
    ierr = MatDenseGetArray(Smat, &Sarr); CHKERRV(ierr);   // column-major, lda = nc11

    PetscBLASInt n, lda, lwork, info;
    ierr = PetscBLASIntCast(nc11, &n); CHKERRV(ierr);
    lda = n;
    std::vector<PetscReal> w(nc11);          // eigenvalues, ascending
    const char jobz = 'V', uplo = 'U';

    // optimal workspace query, then the actual decomposition (overwrites Sarr with Q)
    PetscScalar work_query;
    lwork = -1;
    LAPACKsyev_(&jobz, &uplo, &n, Sarr, &lda, w.data(), &work_query, &lwork, &info);
    ierr = PetscBLASIntCast((PetscInt)PetscRealPart(work_query), &lwork); CHKERRV(ierr);
    std::vector<PetscScalar> work(lwork);
    LAPACKsyev_(&jobz, &uplo, &n, Sarr, &lda, w.data(), work.data(), &lwork, &info);
    if (info != 0) {
        PetscPrintf(PETSC_COMM_WORLD, "Schur eigendecomposition (syev) failed, info=%d\n", (int)info);
    }
    // Sarr columns now hold the eigenvectors Q: Q[i,j] = Sarr[i + j*lda].

    // Adaptive, per-eigen-direction correction: lambda_corrected_j = max(lambda_j, eps);
    // delta_j = lambda_corrected_j - lambda_j is zero for any direction that was already
    // >= eps, and only pushes up the directions that actually need it. The correction
    // Delta = Q diag(delta) Q^T is added onto Mcc_block (reset from Mcc_base first, so
    // nothing accumulates step to step).
    const PetscReal eps = 0.1;   // small positive margin, not a large blanket shift
    std::vector<PetscReal> delta(nc11);
    PetscInt nneg = 0;
    for (PetscInt j = 0; j < nc11; j++) {
        PetscReal lam = PetscRealPart(w[j]);
        delta[j] = (lam < eps) ? (eps - lam) : 0.0;
        if (lam < 0.0) nneg++;
    }

    ierr = MatCopy(Mcc_base, Mcc_block, SAME_NONZERO_PATTERN); CHKERRV(ierr);
    {
        PetscScalar *ccarr;
        ierr = MatDenseGetArray(Mcc_block, &ccarr); CHKERRV(ierr);   // column-major, lda = nc11
        for (PetscInt a = 0; a < nc11; a++) {
            for (PetscInt b = 0; b < nc11; b++) {
                PetscScalar corr = 0.0;
                for (PetscInt j = 0; j < nc11; j++) {
                    if (delta[j] == 0.0) continue;
                    corr += Sarr[a + j*lda] * delta[j] * Sarr[b + j*lda];
                }
                ccarr[a + b*nc11] += corr;
            }
        }
        ierr = MatDenseRestoreArray(Mcc_block, &ccarr); CHKERRV(ierr);
    }

    // Cache Q and the floored spectrum for build_slip_vel_schur() to reuse (avoids a
    // second syev call for the noise sample).
    schur_Q.assign(Sarr, Sarr + (size_t)nc11 * nc11);
    schur_lambda_corrected.resize(nc11);
    for (PetscInt j = 0; j < nc11; j++) {
        schur_lambda_corrected[j] = PetscRealPart(w[j]) + delta[j];
    }

    ierr = MatDenseRestoreArray(Smat, &Sarr); CHKERRV(ierr);

    // Background stats: accumulate every stage (whether or not we print). w is ascending, so
    // w[0] is this stage's most-negative eigenvalue.
    neig_stats.corr_stages++;
    if (nneg > 0) {
        double worst = (double)PetscRealPart(w[0]);
        neig_stats.corr_events++;
        neig_stats.corr_count += (long)nneg;
        neig_stats.corr_sum_worst += worst;
        if (worst < neig_stats.corr_worst) neig_stats.corr_worst = worst;
        if (warn_neg_eig) {
            PetscPrintf(PETSC_COMM_WORLD,
                "[schur] corrected %ld negative Schur eigenvalue(s), most negative = %.6e "
                "(truncated far-field mobility not SPD)\n", (long)nneg, worst);
        }
    }

    return;
}

void PolyStokes::build_slip_vel_schur(){
    // Block-Cholesky sampling of the far-field Brownian slip velocity for the
    // mm_HI == false case. With the monomer block A = M^mm = beta_inv*I, the
    // lower factor L = [[A^{1/2}, 0], [M^cm A^{-1/2}, S^{1/2}]] gives, for white
    // noise xi = fb split into monomer (nm3) and colloid (nc11) parts:
    //     u_m = sqrt(beta_inv) * xi_m
    //     u_c = sqrt(beta) * (M^cm xi_m) + S^{1/2} xi_c
    //     S   = M^cc - beta * M^cm (M^cm)^T           (beta = 1/beta_inv)
    // The assembled W = [u_m; u_c] then flows through the common scale/scatter
    // tail of solve_slip_vel(). Sub-blocks of the dense grand mobility M are
    // taken as zero-copy views via MatDenseGetSubMatrix (one view at a time).

    PetscErrorCode ierr;

    PetscInt nm3     = consts.nm3;
    PetscInt nm3nc11 = consts.nm3nc11;
    PetscInt nc11    = consts.nc11;
    double   beta     = pinfo.beta;
    double   beta_inv = pinfo.beta_inv;

    // Split the sampled noise fb into xi_m (nm3) and xi_c (nc11).
    Vec xi_m, xi_c;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nm3,  &xi_m); CHKERRV(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF, nc11, &xi_c); CHKERRV(ierr);
    {
        PetscScalar *am, *ac;
        ierr = VecGetArray(xi_m, &am); CHKERRV(ierr);
        ierr = VecGetArray(xi_c, &ac); CHKERRV(ierr);
        std::copy(fb.begin(),       fb.begin() + nm3,     am);
        std::copy(fb.begin() + nm3, fb.begin() + nm3nc11, ac);
        ierr = VecRestoreArray(xi_m, &am); CHKERRV(ierr);
        ierr = VecRestoreArray(xi_c, &ac); CHKERRV(ierr);
    }

    // Own copies of the colloid sub-blocks M^cm (nc11 x nm3) and M^cc (nc11 x nc11),
    // taken from zero-copy views of the dense M. Owned copies let both the matrix-free
    // Lanczos matvec and the dense path use them freely (MatDenseGetSubMatrix only
    // checks out one view at a time). The copy is O(nc11 nm3) -- at most one Lanczos
    // matvec -- so it is negligible next to either sqrt path.
    // The colloid sub-blocks are the standalone arrowhead mobility pieces (assembled in
    // mob()/fill_self), independent of how the saddle operator A is represented:
    //   M^cm = Mcm_block (nc11 x nm3, rebuilt each step), M^cc = Mcc_block (nc11 x nc11, const).
    // Cross term: uc = sqrt(beta) * (M^cm xi_m).
    Vec uc;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nc11, &uc); CHKERRV(ierr);
    // Stage-2c: Mcm_block holds only this rank's local columns, so M^cm xi_m needs the LOCAL
    // slice of the (replicated) noise, and the result is a partial summed across ranks. xi_m
    // itself stays full for the monomer slip u_m below.
    Vec xi_m_use = xi_m; Vec xi_m_loc = nullptr;
    if (mpi_size > 1) {
        PetscInt m0, nloc; arrays::mono_partition(pinfo.Nm, mpi_rank, mpi_size, m0, nloc);
        ierr = VecCreateSeq(PETSC_COMM_SELF, 3*nloc, &xi_m_loc); CHKERRV(ierr);
        PetscScalar *xl; ierr = VecGetArray(xi_m_loc, &xl); CHKERRV(ierr);
        for (PetscInt t = 0; t < 3*nloc; t++) xl[t] = fb[3*m0 + t];
        ierr = VecRestoreArray(xi_m_loc, &xl); CHKERRV(ierr);
        xi_m_use = xi_m_loc;
    }
    ierr = MatMult(Mcm_block, xi_m_use, uc); CHKERRV(ierr);
    // (Dense Schur path only; the Lanczos path, nc11 > threshold, is not used in the
    // single-colloid regime and would need the same treatment on its matvec.)
    if (mpi_size > 1) {
        PetscScalar *ua; ierr = VecGetArray(uc, &ua); CHKERRV(ierr);
        std::vector<PetscScalar> tmp(nc11);
        MPI_Allreduce(ua, tmp.data(), nc11, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);
        std::copy(tmp.begin(), tmp.end(), ua);
        ierr = VecRestoreArray(uc, &ua); CHKERRV(ierr);
    }
    if (xi_m_loc) { ierr = VecDestroy(&xi_m_loc); CHKERRV(ierr); }
    ierr = VecScale(uc, PetscSqrtReal(beta)); CHKERRV(ierr);

    // Add S^{1/2} xi_c onto uc, where S = M^cc - beta M^cm M^mc = Ccc - beta Bcm Bcm^T.
    // Both routes floor negative eigenvalues to zero (the truncated far-field mobility
    // is not always SPD; those modes then carry no thermal noise -- the standard
    // Brownian-dynamics remedy). For small nc11 a direct dense eigendecomposition is
    // cheapest; for large nc11 a matrix-free Lanczos avoids the O(nc11^2 nm3) formation
    // of M^cm M^mc and the O(nc11^3) full eigendecomposition, needing only k matvecs.
    // Crossover: direct dense eigendecomposition at/below the threshold, matrix-free
    // Lanczos above it. Default 64; override with POLYSTOKES_LANCZOS_NC (the optimal
    // crossover is size/machine dependent, and =0 forces Lanczos for testing).
    PetscInt lanczos_threshold = 64;
    { const char *e = std::getenv("POLYSTOKES_LANCZOS_NC"); if (e) lanczos_threshold = atoi(e); }

    if (nc11 > lanczos_threshold) {
        // NOTE: the Lanczos path floors the small projected tridiagonal locally (see
        // schur_sqrt_lanczos) but does not feed a correction back into Mcc_block -- unlike
        // the dense path below, it does not keep the deterministic solve consistent with
        // the sqrt. Not reached in the current single-colloid regime (nc11=11 << threshold);
        // revisit if scaling to many colloids ever engages this path.
        schur_sqrt_lanczos(Mcm_block, Mcc_block, beta, xi_c, uc, /*kmax=*/40);
    }
    else {
        // Dense path: Mcc_block and the Schur eigendecomposition (schur_Q,
        // schur_lambda_corrected) were already refreshed for the CURRENT position by
        // sync_mcc_schur_correction() -- called right after mob(), before any solve
        // reads Mcc_block (see run.cpp) -- so the deterministic solve and this noise
        // sample agree on M^cc. Reuse that cache here rather than repeating the syev.
        PetscInt lda = nc11;
        const PetscScalar *Sarr = schur_Q.data();

        const PetscScalar *xc;
        ierr = VecGetArrayRead(xi_c, &xc); CHKERRV(ierr);

        // y = diag(sqrt(lambda_corrected)) * (Q^T xi_c)
        std::vector<PetscScalar> y(nc11, 0.0);
        for (PetscInt j = 0; j < nc11; j++) {
            PetscScalar yj = 0.0;
            for (PetscInt i = 0; i < nc11; i++) yj += Sarr[i + j*lda] * xc[i];
            y[j] = PetscSqrtReal(schur_lambda_corrected[j]) * yj;
        }
        ierr = VecRestoreArrayRead(xi_c, &xc); CHKERRV(ierr);

        // uc += Q y   (add S^{1/2} xi_c onto the cross term already stored in uc)
        PetscScalar *ucarr;
        ierr = VecGetArray(uc, &ucarr); CHKERRV(ierr);
        for (PetscInt i = 0; i < nc11; i++) {
            PetscScalar si = 0.0;
            for (PetscInt j = 0; j < nc11; j++) si += Sarr[i + j*lda] * y[j];
            ucarr[i] += si;
        }
        ierr = VecRestoreArray(uc, &ucarr); CHKERRV(ierr);
    }

    // Mcm_block / Mcc_block are persistent (owned by arrays); do not destroy here.

    // Assemble W = [ sqrt(beta_inv) * xi_m ; uc ].
    {
        PetscScalar       *warr;
        const PetscScalar *xm, *ucv;
        ierr = VecGetArray(W, &warr); CHKERRV(ierr);
        ierr = VecGetArrayRead(xi_m, &xm);  CHKERRV(ierr);
        ierr = VecGetArrayRead(uc,   &ucv); CHKERRV(ierr);
        PetscReal cm = PetscSqrtReal(beta_inv);
        for(PetscInt i = 0; i < nm3;  i++){ warr[i]       = cm * xm[i]; }
        for(PetscInt i = 0; i < nc11; i++){ warr[nm3 + i] = ucv[i];     }
        ierr = VecRestoreArrayRead(uc,   &ucv); CHKERRV(ierr);
        ierr = VecRestoreArrayRead(xi_m, &xm);  CHKERRV(ierr);
        ierr = VecRestoreArray(W, &warr); CHKERRV(ierr);
    }

    ierr = VecDestroy(&xi_m); CHKERRV(ierr);
    ierr = VecDestroy(&xi_c); CHKERRV(ierr);
    ierr = VecDestroy(&uc);   CHKERRV(ierr);

    return;
}

void PolyStokes::schur_sqrt_lanczos(Mat Bcm, Mat Ccc, double beta, Vec b, Vec out, PetscInt kmax){
    // Accumulate sqrt(S) * b into `out`, where S = Ccc - beta*Bcm*Bcm^T is the colloid
    // Schur complement (Ccc = M^cc, Bcm = M^cm). Uses a Lanczos iteration that applies
    // S matrix-free -- S*q = Ccc*q - beta*Bcm*(Bcm^T*q) -- so M^cm M^mc is never formed.
    // The matrix square root is evaluated on the small k x k projected tridiagonal T_k
    // (dense syev), with negative eigenvalues floored to zero: this is cheap (k matvecs,
    // k << nc11) and robust to an indefinite truncated far-field mobility.
    //   sqrt(S) b ~= beta0 * Q_k * [ U diag(sqrt(max(theta,0))) U^T e1 ],   T_k = U diag(theta) U^T
    PetscErrorCode ierr;

    PetscInt nc11, nm3;
    ierr = VecGetSize(b, &nc11); CHKERRV(ierr);
    ierr = MatGetSize(Bcm, NULL, &nm3); CHKERRV(ierr);   // Bcm is nc11 x nm3

    PetscReal beta0;
    ierr = VecNorm(b, NORM_2, &beta0); CHKERRV(ierr);
    if (beta0 == 0.0) return;

    if (kmax > nc11) kmax = nc11;
    if (kmax < 1)    kmax = 1;

    // Work vectors for the matrix-free S*q.
    Vec w, Sq, tmp_nm3;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nc11, &w);       CHKERRV(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF, nc11, &Sq);      CHKERRV(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF, nm3,  &tmp_nm3); CHKERRV(ierr);

    std::vector<Vec>       Q;              // Lanczos basis (kept for reorthogonalization + recombination)
    std::vector<PetscReal> alpha, betad;  // tridiagonal diagonal / off-diagonal
    Q.reserve(kmax);

    Vec q1;
    ierr = VecDuplicate(b, &q1); CHKERRV(ierr);
    ierr = VecCopy(b, q1); CHKERRV(ierr);
    ierr = VecScale(q1, 1.0/beta0); CHKERRV(ierr);
    Q.push_back(q1);

    PetscInt m = 0;   // subspace dimension actually built
    for (PetscInt j = 0; j < kmax; j++) {
        Vec qj = Q[j];

        // w = S qj = Ccc qj - beta Bcm (Bcm^T qj)
        ierr = MatMultTranspose(Bcm, qj, tmp_nm3); CHKERRV(ierr);
        ierr = MatMult(Bcm, tmp_nm3, Sq); CHKERRV(ierr);
        ierr = MatMult(Ccc, qj, w); CHKERRV(ierr);
        ierr = VecAXPY(w, -beta, Sq); CHKERRV(ierr);

        if (j > 0) { ierr = VecAXPY(w, -betad[j-1], Q[j-1]); CHKERRV(ierr); }

        PetscScalar aj;
        ierr = VecDot(qj, w, &aj); CHKERRV(ierr);
        alpha.push_back(PetscRealPart(aj));
        ierr = VecAXPY(w, -PetscRealPart(aj), qj); CHKERRV(ierr);

        // full reorthogonalization (two passes) for a numerically faithful basis
        for (int pass = 0; pass < 2; pass++) {
            for (PetscInt i = 0; i <= j; i++) {
                PetscScalar h;
                ierr = VecDot(Q[i], w, &h); CHKERRV(ierr);
                ierr = VecAXPY(w, -h, Q[i]); CHKERRV(ierr);
            }
        }

        m = j + 1;
        PetscReal bj;
        ierr = VecNorm(w, NORM_2, &bj); CHKERRV(ierr);
        if (bj <= 1e-12 * beta0 || j == kmax - 1) break;   // invariant subspace or iteration cap

        betad.push_back(bj);
        Vec qnext;
        ierr = VecDuplicate(b, &qnext); CHKERRV(ierr);
        ierr = VecCopy(w, qnext); CHKERRV(ierr);
        ierr = VecScale(qnext, 1.0/bj); CHKERRV(ierr);
        Q.push_back(qnext);
    }

    // Eigendecompose the m x m symmetric tridiagonal T (dense syev), floor negatives.
    std::vector<PetscScalar> T((size_t)m*m, 0.0);   // column-major
    for (PetscInt i = 0; i < m; i++)   T[i + i*m] = alpha[i];
    for (PetscInt i = 0; i < m-1; i++) { T[(i+1) + i*m] = betad[i]; T[i + (i+1)*m] = betad[i]; }

    PetscBLASInt n_, lda_, lwork_, info_;
    ierr = PetscBLASIntCast(m, &n_); CHKERRV(ierr);
    lda_ = n_;
    std::vector<PetscReal> theta(m);
    const char jobz = 'V', uplo = 'U';
    PetscScalar wq; lwork_ = -1;
    LAPACKsyev_(&jobz, &uplo, &n_, T.data(), &lda_, theta.data(), &wq, &lwork_, &info_);
    ierr = PetscBLASIntCast((PetscInt)PetscRealPart(wq), &lwork_); CHKERRV(ierr);
    std::vector<PetscScalar> lwork(lwork_);
    LAPACKsyev_(&jobz, &uplo, &n_, T.data(), &lda_, theta.data(), lwork.data(), &lwork_, &info_);
    if (info_ != 0) {
        PetscPrintf(PETSC_COMM_WORLD, "Schur/Lanczos tridiagonal eig (syev) failed, info=%d\n", (int)info_);
    }
    // T columns now hold the eigenvectors U: U[i,j] = T[i + j*m].

    // g = beta0 * U diag(sqrt(max(theta,0))) U^T e1   (m-vector of recombination weights)
    PetscInt nneg = 0;
    std::vector<PetscScalar> g(m, 0.0);
    for (PetscInt i = 0; i < m; i++) {
        PetscScalar gi = 0.0;
        for (PetscInt jj = 0; jj < m; jj++) {
            PetscReal th = PetscRealPart(theta[jj]);
            if (th < 0.0) th = 0.0;
            gi += T[i + jj*m] * PetscSqrtReal(th) * T[0 + jj*m];   // U[i,jj]*sqrt(theta_jj)*U[0,jj]
        }
        g[i] = beta0 * gi;
    }
    for (PetscInt jj = 0; jj < m; jj++) if (PetscRealPart(theta[jj]) < 0.0) nneg++;

    // out += sum_i g[i] Q[i]
    ierr = VecMAXPY(out, m, g.data(), Q.data()); CHKERRV(ierr);

    if (nneg > 0) {
        neig_stats.lanc_events++;
        neig_stats.lanc_count += (long)nneg;
        if (warn_neg_eig) {
            PetscPrintf(PETSC_COMM_WORLD,
                "[schur/lanczos] floored %ld negative projected eigenvalue(s)\n", (long)nneg);
        }
    }

    for (size_t i = 0; i < Q.size(); i++) { ierr = VecDestroy(&Q[i]); CHKERRV(ierr); }
    ierr = VecDestroy(&w);       CHKERRV(ierr);
    ierr = VecDestroy(&Sq);      CHKERRV(ierr);
    ierr = VecDestroy(&tmp_nm3); CHKERRV(ierr);
}

void PolyStokes::report_neg_eig_stats(){
    // One-line-ish end-of-run summary of the negative-eigenvalue events collected in neig_stats
    // (always accumulated, even when the per-event [schur] warnings are silenced for speed).
    const NegEigStats& s = neig_stats;
    double frac       = (s.corr_stages > 0) ? (double)s.corr_events / (double)s.corr_stages : 0.0;
    double mean_worst = (s.corr_events > 0) ? s.corr_sum_worst / (double)s.corr_events        : 0.0;
    PetscPrintf(PETSC_COMM_WORLD,
        "[schur:stats] negative-eigenvalue summary: Schur correction ran on %ld stage(s); "
        "%ld had negatives (%.1f%%), %ld eigenvalue(s) corrected total; most-negative over the run "
        "= %.6e, mean per-event most-negative = %.6e.\n",
        s.corr_stages, s.corr_events, 100.0 * frac,
        s.corr_count, s.corr_worst, mean_worst);
    if (s.lanc_events > 0) {
        PetscPrintf(PETSC_COMM_WORLD,
            "[schur:stats] Lanczos noise path floored negatives on %ld stage(s), %ld total.\n",
            s.lanc_events, s.lanc_count);
    }
}
