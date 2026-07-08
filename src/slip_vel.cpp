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
    // The grand mobility lives in the top-left [0, nm3nc11) block of A, so the colloid
    // sub-blocks are M^cm = A[nm3:nm3nc11, 0:nm3] and M^cc = A[nm3:nm3nc11, nm3:nm3nc11].
    Mat Mcm, Bcm, Mcc, Ccc;
    ierr = MatDenseGetSubMatrix(A, nm3, nm3nc11, 0, nm3, &Mcm); CHKERRV(ierr);

    // Cross term: uc = sqrt(beta) * (M^cm xi_m).
    Vec uc;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nc11, &uc); CHKERRV(ierr);
    ierr = MatMult(Mcm, xi_m, uc); CHKERRV(ierr);
    ierr = VecScale(uc, PetscSqrtReal(beta)); CHKERRV(ierr);

    ierr = MatDuplicate(Mcm, MAT_COPY_VALUES, &Bcm); CHKERRV(ierr);
    ierr = MatDenseRestoreSubMatrix(A, &Mcm); CHKERRV(ierr);

    ierr = MatDenseGetSubMatrix(A, nm3, nm3nc11, nm3, nm3nc11, &Mcc); CHKERRV(ierr);
    ierr = MatDuplicate(Mcc, MAT_COPY_VALUES, &Ccc); CHKERRV(ierr);
    ierr = MatDenseRestoreSubMatrix(A, &Mcc); CHKERRV(ierr);

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
        schur_sqrt_lanczos(Bcm, Ccc, beta, xi_c, uc, /*kmax=*/40);
    }
    else {
        // Dense path: S = Ccc - beta Bcm Bcm^T into the persistent workspace Smat,
        //   S = Q diag(lambda) Q^T,  S^{1/2} xi_c = Q diag(sqrt(max(lambda,0))) Q^T xi_c.
        Mat P;
        ierr = MatMatTransposeMult(Bcm, Bcm, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &P); CHKERRV(ierr);
        ierr = MatScale(P, -beta); CHKERRV(ierr);
        ierr = MatAXPY(P, 1.0, Ccc, SAME_NONZERO_PATTERN); CHKERRV(ierr);
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

        PetscInt nneg = 0;
        const PetscScalar *xc;
        ierr = VecGetArrayRead(xi_c, &xc); CHKERRV(ierr);

        // y = diag(sqrt(max(lambda,0))) * (Q^T xi_c)
        std::vector<PetscScalar> y(nc11, 0.0);
        for (PetscInt j = 0; j < nc11; j++) {
            PetscScalar yj = 0.0;
            for (PetscInt i = 0; i < nc11; i++) yj += Sarr[i + j*lda] * xc[i];
            PetscReal lam = PetscRealPart(w[j]);
            if (lam < 0.0) { nneg++; lam = 0.0; }
            y[j] = PetscSqrtReal(lam) * yj;
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

        ierr = MatDenseRestoreArray(Smat, &Sarr); CHKERRV(ierr);

        if (nneg > 0) {
            PetscPrintf(PETSC_COMM_WORLD,
                "[schur] floored %D negative Schur eigenvalue(s) "
                "(truncated far-field mobility not SPD)\n", (PetscInt)nneg);
        }
    }

    ierr = MatDestroy(&Bcm); CHKERRV(ierr);
    ierr = MatDestroy(&Ccc); CHKERRV(ierr);

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
        PetscPrintf(PETSC_COMM_WORLD,
            "[schur/lanczos] floored %D negative projected eigenvalue(s)\n", (PetscInt)nneg);
    }

    for (size_t i = 0; i < Q.size(); i++) { ierr = VecDestroy(&Q[i]); CHKERRV(ierr); }
    ierr = VecDestroy(&w);       CHKERRV(ierr);
    ierr = VecDestroy(&Sq);      CHKERRV(ierr);
    ierr = VecDestroy(&tmp_nm3); CHKERRV(ierr);
}
