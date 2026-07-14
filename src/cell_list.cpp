// Monomer cell list for the mm_HI == false case.
//
// When monomer-monomer HI is off, monomer-monomer (AA) pairs are needed ONLY for the
// excluded-volume (WCA) force (mono_ev). Rather than the O(Nm^2) all-pairs scan in
// check_dist()/pair_interaction(), we bin the Nm monomers into a linked cell grid and
// visit only near-neighbor cells, giving an O(Nm) neighbor search (and avoiding the
// O(Nm^2) pd/id_AA arrays entirely -- those are only allocated when mm_HI == true).
//
// build_monomer_cell_list() rebuilds the grid each step (on the wrapped positions left
// by step()); monomer_wca() walks it and adds the WCA forces. The colloid-involving
// (AB/BB) pairs are still handled by check_dist()/pair_interaction().

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include "arrays.h"
#include "Stokes.h"
#include "pair_interaction.h"   // wca_force_mag

using namespace arrays;

void PolyStokes::build_monomer_cell_list(){
    const int   Nm    = pinfo.Nm;
    const int   ndim  = consts.ndim;
    const double rcell = dataStruct.rverls[0];   // AA Verlet cutoff (>= WCA rcut[0])

    // --- Cell geometry -------------------------------------------------------------
    // Periodic fast path needs >= 3 cells per axis so the 13-offset half stencil maps
    // to distinct cells under wrap. Otherwise fall back to a single cell (O(Nm^2) but
    // correct; minimum_image still supplies the right separations).
    if( box.active() ){
        int nc_min = std::numeric_limits<int>::max();
        for(int d = 0; d < 3; d++){
            double L = box.length(d);
            cl_nc[d] = (int)std::floor(L / rcell);
            nc_min   = std::min(nc_min, cl_nc[d]);
        }
        if( nc_min >= 3 ){
            for(int d = 0; d < 3; d++){
                double L = box.length(d);
                cl_size[d] = L / cl_nc[d];
                cl_org[d]  = -0.5 * L;          // wrapped coords live in [-L/2, L/2)
            }
            cl_periodic = true;
        }
        else{
            for(int d = 0; d < 3; d++){
                double L = box.length(d);
                cl_nc[d]   = 1;
                cl_size[d] = (L > 0.0) ? L : 1.0;
                cl_org[d]  = -0.5 * L;
            }
            cl_periodic = false;
        }
    }
    else{
        // Unbounded: bin over the current monomer bounding box, non-periodic.
        double lo[3] = { 1e300,  1e300,  1e300};
        double hi[3] = {-1e300, -1e300, -1e300};
        for(int i = 0; i < Nm; i++){
            const double* xi = &x[ndim * i];
            for(int d = 0; d < 3; d++){
                lo[d] = std::min(lo[d], xi[d]);
                hi[d] = std::max(hi[d], xi[d]);
            }
        }
        if( Nm == 0 ){ for(int d = 0; d < 3; d++){ lo[d] = 0.0; hi[d] = 0.0; } }
        for(int d = 0; d < 3; d++){
            double ext = hi[d] - lo[d];
            cl_nc[d]   = std::max(1, (int)std::floor(ext / rcell));
            cl_size[d] = (ext > 0.0) ? ext / cl_nc[d] : rcell;
            cl_org[d]  = lo[d];
        }
        cl_periodic = false;
    }

    const long ncells = (long)cl_nc[0] * cl_nc[1] * cl_nc[2];

    // --- Bin monomers (prepend into head-of-chain) ---------------------------------
    cl_head.assign(ncells, -1);
    cl_next.assign(Nm, -1);
    for(int i = 0; i < Nm; i++){
        const double* xi = &x[ndim * i];
        int c[3];
        for(int d = 0; d < 3; d++){
            int ci = (int)std::floor((xi[d] - cl_org[d]) / cl_size[d]);
            if( ci < 0 )           ci = 0;
            if( ci >= cl_nc[d] )   ci = cl_nc[d] - 1;   // guard FP edge at the top face
            c[d] = ci;
        }
        long cell = (long)c[0] + (long)cl_nc[0] * (c[1] + (long)cl_nc[1] * c[2]);
        cl_next[i]    = cl_head[cell];
        cl_head[cell] = i;
    }
}

void PolyStokes::monomer_wca(PetscScalar* fext){
    const int    Nm   = pinfo.Nm;
    const int    ndim = consts.ndim;
    const double sig  = dataStruct.sigmas[0];
    const double rcut = dataStruct.rcuts[0];
    const double rcut2 = rcut * rcut;
    const double eps  = pinfo.epsilon;

    // 13 forward half-stencil neighbor offsets; their negatives cover the other 13, so
    // together with the home cell (self-pairs, i after j in the list) each AA pair is
    // visited exactly once.
    static const int OFF[13][3] = {
        { 1, 0, 0}, {-1, 1, 0}, { 0, 1, 0}, { 1, 1, 0},
        {-1,-1, 1}, { 0,-1, 1}, { 1,-1, 1}, {-1, 0, 1},
        { 0, 0, 1}, { 1, 0, 1}, {-1, 1, 1}, { 0, 1, 1}, { 1, 1, 1}
    };

    // Local lambda: apply WCA between monomers i and j (i != j, counted once).
    auto interact = [&](int i, int j){
        const double* xi = &x[ndim * i];
        const double* xj = &x[ndim * j];
        double dx = xi[0] - xj[0];
        double dy = xi[1] - xj[1];
        double dz = xi[2] - xj[2];
        box.minimum_image(dx, dy, dz);              // no-op when the box is inactive
        double dr2 = dx*dx + dy*dy + dz*dz;
        if( dr2 >= rcut2 || dr2 == 0.0 ) return;
        double dr     = std::sqrt(dr2);
        double dr_inv = 1.0 / dr;
        double fmag   = wca_force_mag(dr, dr_inv, sig, rcut, eps);
        if( fmag == 0.0 ) return;
        // dx = x_i - x_j points from j to i; positive fmag pushes i away from j. fext
        // stores the NEGATIVE of the physical force (cf. pair_interaction / trapping).
        double fx = fmag * dx * dr_inv;
        double fy = fmag * dy * dr_inv;
        double fz = fmag * dz * dr_inv;
        int i3 = ndim * i, j3 = ndim * j;
        fext[i3]   -= fx; fext[i3+1] -= fy; fext[i3+2] -= fz;
        fext[j3]   += fx; fext[j3+1] += fy; fext[j3+2] += fz;
    };

    for(int cz = 0; cz < cl_nc[2]; cz++){
    for(int cy = 0; cy < cl_nc[1]; cy++){
    for(int cx = 0; cx < cl_nc[0]; cx++){
        long home = (long)cx + (long)cl_nc[0] * (cy + (long)cl_nc[1] * cz);
        int hi = cl_head[home];
        if( hi < 0 ) continue;

        // Self-cell: each unordered pair once (j walks the chain after i).
        for(int i = hi; i != -1; i = cl_next[i])
            for(int j = cl_next[i]; j != -1; j = cl_next[j])
                interact(i, j);

        // Forward neighbor cells.
        for(int o = 0; o < 13; o++){
            int nx = cx + OFF[o][0];
            int ny = cy + OFF[o][1];
            int nz = cz + OFF[o][2];
            if( cl_periodic ){
                nx = (nx % cl_nc[0] + cl_nc[0]) % cl_nc[0];
                ny = (ny % cl_nc[1] + cl_nc[1]) % cl_nc[1];
                nz = (nz % cl_nc[2] + cl_nc[2]) % cl_nc[2];
            }
            else{
                if( nx < 0 || nx >= cl_nc[0] ) continue;
                if( ny < 0 || ny >= cl_nc[1] ) continue;
                if( nz < 0 || nz >= cl_nc[2] ) continue;
            }
            long neigh = (long)nx + (long)cl_nc[0] * (ny + (long)cl_nc[1] * nz);
            for(int i = hi; i != -1; i = cl_next[i])
                for(int j = cl_head[neigh]; j != -1; j = cl_next[j])
                    interact(i, j);
        }
    }}}
}
