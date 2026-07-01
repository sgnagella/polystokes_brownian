// Periodic simulation box: minimum-image convention and position wrapping.
//
// Scope: minimum-image + wrapping only. This does NOT implement periodic
// hydrodynamics (Ewald summation of the mobility); when active the far-field
// mobility uses nearest-image separations, a standard documented approximation.
//
// Geometry: the public API takes three box lengths (orthorhombic). Internally a
// general triclinic h-matrix (+ inverse) is stored so tilt factors can be added
// later without reworking callers; only the orthorhombic path is constructible now.

#ifndef BOX_H
#define BOX_H

#include <vector>

class Box {
public:
    Box() = default;

    // Configure the box. An empty vector disables periodicity; a size-3 vector
    // {Lx, Ly, Lz} enables an orthorhombic periodic cell. Throws on other sizes
    // or non-positive lengths.
    void configure(const std::vector<double>& lengths);

    // True when periodic boundary conditions are enabled.
    bool active() const { return enabled_; }

    // Replace a separation vector with its minimum image (in place).
    // No-op when the box is inactive.
    void minimum_image(double& dx, double& dy, double& dz) const;

    // Wrap a position into the primary cell centered at the origin, i.e. each
    // coordinate into [-L/2, L/2) (in place). No-op when the box is inactive.
    void wrap(double& x, double& y, double& z) const;

private:
    bool   enabled_ = false;
    bool   ortho_   = true;
    double L_[3]    = {0.0, 0.0, 0.0};   // orthorhombic fast-path lengths
    double Linv_[3] = {0.0, 0.0, 0.0};   // 1 / L_
    double h_[9]    = {0.0};             // triclinic-ready 3x3, column-major (lattice vectors)
    double hinv_[9] = {0.0};             // inverse of h_
};

#endif // BOX_H
