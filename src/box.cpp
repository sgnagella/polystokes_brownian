#include "box.h"

#include <cmath>
#include <stdexcept>

void Box::configure(const std::vector<double>& lengths) {
    if (lengths.empty()) {
        enabled_ = false;
        return;
    }
    if (lengths.size() != 3) {
        throw std::invalid_argument(
            "Box::configure expects an empty list (no PBC) or exactly 3 lengths [Lx, Ly, Lz]");
    }
    for (int i = 0; i < 3; ++i) {
        if (!(lengths[i] > 0.0)) {
            throw std::invalid_argument("Box lengths must be positive");
        }
    }

    ortho_   = true;
    enabled_ = true;

    // Orthorhombic fast path.
    for (int i = 0; i < 3; ++i) {
        L_[i]    = lengths[i];
        Linv_[i] = 1.0 / lengths[i];
    }

    // Triclinic-ready representation: diagonal h-matrix (column-major) and its
    // inverse. Kept in sync so a future triclinic constructor only needs to fill
    // the off-diagonals and recompute hinv_.
    for (int i = 0; i < 9; ++i) { h_[i] = 0.0; hinv_[i] = 0.0; }
    for (int i = 0; i < 3; ++i) {
        h_[i + 3 * i]    = L_[i];      // column-major diagonal
        hinv_[i + 3 * i] = Linv_[i];
    }
}

void Box::minimum_image(double& dx, double& dy, double& dz) const {
    if (!enabled_) return;

    if (ortho_) {
        dx -= L_[0] * std::nearbyint(dx * Linv_[0]);
        dy -= L_[1] * std::nearbyint(dy * Linv_[1]);
        dz -= L_[2] * std::nearbyint(dz * Linv_[2]);
        return;
    }

    // General triclinic: wrap the separation in fractional coordinates.
    // s = hinv_ * r  (column-major matrix-vector product)
    const double r[3] = {dx, dy, dz};
    double s[3];
    for (int i = 0; i < 3; ++i) {
        s[i] = hinv_[i + 0] * r[0] + hinv_[i + 3] * r[1] + hinv_[i + 6] * r[2];
        s[i] -= std::nearbyint(s[i]);
    }
    // r = h_ * s
    dx = h_[0 + 0] * s[0] + h_[0 + 3] * s[1] + h_[0 + 6] * s[2];
    dy = h_[1 + 0] * s[0] + h_[1 + 3] * s[1] + h_[1 + 6] * s[2];
    dz = h_[2 + 0] * s[0] + h_[2 + 3] * s[1] + h_[2 + 6] * s[2];
}

void Box::wrap(double& x, double& y, double& z) const {
    if (!enabled_) return;

    if (ortho_) {
        x -= L_[0] * std::nearbyint(x * Linv_[0]);
        y -= L_[1] * std::nearbyint(y * Linv_[1]);
        z -= L_[2] * std::nearbyint(z * Linv_[2]);
        return;
    }

    // General triclinic: wrap fractional coordinates into [-0.5, 0.5).
    const double r[3] = {x, y, z};
    double s[3];
    for (int i = 0; i < 3; ++i) {
        s[i] = hinv_[i + 0] * r[0] + hinv_[i + 3] * r[1] + hinv_[i + 6] * r[2];
        s[i] -= std::nearbyint(s[i]);
    }
    x = h_[0 + 0] * s[0] + h_[0 + 3] * s[1] + h_[0 + 6] * s[2];
    y = h_[1 + 0] * s[0] + h_[1 + 3] * s[1] + h_[1 + 6] * s[2];
    z = h_[2 + 0] * s[0] + h_[2 + 3] * s[1] + h_[2 + 6] * s[2];
}
