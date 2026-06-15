#include <iostream>
#include "arrays.h"
#include <cmath>

using namespace arrays;

std::vector<double> quaternion_multiply(const std::vector<double>& q1, const std::vector<double>& q2){
    // Quaternion multiplication
    double w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
    double w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];

    return {
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  // w
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  // x
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  // y
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2   // z
    };
}


std::vector<double> normalize_quat(const std::vector<double>& q){
    double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    return {q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm};
}

std::vector<double> quaternion_deriv(const std::vector<double>& q, const std::vector<double>& omega){
    // Angular velocity as a quaternion
    std::vector<double> omega_q = {0.0, omega[0], omega[1], omega[2]};
    std::vector<double> dqdt = quaternion_multiply(q, omega_q);

    // Scale by 0.5
    // for (auto& val : dqdt) {
    //     val *= 0.5;
    // }

    return dqdt;
}

// Time integrate quaternion
std::vector<double> integrate_quat(const std::vector<double>& q, const std::vector<double>& omega, double dt) {
    // Compute quaternion derivative
    std::vector<double> dqdt = quaternion_deriv(q, omega);

    // Update quaternion using Euler integration
    std::vector<double> q_next = {
        q[0] + 0.5 * dqdt[0] * dt,
        q[1] + 0.5 * dqdt[1] * dt,
        q[2] + 0.5 * dqdt[2] * dt,
        q[3] + 0.5 * dqdt[3] * dt
    };

    // Normalize the quaternion to maintain unit length
    return normalize_quat(q_next);
}