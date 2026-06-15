#ifndef QUAT_H
#define QUAT_H
#include <vector>
std::vector<double> integrate_quat(const std::vector<double>& q, const std::vector<double>& omega, double dt);
#endif