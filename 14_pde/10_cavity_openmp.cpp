/**
 * @file 10_cavity_openmp.cpp
 * @brief parallelizate in OpenMP
 * @author Masaki Otsubo
 * @date 2023-06-01
 *
 * @note module load gcc/12.2.0
 */

#include <cmath>
#include <iostream>
#include <string.h>
#include <vector>

constexpr int nx = 41;
constexpr int ny = 41;
constexpr int nt = 500;
constexpr int nit = 50;
constexpr double dx = (double)2 / (nx - 1);
constexpr double dy = (double)2 / (ny - 1);
constexpr double dt = 0.01;
constexpr int rho = 1;
constexpr double nu = 0.02;

constexpr bool debug_full = true;

class print {
  public:
    static void null() { std::cout << std::endl; }
    static void value(std::string name, double value) {
        std::cout << name << ": " << value << std::endl;
    }
    static void array(std::string name, std::vector<double> array) {
        std::cout << name << ": [";
        for(double value : array) {
            std::cout << value << ",";
        }
        std::cout << "]" << std::endl;
    }
    static void array2(std::string name,
                       std::vector<std::vector<double>> array2) {
        std::cout << name << ": [" << std::endl;
        for(std::vector<double> array1 : array2) {
            std::cout << "[";
            for(double value : array1) {
                std::cout << value << ",";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }
};

int main() {
    // initialize
    if(debug_full) {
        print::value("nx", nx);
        print::value("ny", ny);
        print::value("nt", nt);
        print::value("nit", nit);
        print::value("dx", dx);
        print::value("dy", dy);
        print::value("dt", dt);
        print::value("rho", rho);
        print::value("nu", nu);
        print::null();
    }

    std::vector<double> x(nx, 0);
    std::vector<double> y(ny, 0);
#pragma omp parallel for
    for(int i = 0; i < nx; i++) {
        x[i] = dx * i;
    }
#pragma omp parallel for
    for(int i = 0; i < ny; i++) {
        y[i] = dy * i;
    }

    if(debug_full) {
        print::array("x", x);
        print::array("y", y);
        print::null();
    }

    std::vector<std::vector<double>> u(nx, std::vector<double>(ny, 0));
    std::vector<std::vector<double>> v(nx, std::vector<double>(ny, 0));
    std::vector<std::vector<double>> p(nx, std::vector<double>(ny, 0));
    std::vector<std::vector<double>> b(nx, std::vector<double>(ny, 0));

    if(debug_full) {
        print::array2("u", u);
        print::array2("v", v);
        print::array2("p", p);
        print::array2("b", b);
    }

    // calculation
    for(int n = 0; n < nt; n++) {
#pragma omp parallel
        for(int j = 1; j < ny - 1; j++) {
#pragma omp for
            for(int i = 1; i < nx - 1; i++) {
                double tmp1 = (u[j][i + 1] - u[j][i - 1]) / (2 * dx) +
                              (v[j + 1][i] - v[j - 1][i]) / (2 * dy);
                double tmp2 = (u[j][i + 1] - u[j][i - 1]) / (2 * dx);
                double tmp3 = (u[j + 1][i] - u[j - 1][i]) / (2 * dy) *
                              (v[j][i + 1] - v[j][i - 1]) / (2 * dx);
                double tmp4 = (v[j + 1][i] - v[j - 1][i]) / (2 * dy);
                b[j][i] = rho * (1 / dt * tmp1 - std::pow(tmp2, 2) - 2 * tmp3 -
                                 std::pow(tmp4, 2));
            }
        }

        if(debug_full) {
            print::array2("b", b);
        }

        for(int it = 0; it < nit; it++) {
            std::vector<std::vector<double>> pn = p;
#pragma omp parallel for
            for(int j = 1; j < ny - 1; j++) {
                for(int i = 1; i < nx - 1; i++) {
                    double tmp1 =
                        std::pow(dy, 2) * (pn[j][i + 1] + pn[j][i - 1]);
                    double tmp2 =
                        std::pow(dx, 2) * (pn[j + 1][i] + pn[j - 1][i]);
                    double tmp3 = b[j][i] * std::pow(dx, 2) * std::pow(dy, 2);
                    double tmp4 = 2 * (std::pow(dx, 2) + std::pow(dy, 2));
                    p[j][i] = (tmp1 + tmp2 - tmp3) / tmp4;
                }
            }
#pragma omp parallel for
            for(int j = 0; j < nx; j++) {
                p[j][ny - 1] = p[j][ny - 2];
                p[j][0] = p[j][1];
            }
#pragma omp parallel for
            for(int i = 0; i < ny; i++) {
                p[0][i] = p[1][i];
                p[nx - 1][i] = 0;
            }
        }

        if(debug_full) {
            print::array2("p", p);
        }

        std::vector<std::vector<double>> un = u;
        std::vector<std::vector<double>> vn = v;
#pragma omp parallel
        for(int j = 1; j < ny - 1; j++) {
#pragma omp for
            for(int i = 1; i < nx - 1; i++) {
                double tmp1 = un[j][i];
                double tmp2 = un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]);
                double tmp3 = un[j][i] * dt / dy * (un[j][i] - un[j - 1][i]);
                double tmp4 = dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1]);
                double tmp5 = nu * dt / std::pow(dx, 2) *
                              (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1]);
                double tmp6 = nu * dt / std::pow(dy, 2) *
                              (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]);
                u[j][i] = tmp1 - tmp2 - tmp3 - tmp4 + tmp5 + tmp6;

                tmp1 = vn[j][i];
                tmp2 = vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]);
                tmp3 = vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]);
                tmp4 = dt / (2 * rho * dx) * (p[j + 1][i] - p[j - 1][i]);
                tmp5 = nu * dt / std::pow(dx, 2) *
                       (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1]);
                tmp6 = nu * dt / std::pow(dy, 2) *
                       (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i]);
                v[j][i] = tmp1 - tmp2 - tmp3 - tmp4 + tmp5 + tmp6;
            }
        }
#pragma omp parallel for
        for(int j = 0; j < nx; j++) {
            u[j][0] = 0;
            u[j][ny - 1] = 0;
            v[j][0] = 0;
            v[j][ny - 1] = 0;
        }
#pragma omp parallel for
        for(int i = 0; i < ny; i++) {
            u[0][i] = 0;
            u[nx - 1][i] = 1;
            v[0][i] = 0;
            v[nx - 1][i] = 0;
        }

        if(debug_full) {
            print::array2("u", u);
            print::array2("v", v);
        }
    }
    print::array2("u", u);
    print::array2("v", v);

    return 0;
}
