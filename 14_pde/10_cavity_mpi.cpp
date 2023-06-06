/**
 * @file 10_cavity_mpi.cpp
 * @brief parallelizate in MPI
 * @author Masaki Otsubo
 * @date 2023-06-06
 *
 * @note module load gcc/12.2.0 intel-mpi/21.8.0
 * @note mpirun -np 4 ./a.out
 */

#include <cmath>
#include <iostream>
#include <mpi.h>
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
    MPI_Init(NULL, NULL);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin, end;

    // initialize
    if(debug_full && rank == 0) {
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

    // initialize x
    std::vector<double> x0(nx, 0);
    begin = rank * (nx / size);
    end = (rank + 1) * (nx / size);
    for(int i = begin; i < end; i++) {
        x0[i] = dx * i;
    }
    MPI_Allgather(&x0[begin], end - begin, MPI_DOUBLE, &x[0], end - begin,
                  MPI_DOUBLE, MPI_COMM_WORLD);
    // fraction
    for(int i = size * (nx / size); i < nx; i++) {
        x[i] = dx * i;
    }

    // initialize y
    std::vector<double> y0(ny, 0);
    begin = rank * (ny / size);
    end = (rank + 1) * (ny / size);
    for(int i = begin; i < end; i++) {
        y0[i] = dy * i;
    }
    MPI_Allgather(&y0[begin], end - begin, MPI_DOUBLE, &y[0], end - begin,
                  MPI_DOUBLE, MPI_COMM_WORLD);
    // fraction
    for(int i = size * (ny / size); i < ny; i++) {
        y[i] = dy * i;
    }

    if(debug_full && rank == 0) {
        print::array("x", x);
        print::array("y", y);
        print::null();
    }

    std::vector<std::vector<double>> u(nx, std::vector<double>(ny, 0));
    std::vector<std::vector<double>> v(nx, std::vector<double>(ny, 0));
    std::vector<std::vector<double>> p(nx, std::vector<double>(ny, 0));
    std::vector<std::vector<double>> b(nx, std::vector<double>(ny, 0));

    if(debug_full && rank == 0) {
        print::array2("u", u);
        print::array2("v", v);
        print::array2("p", p);
        print::array2("b", b);
    }

    // calculation
    for(int n = 0; n < nt; n++) {
        for(int j = 1; j < ny - 1; j++) {
            // calc b
            std::vector<double> bj0(ny, 0);
            begin = rank * ((ny - 2) / size) + 1;
            end = (rank + 1) * ((ny - 2) / size) + 1;
            for(int i = begin; i < end; i++) {
                double tmp1 = (u[j][i + 1] - u[j][i - 1]) / (2 * dx) +
                              (v[j + 1][i] - v[j - 1][i]) / (2 * dy);
                double tmp2 = (u[j][i + 1] - u[j][i - 1]) / (2 * dx);
                double tmp3 = (u[j + 1][i] - u[j - 1][i]) / (2 * dy) *
                              (v[j][i + 1] - v[j][i - 1]) / (2 * dx);
                double tmp4 = (v[j + 1][i] - v[j - 1][i]) / (2 * dy);
                bj0[i] = rho * (1 / dt * tmp1 - std::pow(tmp2, 2) - 2 * tmp3 -
                                std::pow(tmp4, 2));
            }
            MPI_Allgather(&bj0[begin], end - begin, MPI_DOUBLE, &b[j][1],
                          end - begin, MPI_DOUBLE, MPI_COMM_WORLD);
            // fraction
            for(int i = size * ((ny - 2) / size) + 1; i < (ny - 1); i++) {
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

        if(debug_full && rank == 0) {
            print::array2("b", b);
        }

        for(int it = 0; it < nit; it++) {
            std::vector<std::vector<double>> pn = p;
            for(int j = 1; j < ny - 1; j++) {
                // calc p
                std::vector<double> pj0(ny, 0);
                begin = rank * ((ny - 2) / size) + 1;
                end = (rank + 1) * ((ny - 2) / size) + 1;
                for(int i = begin; i < end; i++) {
                    double tmp1 =
                        std::pow(dy, 2) * (pn[j][i + 1] + pn[j][i - 1]);
                    double tmp2 =
                        std::pow(dx, 2) * (pn[j + 1][i] + pn[j - 1][i]);
                    double tmp3 = b[j][i] * std::pow(dx, 2) * std::pow(dy, 2);
                    double tmp4 = 2 * (std::pow(dx, 2) + std::pow(dy, 2));
                    pj0[i] = (tmp1 + tmp2 - tmp3) / tmp4;
                }
                MPI_Allgather(&pj0[begin], end - begin, MPI_DOUBLE, &p[j][1],
                              end - begin, MPI_DOUBLE, MPI_COMM_WORLD);
                // fraction
                for(int i = size * ((ny - 2) / size) + 1; i < (ny - 1); i++) {
                    double tmp1 =
                        std::pow(dy, 2) * (pn[j][i + 1] + pn[j][i - 1]);
                    double tmp2 =
                        std::pow(dx, 2) * (pn[j + 1][i] + pn[j - 1][i]);
                    double tmp3 = b[j][i] * std::pow(dx, 2) * std::pow(dy, 2);
                    double tmp4 = 2 * (std::pow(dx, 2) + std::pow(dy, 2));
                    p[j][i] = (tmp1 + tmp2 - tmp3) / tmp4;
                }
            }

            for(int j = 0; j < nx; j++) {
                p[j][ny - 1] = p[j][ny - 2];
                p[j][0] = p[j][1];
            }

            // set p
            std::vector<double> p00(ny, 0);
            std::vector<double> pnxm10(ny, 0);
            begin = rank * (ny / size);
            end = (rank + 1) * (ny / size);
            for(int i = begin; i < end; i++) {
                p00[i] = p[1][i];
                pnxm10[i] = 0;
            }
            MPI_Allgather(&p00[begin], end - begin, MPI_DOUBLE, &p[0][0],
                          end - begin, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Allgather(&pnxm10[begin], end - begin, MPI_DOUBLE,
                          &p[nx - 1][0], end - begin, MPI_DOUBLE,
                          MPI_COMM_WORLD);
            // fraction
            for(int i = size * (ny / size); i < ny; i++) {
                p[0][i] = p[1][i];
                p[nx - 1][i] = 0;
            }
        }

        if(debug_full && rank == 0) {
            print::array2("p", p);
        }

        std::vector<std::vector<double>> un = u;
        std::vector<std::vector<double>> vn = v;
        for(int j = 1; j < ny - 1; j++) {
            // calc u, j
            std::vector<double> uj0(ny, 0);
            std::vector<double> vj0(ny, 0);
            begin = rank * ((ny - 2) / size) + 1;
            end = (rank + 1) * ((ny - 2) / size) + 1;
            for(int i = begin; i < end; i++) {
                double tmp1 = un[j][i];
                double tmp2 = un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]);
                double tmp3 = un[j][i] * dt / dy * (un[j][i] - un[j - 1][i]);
                double tmp4 = dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1]);
                double tmp5 = nu * dt / std::pow(dx, 2) *
                              (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1]);
                double tmp6 = nu * dt / std::pow(dy, 2) *
                              (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]);
                uj0[i] = tmp1 - tmp2 - tmp3 - tmp4 + tmp5 + tmp6;

                tmp1 = vn[j][i];
                tmp2 = vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]);
                tmp3 = vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]);
                tmp4 = dt / (2 * rho * dx) * (p[j + 1][i] - p[j - 1][i]);
                tmp5 = nu * dt / std::pow(dx, 2) *
                       (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1]);
                tmp6 = nu * dt / std::pow(dy, 2) *
                       (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i]);
                vj0[i] = tmp1 - tmp2 - tmp3 - tmp4 + tmp5 + tmp6;
            }
            MPI_Allgather(&uj0[begin], end - begin, MPI_DOUBLE, &u[j][1],
                          end - begin, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Allgather(&vj0[begin], end - begin, MPI_DOUBLE, &v[j][1],
                          end - begin, MPI_DOUBLE, MPI_COMM_WORLD);
            // fraction
            for(int i = size * ((ny - 2) / size) + 1; i < (ny - 1); i++) {
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

        for(int j = 0; j < nx; j++) {
            u[j][0] = 0;
            u[j][ny - 1] = 0;
            v[j][0] = 0;
            v[j][ny - 1] = 0;
        }

        // set u, v
        std::vector<double> u00(ny, 0);
        std::vector<double> unxm10(ny, 0);
        std::vector<double> v00(ny, 0);
        std::vector<double> vnxm10(ny, 0);
        begin = rank * (ny / size);
        end = (rank + 1) * (ny / size);
        for(int i = begin; i < end; i++) {
            u00[i] = 0;
            unxm10[i] = 1;
            v00[i] = 0;
            vnxm10[i] = 0;
        }
        MPI_Allgather(&u00[begin], end - begin, MPI_DOUBLE, &u[0][0],
                      end - begin, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(&unxm10[begin], end - begin, MPI_DOUBLE, &u[nx - 1][0],
                      end - begin, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(&v00[begin], end - begin, MPI_DOUBLE, &v[0][0],
                      end - begin, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(&vnxm10[begin], end - begin, MPI_DOUBLE, &v[nx - 1][0],
                      end - begin, MPI_DOUBLE, MPI_COMM_WORLD);
        // fraction
        for(int i = size * (ny / size); i < ny; i++) {
            u[0][i] = 0;
            u[nx - 1][i] = 1;
            v[0][i] = 0;
            v[nx - 1][i] = 0;
        }

        if(debug_full && rank == 0) {
            print::array2("u", u);
            print::array2("v", v);
        }
    }

    if(rank == 0) {
        print::array2("u", u);
        print::array2("v", v);
    }

    MPI_Finalize();
    return 0;
}
