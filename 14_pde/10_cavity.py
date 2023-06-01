import numpy as np
import matplotlib.pyplot as plt

debug_full = True

def print_array(name, array):
    print(name, end="")
    print(": [", end="")
    for value in array:
        print("{:.6g}".format(value), end=",")
    print("]")

def print_array2(name, array):
    print(name, end="")
    print(": [")
    for array1 in array:
        print("[", end="")
        for value in array1:
            print("{:.6g}".format(value), end=",")
        print("]")
    print("]")

nx = 41
ny = 41
nt = 500
nit = 50
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = .01
rho = 1
nu = .02

if debug_full:
    print("nx:", nx)
    print("ny:", ny)
    print("nt:", nt)
    print("nit:", nit)
    print("dx:", dx)
    print("dy:", dy)
    print("dt:", dt)
    print("rho:", rho)
    print("nu:", nu)
    print()

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

if debug_full:
    print_array("x", x)
    print_array("y", y)
    print()

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

if debug_full:
    print_array2("u", u)
    print_array2("v", v)
    print_array2("p", p)
    print_array2("b", b)

X, Y = np.meshgrid(x, y)

for n in range(nt):
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            b[j, i] = rho * (1 / dt *\
                    ((u[j, i+1] - u[j, i-1]) / (2 * dx) + (v[j+1, i] - v[j-1, i]) / (2 * dy)) -\
                    ((u[j, i+1] - u[j, i-1]) / (2 * dx))**2 - 2 * ((u[j+1, i] - u[j-1, i]) / (2 * dy) *\
                     (v[j, i+1] - v[j, i-1]) / (2 * dx)) - ((v[j+1, i] - v[j-1, i]) / (2 * dy))**2)

    if debug_full:
        print_array2("b", b)

    for it in range(nit):
        pn = p.copy()
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                p[j, i] = (dy**2 * (pn[j, i+1] + pn[j, i-1]) +\
                           dx**2 * (pn[j+1, i] + pn[j-1, i]) -\
                           b[j, i] * dx**2 * dy**2)\
                          / (2 * (dx**2 + dy**2))
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[:, 0] = p[:, 1]
        p[-1, :] = 0

    if debug_full:
        print_array2("p", p)

    un = u.copy()
    vn = v.copy()
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            u[j, i] = un[j, i] - un[j, i] * dt / dx * (un[j, i] - un[j, i - 1])\
                               - un[j, i] * dt / dy * (un[j, i] - un[j - 1, i])\
                               - dt / (2 * rho * dx) * (p[j, i+1] - p[j, i-1])\
                               + nu * dt / dx**2 * (un[j, i+1] - 2 * un[j, i] + un[j, i-1])\
                               + nu * dt / dy**2 * (un[j+1, i] - 2 * un[j, i] + un[j-1, i])
            v[j, i] = vn[j, i] - vn[j, i] * dt / dx * (vn[j, i] - vn[j, i - 1])\
                               - vn[j, i] * dt / dy * (vn[j, i] - vn[j - 1, i])\
                               - dt / (2 * rho * dx) * (p[j+1, i] - p[j-1, i])\
                               + nu * dt / dx**2 * (vn[j, i+1] - 2 * vn[j, i] + vn[j, i-1])\
                               + nu * dt / dy**2 * (vn[j+1, i] - 2 * vn[j, i] + vn[j-1, i])
    u[0, :]  = 0
    u[:, 0]  = 0
    u[:, -1] = 0
    u[-1, :] = 1
    v[0, :]  = 0
    v[-1, :] = 0
    v[:, 0]  = 0
    v[:, -1] = 0

    if debug_full:
        print_array2("u", u)
        print_array2("v", v)

    # plt.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
    # plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
    # plt.pause(.01)
    # plt.clf()

print_array2("u", u)
print_array2("v", v)

# 2次元 Navier-Stokes方程式
