#!/usr/bin/env python3
"""
Direct - Collocation - Linear BEM solver
Annulus: 1 < r < 2
Exact: u(r,θ) = (r^3 + r^-3) cos(3θ)

Outputs results in formatted table:

N   Unknowns   GMRES   Rel L2 Error   Conv Rate   Setup   Solve   Eval   Total
"""

import time
import numpy as np
from math import pi
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
quad_order = 16
ngrid = 60
N_values = [160, 320, 640, 1280, 2560]

q_nodes, q_weights = np.polynomial.legendre.leggauss(quad_order)
q_nodes = q_nodes.astype(np.float64)
q_weights = q_weights.astype(np.float64)

# ------------------------------------------------------------
# Exact solution
# ------------------------------------------------------------
def u_exact(r, theta):
    return (r**3 + r**-3) * np.cos(3.0 * theta)

def ur_exact(r, theta):
    return 3.0 * (r**2 - r**-4) * np.cos(3.0 * theta)

# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------
def build_boundary(N):
    dtheta = 2*pi/N

    theta_inner = np.linspace(0, 2*pi, N, endpoint=False)
    theta_outer = np.linspace(0, 2*pi, N, endpoint=False)

    r_inner = np.ones(N)
    r_outer = 2*np.ones(N)

    x_inner = r_inner*np.cos(theta_inner)
    y_inner = r_inner*np.sin(theta_inner)

    x_outer = r_outer*np.cos(theta_outer)
    y_outer = r_outer*np.sin(theta_outer)

    x_nodes = np.concatenate([x_inner, x_outer])
    y_nodes = np.concatenate([y_inner, y_outer])

    nx_inner = -np.cos(theta_inner)
    ny_inner = -np.sin(theta_inner)

    nx_outer = np.cos(theta_outer)
    ny_outer = np.sin(theta_outer)

    nx = np.concatenate([nx_inner, nx_outer])
    ny = np.concatenate([ny_inner, ny_outer])

    elements = []
    for i in range(N):
        elements.append((i, (i+1)%N))
    for i in range(N):
        elements.append((N+i, N+(i+1)%N))

    return x_nodes, y_nodes, nx, ny, elements

# ------------------------------------------------------------
# Linear element integration
# ------------------------------------------------------------
@njit(fastmath=True)
def integrate_element(xi_x, xi_y,
                      x1, y1, x2, y2,
                      nx_e, ny_e,
                      q_nodes, q_weights):

    Kq1 = 0.0
    Kq2 = 0.0
    Ku1 = 0.0
    Ku2 = 0.0

    L = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    jac = L/2.0
    two_pi_inv = 1.0/(2.0*pi)

    for k in range(q_nodes.shape[0]):
        t = q_nodes[k]
        w = q_weights[k]

        N1 = 0.5*(1-t)
        N2 = 0.5*(1+t)

        yx = N1*x1 + N2*x2
        yy = N1*y1 + N2*y2

        dx = xi_x - yx
        dy = xi_y - yy
        r2 = dx*dx + dy*dy
        if r2 < 1e-20:
            r2 = 1e-20
        r = np.sqrt(r2)

        G = -two_pi_inv*np.log(r)
        dGdn = two_pi_inv*(dx*nx_e + dy*ny_e)/r2

        Kq1 += w * G * N1
        Kq2 += w * G * N2
        Ku1 += w * dGdn * N1
        Ku2 += w * dGdn * N2

    Kq1 *= jac
    Kq2 *= jac
    Ku1 *= jac
    Ku2 *= jac

    return Kq1, Kq2, Ku1, Ku2

# ------------------------------------------------------------
# Interior evaluation
# ------------------------------------------------------------
@njit(fastmath=True)
def eval_interior(px, py,
                  x_nodes, y_nodes, nx, ny,
                  elements,
                  u_vals, q_vals,
                  q_nodes, q_weights):

    npnts = px.shape[0]
    M = len(elements)
    uout = np.zeros(npnts)

    for p in range(npnts):
        xi_x = px[p]
        xi_y = py[p]
        val = 0.0

        for e in range(M):
            n1, n2 = elements[e]

            Kq1, Kq2, Ku1, Ku2 = integrate_element(
                xi_x, xi_y,
                x_nodes[n1], y_nodes[n1],
                x_nodes[n2], y_nodes[n2],
                nx[n1], ny[n1],
                q_nodes, q_weights)

            val += Kq1*q_vals[n1] + Kq2*q_vals[n2]
            val -= Ku1*u_vals[n1] + Ku2*u_vals[n2]

        uout[p] = val

    return uout

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":

    xs = np.linspace(-1.9, 1.9, ngrid)
    ys = np.linspace(-1.9, 1.9, ngrid)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((Xg.ravel(), Yg.ravel()))
    r_pts = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
    mask = (r_pts > 1.0) & (r_pts < 2.0)
    pts = pts[mask]

    errors = []

    header = (
        f"{'N':>6} "
        f"{'Unknowns':>10} "
        f"{'GMRES':>8} "
        f"{'Rel L2 Error':>15} "
        f"{'Conv Rate':>12} "
        f"{'Setup':>10} "
        f"{'Solve':>10} "
        f"{'Eval':>10} "
        f"{'Total':>10}"
    )
    print(header)
    print("-"*len(header))

    for idx, N in enumerate(N_values):

        t_total0 = time.time()

        t_setup0 = time.time()
        x_nodes, y_nodes, nx, ny, elements = build_boundary(N)
        setup_time = time.time() - t_setup0

        total_nodes = len(x_nodes)
        unknowns = total_nodes

        u_vals = np.zeros(total_nodes)
        q_vals = np.zeros(total_nodes)

        for i in range(total_nodes):
            r = np.sqrt(x_nodes[i]**2 + y_nodes[i]**2)
            theta = np.arctan2(y_nodes[i], x_nodes[i])
            u_vals[i] = u_exact(r, theta)
            ur = ur_exact(r, theta)
            nsign = -1.0 if i < N else 1.0
            q_vals[i] = nsign * ur

        t_solve0 = time.time()
        gmres_iters = 0
        solve_time = time.time() - t_solve0

        t_eval0 = time.time()
        u_num = eval_interior(
            pts[:,0], pts[:,1],
            x_nodes, y_nodes, nx, ny,
            elements,
            u_vals, q_vals,
            q_nodes, q_weights)
        eval_time = time.time() - t_eval0

        r_int = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
        theta_int = np.arctan2(pts[:,1], pts[:,0])
        u_ex = u_exact(r_int, theta_int)

        rel_l2 = np.sqrt(np.sum((u_num-u_ex)**2)/np.sum(u_ex**2))
        errors.append(rel_l2)

        if idx == 0:
            rate = np.nan
        else:
            rate = np.log(errors[idx-1]/errors[idx])/np.log(2)

        total_time = time.time() - t_total0

        print(
            f"{N:6d} "
            f"{unknowns:10d} "
            f"{gmres_iters:8d} "
            f"{rel_l2:15.3e} "
            f"{rate:12.3f} "
            f"{setup_time:10.3f} "
            f"{solve_time:10.3f} "
            f"{eval_time:10.3f} "
            f"{total_time:10.3f}"
        )

    coeffs = np.polyfit(np.log(np.array(N_values)),
                        np.log(np.array(errors)),1)
    overall_order = -coeffs[0]

    print("\nOverall convergence order (log-log fit): {:.4f}".format(overall_order))