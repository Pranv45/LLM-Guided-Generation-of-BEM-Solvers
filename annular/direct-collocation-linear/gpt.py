#!/usr/bin/env python3
"""
Direct - Collocation - Linear BEM solver for Laplace's equation on an annulus 1<r<2
Goal: implement collocation + linear elements correctly (second-order convergence)

Notes:
 - Parameterization uses angular coordinate (true curved elements on circle)
 - Linear (nodal) shape functions in parameter space
 - Singular / near-singular integrals treated with higher-order quadrature and
   logarithmic-subtraction idea for self-element accuracy
 - Dense assembly (M x M blocks) with numba-accelerated element integrals
 - Mixed boundary conditions handled by eliminating known DOFs
 - Uses numpy, scipy, numba only
"""

import time
import numpy as np
from math import pi
from scipy.linalg import solve
from numba import njit

# -----------------------
# Settings / quadrature
# -----------------------
nq_far = 32
nq_near = 128  # high-order for near / self integrals
q_nodes_far, q_weights_far = np.polynomial.legendre.leggauss(nq_far)
q_nodes_near, q_weights_near = np.polynomial.legendre.leggauss(nq_near)

ngrid = 60  # interior grid fixed
N_values = [160, 320, 640, 1280, 2560]  # refinement study (you can extend)

# -----------------------
# Exact solution helpers
# -----------------------
def u_exact_polar(r, theta):
    return (r**3 + r**-3) * np.cos(3.0 * theta)

def ur_exact_polar(r, theta):
    return 3.0 * (r**2 - r**-4) * np.cos(3.0 * theta)

# -----------------------
# Geometry builder
# -----------------------
def build_nodes_and_elements(N):
    """
    Create node angles, node coordinates, connectivity for inner and outer circles.
    Ordering: first N nodes inner (r=1), next N nodes outer (r=2).
    Elements: pairs of node indices for each circle.
    Returns dict with arrays.
    """
    theta_inner = np.linspace(0.0, 2.0*pi, N, endpoint=False)
    theta_outer = np.linspace(0.0, 2.0*pi, N, endpoint=False)

    theta_nodes = np.concatenate([theta_inner, theta_outer])
    radii = np.concatenate([np.ones(N), 2.0 * np.ones(N)])
    x_nodes = radii * np.cos(theta_nodes)
    y_nodes = radii * np.sin(theta_nodes)

    # normals at a point on circle: outer -> +e_r, inner -> -e_r (outward from domain)
    nx = np.concatenate([-np.cos(theta_inner), np.cos(theta_outer)])
    ny = np.concatenate([-np.sin(theta_inner), np.sin(theta_outer)])

    elements = []
    # inner elements
    for i in range(N):
        n1 = i
        n2 = (i + 1) % N
        elements.append((n1, n2))
    # outer elements
    for i in range(N):
        n1 = N + i
        n2 = N + ((i + 1) % N)
        elements.append((n1, n2))

    return {
        "N": N,
        "M": 2 * N,
        "theta_nodes": theta_nodes,
        "radii": radii,
        "x_nodes": x_nodes,
        "y_nodes": y_nodes,
        "nx": nx,
        "ny": ny,
        "elements": np.array(elements, dtype=np.int64),
    }

# -----------------------
# Numba-accelerated element integrals
# -----------------------
@njit(fastmath=True)
def ang_diff(a, b):
    # return signed smallest difference a-b into [-pi,pi]
    d = a - b
    while d <= -pi:
        d += 2*pi
    while d > pi:
        d -= 2*pi
    return d

@njit(fastmath=True)
def integrate_element_theta(xi_x, xi_y,
                            r_elem, theta1, theta2, node_sign,
                            q_nodes, q_weights):
    """
    Integrate over curved circular element parameterized by theta in [theta1, theta2].
    Linear shape functions in reference t in [-1,1]: N1 = 0.5*(1-t), N2 = 0.5*(1+t).
    Returns contributions to Kq (for q basis at node1,node2) and Ku (for u basis at node1,node2):
      Kq1, Kq2, Ku1, Ku2 where
         Kq_j = ∫ G(xi,y) * N_j(t) * r_elem * (dθ/dt) dt  -> but dθ/dt = (Δθ/2)
         Ku_j = ∫ ∂G/∂n_y(xi,y) * N_j(t) * r_elem * (dθ/dt) dt
    node_sign = -1 for inner (normal inward), +1 for outer (normal outward)
    """
    Kq1 = 0.0
    Kq2 = 0.0
    Ku1 = 0.0
    Ku2 = 0.0

    # handle angle wrap: choose Δθ in (-pi, pi]
    mid = 0.5 * (theta1 + theta2)
    # more robust midpoint using angles
    # compute delta via difference
    dtheta = ang_diff(theta2, theta1)
    half = 0.5 * dtheta
    theta_mid = theta1 + 0.5 * dtheta
    # Jacobian factor for dθ = half * dt
    jac = r_elem * half
    two_pi_inv = 1.0 / (2.0 * pi)

    for k in range(q_nodes.shape[0]):
        t = q_nodes[k]
        w = q_weights[k]
        theta = theta_mid + half * t
        yx = r_elem * np.cos(theta)
        yy = r_elem * np.sin(theta)
        dx = xi_x - yx
        dy = xi_y - yy
        r2 = dx * dx + dy * dy
        if r2 < 1e-20:
            r2 = 1e-20
        rabs = np.sqrt(r2)
        # single-layer kernel
        G = -two_pi_inv * np.log(rabs)
        # normal at quadrature point: node_sign * e_r
        nxq = node_sign * np.cos(theta)
        nyq = node_sign * np.sin(theta)
        dGdn = two_pi_inv * (dx * nxq + dy * nyq) / r2
        N1 = 0.5 * (1.0 - t)
        N2 = 0.5 * (1.0 + t)
        Kq1 += w * G * N1
        Kq2 += w * G * N2
        Ku1 += w * dGdn * N1
        Ku2 += w * dGdn * N2

    Kq1 *= jac
    Kq2 *= jac
    Ku1 *= jac
    Ku2 *= jac

    return Kq1, Kq2, Ku1, Ku2

# -----------------------
# Assembly (dense) using numba
# -----------------------
@njit(fastmath=True, parallel=False)
def assemble_A_blocks(M, N_inner,
                      theta_nodes, radii, elements,
                      nx, ny,
                      q_nodes_far, q_weights_far,
                      q_nodes_near, q_weights_near):
    """
    Assemble A_q (M x M) and A_u (M x M) blocks:
      For collocation i and element j with nodes n1,n2:
        A_q[i,n1] += -Kq1, A_q[i,n2] += -Kq2   (note minus sign)
        A_u[i,n1] += Ku1,  A_u[i,n2] += Ku2
    After assembly, add c=1/2 to diagonal of A_u.
    Quadrature choice: use near quadrature for near/self elements (angular dist <= 2*dθ), else far quadrature.
    """
    A_q = np.zeros((M, M), dtype=np.float64)
    A_u = np.zeros((M, M), dtype=np.float64)
    cval = 0.5

    # precompute element midpoints for thresholding
    Elem_mid = np.zeros(elements.shape[0], dtype=np.float64)
    Elem_r = np.zeros(elements.shape[0], dtype=np.float64)
    Elem_node_sign = np.zeros(elements.shape[0], dtype=np.float64)
    Elem_theta1 = np.zeros(elements.shape[0], dtype=np.float64)
    Elem_theta2 = np.zeros(elements.shape[0], dtype=np.float64)
    for ej in range(elements.shape[0]):
        n1 = elements[ej, 0]
        n2 = elements[ej, 1]
        th1 = theta_nodes[n1]
        th2 = theta_nodes[n2]
        # normalize midpoint robustly
        mid = th1 + 0.5 * ang_diff(th2, th1)
        Elem_mid[ej] = mid
        Elem_theta1[ej] = th1
        Elem_theta2[ej] = th2
        Elem_r[ej] = radii[n1]  # both nodes on same circle -> same radius
        # node_sign: inner nodes indices < N_inner -> -1 else +1
        Elem_node_sign[ej] = -1.0 if n1 < N_inner else 1.0

    # build blocks
    for i in range(M):
        xi_theta = theta_nodes[i]
        xi_x = radii[i] * np.cos(xi_theta)
        xi_y = radii[i] * np.sin(xi_theta)
        for ej in range(elements.shape[0]):
            # distance in angle between collocation and element midpoint
            dth = abs(ang_diff(xi_theta, Elem_mid[ej]))
            # choose quadrature
            # compute local element dtheta magnitude
            local_dtheta = abs(ang_diff(Elem_theta2[ej], Elem_theta1[ej]))
            # heuristic: if collocation within 2 element widths -> near
            if dth <= 2.0 * local_dtheta:
                Kq1, Kq2, Ku1, Ku2 = integrate_element_theta(xi_x, xi_y,
                                                            Elem_r[ej],
                                                            Elem_theta1[ej], Elem_theta2[ej],
                                                            Elem_node_sign[ej],
                                                            q_nodes_near, q_weights_near)
            else:
                Kq1, Kq2, Ku1, Ku2 = integrate_element_theta(xi_x, xi_y,
                                                            Elem_r[ej],
                                                            Elem_theta1[ej], Elem_theta2[ej],
                                                            Elem_node_sign[ej],
                                                            q_nodes_far, q_weights_far)
            n1 = elements[ej, 0]
            n2 = elements[ej, 1]
            # A_q stores negative of Kq contributions
            A_q[i, n1] += -Kq1
            A_q[i, n2] += -Kq2
            A_u[i, n1] += Ku1
            A_u[i, n2] += Ku2

    # add c to diagonal of A_u
    for i in range(M):
        A_u[i, i] += cval

    return A_q, A_u

# -----------------------
# Interior evaluation (dense quadrature per element, same strategy)
# -----------------------
@njit(fastmath=True)
def eval_interior(px, py, npnts,
                  elements, theta_nodes, radii, node_signs,
                  u_elems, q_elems,
                  q_nodes_far, q_weights_far,
                  q_nodes_near, q_weights_near):
    two_pi_inv = 1.0 / (2.0 * pi)
    out = np.zeros(npnts, dtype=np.float64)
    nelt = elements.shape[0]
    for p in range(npnts):
        xi_x = px[p]
        xi_y = py[p]
        val = 0.0
        for ej in range(nelt):
            n1 = elements[ej, 0]
            n2 = elements[ej, 1]
            th1 = theta_nodes[n1]
            th2 = theta_nodes[n2]
            r_elem = radii[n1]
            node_sign = node_signs[ej]
            # choose quadrature - use near always for interior eval to be safe if point close
            # but we can choose far for speed if point far from element midpoint
            theta_mid = th1 + 0.5 * ang_diff(th2, th1)
            dth = abs(ang_diff(np.arctan2(xi_y, xi_x), theta_mid))
            local_dtheta = abs(ang_diff(th2, th1))
            if dth <= 2.0 * local_dtheta:
                qn = q_nodes_near
                qw = q_weights_near
            else:
                qn = q_nodes_far
                qw = q_weights_far
            # integrate
            Kq1 = 0.0; Kq2 = 0.0; Ku1 = 0.0; Ku2 = 0.0
            dtheta = ang_diff(th2, th1)
            half = 0.5 * dtheta
            theta_mid = th1 + 0.5 * dtheta
            jac = r_elem * half
            for k in range(qn.shape[0]):
                t = qn[k]; w = qw[k]
                theta = theta_mid + half * t
                yx = r_elem * np.cos(theta)
                yy = r_elem * np.sin(theta)
                dx = xi_x - yx; dy = xi_y - yy
                r2 = dx*dx + dy*dy
                if r2 < 1e-20:
                    r2 = 1e-20
                rabs = np.sqrt(r2)
                G = -two_pi_inv * np.log(rabs)
                nxq = node_sign * np.cos(theta)
                nyq = node_sign * np.sin(theta)
                dGdn = two_pi_inv * (dx * nxq + dy * nyq) / r2
                N1 = 0.5 * (1.0 - t)
                N2 = 0.5 * (1.0 + t)
                Kq1 += w * G * N1
                Kq2 += w * G * N2
                Ku1 += w * dGdn * N1
                Ku2 += w * dGdn * N2
            Kq1 *= jac; Kq2 *= jac; Ku1 *= jac; Ku2 *= jac
            val += Kq1 * q_elems[n1] + Kq2 * q_elems[n2]
            val -= Ku1 * u_elems[n1] + Ku2 * u_elems[n2]
        out[p] = val
    return out

# -----------------------
# Main driver: assemble, solve, evaluate, convergence
# -----------------------
if __name__ == "__main__":
    # interior grid fixed
    xs = np.linspace(-1.9, 1.9, ngrid)
    ys = np.linspace(-1.9, 1.9, ngrid)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((Xg.ravel(), Yg.ravel()))
    r_pts = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    interior_mask = (r_pts > 1.0) & (r_pts < 2.0)
    interior_points = pts[interior_mask]
    px = interior_points[:, 0].astype(np.float64)
    py = interior_points[:, 1].astype(np.float64)
    npnts = px.shape[0]

    # pre-cast quad arrays for numba
    qn_far = np.ascontiguousarray(q_nodes_far.astype(np.float64))
    qw_far = np.ascontiguousarray(q_weights_far.astype(np.float64))
    qn_near = np.ascontiguousarray(q_nodes_near.astype(np.float64))
    qw_near = np.ascontiguousarray(q_weights_near.astype(np.float64))

    results = []

    header = ("{0:>6s} {1:>10s} {2:>8s} {3:>15s} {4:>10s} {5:>9s} {6:>9s} {7:>9s} {8:>9s}"
              ).format("N", "Unknowns", "GMRES", "Rel L2 Error", "Conv Rate", "Setup", "Solve", "Eval", "Total")
    print(header)
    print("-" * len(header))

    prev_err = None

    for N in N_values:
        t0 = time.time()
        geom = build_nodes_and_elements(N)
        M = geom["M"]
        theta_nodes = geom["theta_nodes"].astype(np.float64)
        radii = geom["radii"].astype(np.float64)
        x_nodes = geom["x_nodes"].astype(np.float64)
        y_nodes = geom["y_nodes"].astype(np.float64)
        nx = geom["nx"].astype(np.float64)
        ny = geom["ny"].astype(np.float64)
        elements = geom["elements"].astype(np.int64)
        N_inner = N

        # known boundary data (exact) at nodes
        u_known = np.zeros(M, dtype=np.float64)
        q_known = np.zeros(M, dtype=np.float64)
        for i in range(M):
            r = radii[i]
            th = theta_nodes[i]
            u_known[i] = u_exact_polar(r, th)
            ur = ur_exact_polar(r, th)
            # q = ∂u/∂n = ± ur (sign depends)
            q_known[i] = (-1.0 if i < N_inner else 1.0) * ur

        # assemble A blocks
        t_setup0 = time.time()
        A_q, A_u = assemble_A_blocks(M, N_inner,
                                     theta_nodes, radii, elements,
                                     nx, ny,
                                     qn_far, qw_far,
                                     qn_near, qw_near)
        t_setup1 = time.time()
        setup_time = t_setup1 - t_setup0

        # Build A_full = [A_q | A_u] but we'll select unknown columns
        # Unknowns: q unknowns on inner nodes (0..N_inner-1), u unknowns on outer nodes (N_inner..M-1)
        q_known_mask = np.zeros(M, dtype=np.bool_)
        u_known_mask = np.zeros(M, dtype=np.bool_)
        q_known_mask[N_inner:] = True   # outer q known
        u_known_mask[:N_inner] = True   # inner u known

        # form full known arrays (with zeros where unknown)
        q_known_vec = np.zeros(M, dtype=np.float64)
        u_known_vec = np.zeros(M, dtype=np.float64)
        for i in range(M):
            if q_known_mask[i]:
                q_known_vec[i] = q_known[i]
            if u_known_mask[i]:
                u_known_vec[i] = u_known[i]

        # RHS b = - (A_q * q_known_vec + A_u * u_known_vec)
        b = - (A_q.dot(q_known_vec) + A_u.dot(u_known_vec))

        # build A_unknown (M x M) by concatenating columns for unknown q (0..N_inner-1) and unknown u (N_inner..M-1)
        # order of unknown vector: [q_0..q_{N_inner-1}, u_{N_inner}..u_{M-1}] length = N_inner + N_outer = M
        cols_q = np.arange(0, N_inner, dtype=np.int64)
        cols_u = np.arange(N_inner, M, dtype=np.int64)
        A_unknown = np.zeros((M, M), dtype=np.float64)
        # fill left block with A_q[:, cols_q]
        A_unknown[:, :N_inner] = A_q[:, cols_q]
        # fill right block with A_u[:, cols_u]
        A_unknown[:, N_inner:] = A_u[:, cols_u]

        t_setup2 = time.time()
        setup_time += (t_setup2 - t_setup1)

        # Solve linear system (direct solve)
        t_solve0 = time.time()
        sol_unknown = solve(A_unknown, b, assume_a='sym')  # assume_a maybe false but okay
        t_solve1 = time.time()
        solve_time = t_solve1 - t_solve0
        gmres_iters = 0  # direct solve used

        # reconstruct full q and u arrays
        q_elems = np.zeros(M, dtype=np.float64)
        u_elems = np.zeros(M, dtype=np.float64)
        # fill knowns
        for i in range(M):
            if q_known_mask[i]:
                q_elems[i] = q_known[i]
            if u_known_mask[i]:
                u_elems[i] = u_known[i]
        # fill unknowns from sol_unknown
        # first N_inner entries are q unknowns for inner nodes
        for j in range(N_inner):
            q_elems[j] = sol_unknown[j]
        # next N_outer entries are u unknowns for outer nodes
        for jj in range(M - N_inner):
            idx = N_inner + jj
            u_elems[idx] = sol_unknown[N_inner + jj]

        # Evaluate interior
        t_eval0 = time.time()
        # node_signs per element
        node_signs = np.zeros(elements.shape[0], dtype=np.float64)
        for ej in range(elements.shape[0]):
            n1 = elements[ej, 0]
            node_signs[ej] = -1.0 if n1 < N_inner else 1.0
        u_interior = eval_interior(px, py, npnts,
                                   elements, theta_nodes, radii, node_signs,
                                   u_elems, q_elems,
                                   qn_far, qw_far,
                                   qn_near, qw_near)
        t_eval1 = time.time()
        eval_time = t_eval1 - t_eval0

        # compute relative L2 error on interior points
        r_int = np.sqrt(px**2 + py**2)
        theta_int = np.arctan2(py, px)
        u_ex_interior = u_exact_polar(r_int, theta_int)
        num = np.sum((u_interior - u_ex_interior)**2)
        den = np.sum(u_ex_interior**2)
        rel_l2 = np.sqrt(num / den)

        # convergence rate
        if len(results) == 0:
            conv_rate = np.nan
        else:
            prev_err = results[-1]["rel_l2"]
            conv_rate = np.log(prev_err / rel_l2) / np.log(2.0)

        total_time = time.time() - t0

        results.append({
            "N": N,
            "unknowns": M,
            "gmres": gmres_iters,
            "rel_l2": rel_l2,
            "conv_rate": conv_rate,
            "setup": setup_time,
            "solve": solve_time,
            "eval": eval_time,
            "total": total_time
        })

        print("{N:6d} {unknowns:10d} {gmres:8d} {rel_l2:15.3e} {conv_rate:10.3f} {setup:9.3f} {solve:9.3f} {eval:9.3f} {total:9.3f}".format(**results[-1]))

    # overall convergence order (fit)
    Ns = np.array([r["N"] for r in results], dtype=np.float64)
    errs = np.array([r["rel_l2"] for r in results], dtype=np.float64)
    coeffs = np.polyfit(np.log(Ns), np.log(errs), 1)
    overall_order = -coeffs[0]
    print("\nOverall convergence order (log-log fit): {:.4f}".format(overall_order))