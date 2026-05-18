"""
2D Boundary Element Method (BEM) solver for Laplace's equation on NACA 4318 airfoil.
Direct Galerkin formulation with mixed Dirichlet-Neumann BCs.
Manufactured solution: u(x,y) = x^3 - 3xy^2
"""

import numpy as np
import time
import sys
from numba import njit, prange

# ============================================================
# 1. NACA 4-digit airfoil geometry
# ============================================================

def naca4_points(m, p, t, N_elem):
    n_half = N_elem // 2
    beta = np.linspace(0, np.pi, n_half + 1)
    xc = 0.5 * (1 - np.cos(beta))

    yc = np.where(xc < p,
                  m/p**2*(2*p*xc - xc**2),
                  m/(1-p)**2*((1-2*p) + 2*p*xc - xc**2))
    dyc = np.where(xc < p,
                   2*m/p**2*(p - xc),
                   2*m/(1-p)**2*(p - xc))
    yt = 5*t*(0.2969*np.sqrt(np.maximum(xc, 0)) - 0.1260*xc
              - 0.3516*xc**2 + 0.2843*xc**3 - 0.1015*xc**4)
    theta = np.arctan(dyc)

    xu = xc - yt*np.sin(theta);  yu = yc + yt*np.cos(theta)
    xl = xc + yt*np.sin(theta);  yl = yc - yt*np.cos(theta)

    upper_x = xu[::-1]; upper_y = yu[::-1]
    lower_x = xl[1:];   lower_y = yl[1:]

    xs = np.concatenate([upper_x, lower_x])
    ys = np.concatenate([upper_y, lower_y])
    xs[0] = 1.0; ys[0] = 0.0
    xs[-1] = 1.0; ys[-1] = 0.0

    nodes = np.column_stack([xs, ys])
    elements = np.column_stack([np.arange(N_elem), np.arange(1, N_elem+1)]).astype(np.int32)

    dx = nodes[elements[:,1],0] - nodes[elements[:,0],0]
    dy = nodes[elements[:,1],1] - nodes[elements[:,0],1]
    lengths = np.sqrt(dx**2 + dy**2)
    normals = np.column_stack([dy/lengths, -dx/lengths])

    return nodes, elements, lengths, normals


def assign_bcs(nodes, split_x=0.8):
    return (nodes[:, 0] >= split_x).astype(np.int32)


def u_exact(x, y):
    return x**3 - 3*x*y**2

def q_exact(x, y, nx, ny):
    return (3*x**2 - 3*y**2)*nx + (-6*x*y)*ny


# ============================================================
# 2. Gaussian quadrature on [0,1]
# ============================================================

def gauss_ref(n):
    pts, wts = np.polynomial.legendre.leggauss(n)
    return 0.5*(pts+1), 0.5*wts


# ============================================================
# 3. Analytical self-integral for G (coinciding elements)
# Handles log singularity via Duffy decomposition on [0,1]^2
# ============================================================

def G_self_analytical(L):
    ng = 32
    gp, gw = gauss_ref(ng)
    G_self = np.zeros((2,2))

    for sub in range(2):
        U, V = np.meshgrid(gp, gp)
        WU, WV = np.meshgrid(gw, gw)
        if sub == 0:
            xi_a  = U
            eta_a = U + (1-U)*V
            jac   = 1 - U
        else:
            eta_a = U
            xi_a  = U + (1-U)*V
            jac   = 1 - U

        r = np.maximum(np.abs(xi_a - eta_a), 1e-300)
        kern = -1/(2*np.pi) * np.log(L * r) * L**2 * jac

        phi_xi  = np.array([1-xi_a,  xi_a ])
        phi_eta = np.array([1-eta_a, eta_a])

        for i in range(2):
            for j in range(2):
                G_self[i,j] += np.sum(kern * phi_xi[i] * phi_eta[j] * WU * WV)

    return G_self


# ============================================================
# 4. Numba kernels
# ============================================================

@njit(cache=True)
def _duffy_adjacent_nb(ex0x, ex0y, ex1x, ex1y,
                        ey0x, ey0y, ey1x, ey1y,
                        Le, Lf, ny0, ny1, se_x, se_y,
                        gp, gw):
    """Duffy-transformed integral for adjacent elements sharing a vertex."""
    ng  = gp.shape[0]
    INV2PI = 1.0 / (6.283185307179586)
    G_c = np.zeros((2, 2))
    H_c = np.zeros((2, 2))

    for tri in range(2):
        for iu in range(ng):
            u  = gp[iu]; wu = gw[iu]
            for iv in range(ng):
                v  = gp[iv]; wv = gw[iv]

                if tri == 0:
                    xi_l  = u
                    eta_l = u * v
                    jac   = u
                else:
                    xi_l  = u * v
                    eta_l = u
                    jac   = u

                xi_a  = xi_l  if se_x == 0 else 1.0 - xi_l
                eta_a = eta_l if se_y == 0 else 1.0 - eta_l

                px = ex0x + xi_a  * (ex1x - ex0x)
                py = ex0y + xi_a  * (ex1y - ex0y)
                qx = ey0x + eta_a * (ey1x - ey0x)
                qy = ey0y + eta_a * (ey1y - ey0y)

                rx = px - qx;  ry = py - qy
                r2 = rx*rx + ry*ry
                if r2 < 1e-300: r2 = 1e-300
                base = Le * Lf * jac

                kern_G = -INV2PI * 0.5 * np.log(r2) * base
                kern_H =  INV2PI * (rx*ny0 + ry*ny1) / r2 * base

                w = wu * wv
                phi_xi  = np.array([1.0-xi_a,  xi_a ])
                phi_eta = np.array([1.0-eta_a, eta_a])
                for i in range(2):
                    for j in range(2):
                        contrib = kern_G * phi_xi[i] * phi_eta[j] * w
                        G_c[i, j] += contrib
                        H_c[i, j] += kern_H * phi_xi[i] * phi_eta[j] * w

    return G_c, H_c


@njit(parallel=True, cache=True)
def _assemble_disjoint_nb(ex_b, ey_b,
                           nodes_x, nodes_y,
                           elem0, elem1,
                           lengths, normals_x, normals_y,
                           gp, gw):
    """
    Vectorised disjoint-pair assembly.
    Returns G_out (n_pairs,2,2) and H_out (n_pairs,2,2).
    Parallelised over pairs.
    """
    n_pairs = ex_b.shape[0]
    ng = gp.shape[0]
    INV2PI = 1.0 / (6.283185307179586)

    G_out = np.zeros((n_pairs, 2, 2))
    H_out = np.zeros((n_pairs, 2, 2))

    for k in prange(n_pairs):
        ex = ex_b[k]; ey = ey_b[k]
        n0x = elem0[ex]; n1x = elem1[ex]
        n0y = elem0[ey]; n1y = elem1[ey]

        ax = nodes_x[n0x]; ay = nodes_y[n0x]
        bx = nodes_x[n1x]; by = nodes_y[n1x]
        cx = nodes_x[n0y]; cy = nodes_y[n0y]
        dx = nodes_x[n1y]; dy = nodes_y[n1y]

        Le = lengths[ex]; Lf = lengths[ey]
        nyx = normals_x[ey]; nyy = normals_y[ey]

        for ixi in range(ng):
            xi  = gp[ixi]; wi = gw[ixi]
            px  = ax + xi*(bx-ax)
            py  = ay + xi*(by-ay)
            phi_xi0 = 1.0 - xi
            phi_xi1 = xi

            for ieta in range(ng):
                eta = gp[ieta]; we = gw[ieta]
                qx  = cx + eta*(dx-cx)
                qy  = cy + eta*(dy-cy)

                rx = px - qx;  ry = py - qy
                r2 = rx*rx + ry*ry
                if r2 < 1e-300: r2 = 1e-300

                kern_G = -INV2PI * 0.5 * np.log(r2) * Le * Lf
                kern_H =  INV2PI * (rx*nyx + ry*nyy) / r2 * Le * Lf
                w = wi * we

                phi_eta0 = 1.0 - eta
                phi_eta1 = eta

                G_out[k,0,0] += kern_G * phi_xi0 * phi_eta0 * w
                G_out[k,0,1] += kern_G * phi_xi0 * phi_eta1 * w
                G_out[k,1,0] += kern_G * phi_xi1 * phi_eta0 * w
                G_out[k,1,1] += kern_G * phi_xi1 * phi_eta1 * w
                H_out[k,0,0] += kern_H * phi_xi0 * phi_eta0 * w
                H_out[k,0,1] += kern_H * phi_xi0 * phi_eta1 * w
                H_out[k,1,0] += kern_H * phi_xi1 * phi_eta0 * w
                H_out[k,1,1] += kern_H * phi_xi1 * phi_eta1 * w

    return G_out, H_out


@njit(parallel=True, cache=True)
def _eval_interior_nb(pts_x, pts_y,
                       qx_e, qy_e,
                       u_e, q_e,
                       ny_x, ny_y, Le,
                       gw):
    """Interior representation formula, parallelised over eval points."""
    N_pts  = pts_x.shape[0]
    N_elem = qx_e.shape[0]
    ng     = gw.shape[0]
    INV2PI = 1.0 / (6.283185307179586)
    u_int  = np.zeros(N_pts)

    for k in prange(N_pts):
        xp = pts_x[k]; yp = pts_y[k]
        val = 0.0
        for e in range(N_elem):
            Le_e = Le[e]
            for j in range(ng):
                rx = xp - qx_e[e, j];  ry = yp - qy_e[e, j]
                r2 = rx*rx + ry*ry
                if r2 < 1e-300: r2 = 1e-300
                kG = -INV2PI * 0.5 * np.log(r2)
                kH =  INV2PI * (rx*ny_x[e] + ry*ny_y[e]) / r2
                val += gw[j] * (kG*q_e[e,j] - kH*u_e[e,j]) * Le_e
        u_int[k] = val

    return u_int


# ============================================================
# 5. Full matrix assembly
# ============================================================

def assemble_HG(nodes, elements, lengths, normals, ng_dis=10, ng_adj=14):
    N_nodes = len(nodes)
    N_elem  = len(elements)
    H = np.zeros((N_nodes, N_nodes))
    G = np.zeros((N_nodes, N_nodes))

    gp_dis, gw_dis = gauss_ref(ng_dis)
    gp_adj, gw_adj = gauss_ref(ng_adj)
    gp_dis = np.ascontiguousarray(gp_dis)
    gw_dis = np.ascontiguousarray(gw_dis)
    gp_adj = np.ascontiguousarray(gp_adj)
    gw_adj = np.ascontiguousarray(gw_adj)

    G_selfs = [G_self_analytical(lengths[e]) for e in range(N_elem)]

    adjacent_pairs = []
    disjoint_pairs = []

    elem_set = [frozenset(elements[e]) for e in range(N_elem)]
    for ex in range(N_elem):
        for ey in range(N_elem):
            if ex == ey:
                continue
            shared = elem_set[ex] & elem_set[ey]
            if len(shared) == 1:
                sn = list(shared)[0]
                se_x = 0 if elements[ex][0]==sn else 1
                se_y = 0 if elements[ey][0]==sn else 1
                adjacent_pairs.append((ex, ey, se_x, se_y))
            else:
                disjoint_pairs.append((ex, ey))

    # Coinciding
    for e in range(N_elem):
        g_s = G_selfs[e]
        for i in range(2):
            for j in range(2):
                G[elements[e,i], elements[e,j]] += g_s[i,j]

    # Adjacent (Duffy, serial — only O(N) pairs)
    nodes_x = np.ascontiguousarray(nodes[:,0])
    nodes_y = np.ascontiguousarray(nodes[:,1])
    elem0   = np.ascontiguousarray(elements[:,0])
    elem1   = np.ascontiguousarray(elements[:,1])

    for (ex, ey, se_x, se_y) in adjacent_pairs:
        nx_e = elements[ex]; ny_e = elements[ey]
        g_c, h_c = _duffy_adjacent_nb(
            nodes_x[elem0[ex]], nodes_y[elem0[ex]],
            nodes_x[elem1[ex]], nodes_y[elem1[ex]],
            nodes_x[elem0[ey]], nodes_y[elem0[ey]],
            nodes_x[elem1[ey]], nodes_y[elem1[ey]],
            lengths[ex], lengths[ey],
            normals[ey,0], normals[ey,1],
            se_x, se_y, gp_adj, gw_adj)
        for i in range(2):
            for j in range(2):
                G[elements[ex,i], elements[ey,j]] += g_c[i,j]
                H[elements[ex,i], elements[ey,j]] += h_c[i,j]

    # Disjoint (Numba parallel)
    dp    = np.array(disjoint_pairs, dtype=np.int32)
    ex_b  = np.ascontiguousarray(dp[:,0])
    ey_b  = np.ascontiguousarray(dp[:,1])
    G_out, H_out = _assemble_disjoint_nb(
        ex_b, ey_b,
        nodes_x, nodes_y, elem0, elem1,
        np.ascontiguousarray(lengths),
        np.ascontiguousarray(normals[:,0]),
        np.ascontiguousarray(normals[:,1]),
        gp_dis, gw_dis)

    for k in range(len(dp)):
        ex = ex_b[k]; ey = ey_b[k]
        for i in range(2):
            for j in range(2):
                G[elements[ex,i], elements[ey,j]] += G_out[k,i,j]
                H[elements[ex,i], elements[ey,j]] += H_out[k,i,j]

    # Equipotential trick
    for i in range(N_nodes):
        H[i,i] = -np.sum(H[i,:]) + H[i,i]

    return H, G


# ============================================================
# 6. Solve
# ============================================================

def solve_bem(nodes, elements, lengths, normals, H, G, split_x=0.8):
    N_nodes = len(nodes)
    node_bc = assign_bcs(nodes, split_x)

    node_n = np.zeros((N_nodes, 2))
    node_c = np.zeros(N_nodes)
    for e in range(len(elements)):
        for loc in range(2):
            n = elements[e,loc]
            node_n[n] += normals[e]
            node_c[n] += 1
    for n in range(N_nodes):
        if node_c[n] > 0:
            nrm = node_n[n]/node_c[n]
            nl = np.linalg.norm(nrm)
            if nl > 0: node_n[n] = nrm/nl

    u_known = np.zeros(N_nodes)
    q_known = np.zeros(N_nodes)
    for i in range(N_nodes):
        xi, yi = nodes[i]
        if node_bc[i] == 1:
            u_known[i] = u_exact(xi, yi)
        else:
            q_known[i] = q_exact(xi, yi, node_n[i,0], node_n[i,1])

    D_idx = np.where(node_bc==1)[0]
    N_idx = np.where(node_bc==0)[0]

    A = np.zeros((N_nodes, N_nodes))
    b = np.zeros(N_nodes)

    for k, i in enumerate(N_idx):
        A[:, k] = H[:, i]
    for k, i in enumerate(D_idx):
        A[:, len(N_idx)+k] = -G[:, i]

    for i in N_idx: b += G[:,i]*q_known[i]
    for i in D_idx: b -= H[:,i]*u_known[i]

    x = np.linalg.solve(A, b)

    u_vec = u_known.copy(); q_vec = q_known.copy()
    for k, i in enumerate(N_idx): u_vec[i] = x[k]
    for k, i in enumerate(D_idx): q_vec[i] = x[len(N_idx)+k]

    return u_vec, q_vec


# ============================================================
# 7. Interior evaluation
# ============================================================

def eval_interior(pts, nodes, elements, lengths, normals, u_vec, q_vec, ng=10):
    gp, gw = gauss_ref(ng)
    gp = np.ascontiguousarray(gp)
    gw = np.ascontiguousarray(gw)

    n0_all = elements[:,0]; n1_all = elements[:,1]
    Ex0 = nodes[n0_all]; Ex1 = nodes[n1_all]
    U0 = u_vec[n0_all]; U1 = u_vec[n1_all]
    Q0 = q_vec[n0_all]; Q1 = q_vec[n1_all]

    qx_e = Ex0[:,0:1] + gp[None,:]*(Ex1[:,0:1]-Ex0[:,0:1])
    qy_e = Ex0[:,1:2] + gp[None,:]*(Ex1[:,1:2]-Ex0[:,1:2])
    u_e  = (1-gp[None,:])*U0[:,None] + gp[None,:]*U1[:,None]
    q_e  = (1-gp[None,:])*Q0[:,None] + gp[None,:]*Q1[:,None]

    return _eval_interior_nb(
        np.ascontiguousarray(pts[:,0]),
        np.ascontiguousarray(pts[:,1]),
        np.ascontiguousarray(qx_e),
        np.ascontiguousarray(qy_e),
        np.ascontiguousarray(u_e),
        np.ascontiguousarray(q_e),
        np.ascontiguousarray(normals[:,0]),
        np.ascontiguousarray(normals[:,1]),
        np.ascontiguousarray(lengths),
        gw)


# ============================================================
# 8. Polygon utilities
# ============================================================

def points_in_polygon(pts, poly):
    n = len(poly)
    px = pts[:,0]; py = pts[:,1]
    inside = np.zeros(len(pts), dtype=bool)
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[(i+1)%n]
        mask = ((yi>py) != (yj>py))
        x_int = (xj-xi)*(py-yi)/((yj-yi)+1e-300)+xi
        inside ^= (mask & (px < x_int))
    return inside


def min_dist_to_boundary(pts, poly):
    n = len(poly)
    dists = np.full(len(pts), np.inf)
    for i in range(n):
        A = np.array(poly[i], dtype=float)
        B = np.array(poly[(i+1)%n], dtype=float)
        AB = B-A; L2 = np.dot(AB,AB)
        if L2 < 1e-20:
            d = np.linalg.norm(pts-A, axis=1)
        else:
            t = np.clip(((pts-A)@AB)/L2, 0, 1)
            d = np.linalg.norm(pts-(A+t[:,None]*AB), axis=1)
        dists = np.minimum(dists, d)
    return dists


# ============================================================
# 9. Refinement study
# ============================================================

def run_study():
    N_values = [400, 800, 1600, 3200, 6400]
    m, p, t_val = 0.04, 0.3, 0.18
    split_x = 0.8

    ngrid = 200
    xs = np.linspace(-0.1, 1.1, ngrid)
    ys = np.linspace(-0.2, 0.2, ngrid)
    XX, YY = np.meshgrid(xs, ys)
    all_pts = np.column_stack([XX.ravel(), YY.ravel()])

    nodes_ref, _, lengths_ref, _ = naca4_points(m, p, t_val, 100)
    poly_ref  = nodes_ref[:-1]
    perim_ref = np.sum(lengths_ref)
    h_coarse  = perim_ref / min(N_values)
    delta     = 2 * h_coarse

    interior_mask = points_in_polygon(all_pts, poly_ref)
    interior_pts  = all_pts[interior_mask]
    dist          = min_dist_to_boundary(interior_pts, poly_ref)
    grid_pts      = interior_pts[dist > delta]
    u_ex_grid     = u_exact(grid_pts[:,0], grid_pts[:,1])

    # Numba warmup
    print("Warming up Numba JIT ...", flush=True)
    _n, _e, _l, _no = naca4_points(m, p, t_val, 10)
    assemble_HG(_n, _e, _l, _no, ng_dis=4, ng_adj=4)
    _gp, _gw = gauss_ref(4)
    _Ne = 10; _ng = 4
    _qx = np.zeros((_Ne,_ng)); _qy = np.zeros((_Ne,_ng))
    _ue = np.zeros((_Ne,_ng)); _qe = np.zeros((_Ne,_ng))
    _eval_interior_nb(np.zeros(2), np.zeros(2), _qx, _qy, _ue, _qe,
                      np.zeros(_Ne), np.zeros(_Ne), np.ones(_Ne), _gw)

    L2_errors = []

    print("=" * 75)
    print(f"{'BEM SOLVER | NACA 4318 | Galerkin | Mixed BC | Linear Elements':^75}")
    print("=" * 75)
    print(f"  Manufactured solution : u(x,y) = x^3 - 3xy^2")
    print(f"  Neumann (x < 0.8)     : q = (3x^2-3y^2)*nx - 6xy*ny")
    print(f"  Dirichlet (x >= 0.8)  : u = x^3 - 3xy^2")
    print(f"  Trailing edge         : double nodes (coincident, distinct normals)")
    print(f"  Interior eval points  : {len(grid_pts)}")
    print("=" * 75)
    print()
    print(f"  {'N':>5}  {'Nodes':>6}  {'L2 Error':>12}  {'Assembly(s)':>12}  "
          f"{'Solve(s)':>9}  {'Eval(s)':>8}  {'Total(s)':>9}")
    print("  " + "-" * 70)

    for N_elem in N_values:
        t0 = time.time()

        nodes, elements, lengths, normals = naca4_points(m, p, t_val, N_elem)
        N_nodes = len(nodes)

        t_asm = time.time()
        H, G = assemble_HG(nodes, elements, lengths, normals)
        t_asm = time.time() - t_asm

        t_slv = time.time()
        u_vec, q_vec = solve_bem(nodes, elements, lengths, normals, H, G, split_x)
        t_slv = time.time() - t_slv

        t_ev = time.time()
        u_num = eval_interior(grid_pts, nodes, elements, lengths, normals, u_vec, q_vec)
        t_ev = time.time() - t_ev

        L2 = np.linalg.norm(u_num - u_ex_grid) / (np.linalg.norm(u_ex_grid)+1e-300)
        L2_errors.append(L2)
        t_tot = time.time() - t0

        print(f"  {N_elem:>5}  {N_nodes:>6}  {L2:>12.4e}  {t_asm:>12.2f}  "
              f"{t_slv:>9.3f}  {t_ev:>8.2f}  {t_tot:>9.2f}")
        sys.stdout.flush()

    print("  " + "-" * 70)
    print()

    h_arr    = 1.0/np.array(N_values, dtype=float)
    log_h    = np.log(h_arr)
    log_err  = np.log(np.array(L2_errors))
    slope, _ = np.polyfit(log_h, log_err, 1)

    print("=" * 75)
    print("  CONVERGENCE ANALYSIS")
    print("=" * 75)
    print(f"  {'N':>5}  {'h = 1/N':>10}  {'L2 Error':>12}  {'log(h)':>10}  {'log(err)':>10}")
    print("  " + "-" * 55)
    for i, N in enumerate(N_values):
        print(f"  {N:>5}  {h_arr[i]:>10.6f}  {L2_errors[i]:>12.4e}  "
              f"{log_h[i]:>10.4f}  {log_err[i]:>10.4f}")
    print()
    print(f"  Estimated convergence order (least-squares slope): {slope:.3f}")
    print(f"  Expected (linear Galerkin BEM):                    ~2.0")
    print("=" * 75)


if __name__ == "__main__":
    run_study()