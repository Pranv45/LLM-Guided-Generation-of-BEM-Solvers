import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange

# ---------------------------------------------------------------------------
# 8-point Gauss–Legendre quadrature (identical to reference)
# ---------------------------------------------------------------------------
@njit(cache=True)
def get_gauss_points():
    pts = np.array([-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
                     0.1834346424956498,  0.5255324099163290,  0.7966664774136267,  0.9602898564975363])
    wts = np.array([ 0.1012285362903763,  0.2223810344533745,  0.3137066458778873,  0.3626837833783620,
                     0.3626837833783620,  0.3137066458778873,  0.2223810344533745,  0.1012285362903763])
    return pts, wts

# ---------------------------------------------------------------------------
# Exact solution and its normal derivative for the rectangle problem
#
#   u(x,y)  = sinh(πx)/sinh(2π) · cos(πy)
#   ∂u/∂x   = π cosh(πx)/sinh(2π) · cos(πy)
#   ∂u/∂y   = −π sinh(πx)/sinh(2π) · sin(πy)
#   q = ∂u/∂n = ∇u · n
# ---------------------------------------------------------------------------
@njit(cache=True)
def exact_u(x, y):
    return np.sinh(np.pi * x) / np.sinh(2.0 * np.pi) * np.cos(np.pi * y)

@njit(cache=True)
def exact_q(x, y, nx, ny):
    S2pi = np.sinh(2.0 * np.pi)
    dudx = np.pi * np.cosh(np.pi * x) / S2pi * np.cos(np.pi * y)
    dudy = -np.pi * np.sinh(np.pi * x) / S2pi * np.sin(np.pi * y)
    return dudx * nx + dudy * ny

# ---------------------------------------------------------------------------
# Matrix-free matvec for continuous linear BEM
#
# Node ordering:
#   Indices 0  .. N_D-1 : Dirichlet nodes  (Right + Left sides, corners included)
#                          unknowns: q_j  → v[j] = q_j
#   Indices N_D .. N_tot-1: Neumann nodes  (Bottom + Top interior)
#                          unknowns: u_j  → v[j] = u_j
#
# BIE (interior problem):   c_i·u_i + Σ_j H_ij·u_j = Σ_j G_ij·q_j
#
# Rearranged LHS (unknowns):
#   Dirichlet j (j < N_D):  +G1/G2 · v[j]   (q unknown)
#   Neumann   j (j ≥ N_D):  −H1/H2 · v[j]   (u unknown)
#   Free term at Neumann i:  −c_i · v[i]
#
# Singular formulas (analytic, for straight element of length L):
#   i == j1:  G1 = −1/(2π)·(L/2·ln L − 3L/4),  G2 = −1/(2π)·(L/2·ln L − L/4),  H=0
#   i == j2:  G1 = −1/(2π)·(L/2·ln L − L/4),   G2 = −1/(2π)·(L/2·ln L − 3L/4), H=0
#   (H=0 because (r·n)=0 when r is along the element tangent)
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def bem_matvec(v, nodes_x, nodes_y, elems, ex, ey, lengths, N_D, c_free, pts, wts):
    N_tot = len(nodes_x)
    N_elems = elems.shape[0]
    out = np.zeros(N_tot)
    for i in prange(N_tot):
        row_val = 0.0
        if i >= N_D:                          # Neumann node: free term on LHS
            row_val -= c_free[i] * v[i]
        for e in range(N_elems):
            j1 = elems[e, 0]
            j2 = elems[e, 1]
            L = lengths[e]
            if i == j1:                       # field pt at start of element → analytic G, H=0
                G1 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.75 * L)
                G2 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.25 * L)
                H1 = 0.0
                H2 = 0.0
            elif i == j2:                     # field pt at end of element → analytic G, H=0
                G1 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.25 * L)
                G2 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.75 * L)
                H1 = 0.0
                H2 = 0.0
            else:                             # regular Gauss quadrature
                G1 = 0.0; G2 = 0.0; H1 = 0.0; H2 = 0.0
                for k in range(len(pts)):
                    xi = pts[k]
                    w  = wts[k]
                    N1 = 0.5 * (1.0 - xi)
                    N2 = 0.5 * (1.0 + xi)
                    yx = N1 * nodes_x[j1] + N2 * nodes_x[j2]
                    yy = N1 * nodes_y[j1] + N2 * nodes_y[j2]
                    rx = nodes_x[i] - yx
                    ry = nodes_y[i] - yy
                    r2 = rx * rx + ry * ry
                    G_val = -1.0 / (4.0 * np.pi) * np.log(r2)
                    H_val =  1.0 / (2.0 * np.pi * r2) * (rx * ex[e] + ry * ey[e])
                    wL2 = w * L * 0.5
                    G1 += G_val * N1 * wL2
                    G2 += G_val * N2 * wL2
                    H1 += H_val * N1 * wL2
                    H2 += H_val * N2 * wL2
            # Accumulate contributions from nodes j1, j2
            if j1 < N_D:                      # Dirichlet j1: q unknown → G
                row_val += G1 * v[j1]
            else:                              # Neumann   j1: u unknown → H
                row_val -= H1 * v[j1]
            if j2 < N_D:
                row_val += G2 * v[j2]
            else:
                row_val -= H2 * v[j2]
        out[i] = row_val
    return out

# ---------------------------------------------------------------------------
# Build RHS from known boundary data
#
# RHS_i = c_i·u_known_i  [if i is Dirichlet: free term]
#        + Σ_{j Dirichlet} H_ij · u_known_j   (move known H·u to RHS)
#        − Σ_{j Neumann}   G_ij · q_known_j   (move known G·q to RHS)
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def build_rhs(u_known, q_known, nodes_x, nodes_y, elems, ex, ey, lengths, N_D, c_free, pts, wts):
    N_tot   = len(nodes_x)
    N_elems = elems.shape[0]
    rhs = np.zeros(N_tot)
    for i in prange(N_tot):
        row_val = 0.0
        if i < N_D:                            # Dirichlet node: free term goes to RHS
            row_val += c_free[i] * u_known[i]
        for e in range(N_elems):
            j1 = elems[e, 0]
            j2 = elems[e, 1]
            L  = lengths[e]
            if i == j1:
                G1 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.75 * L)
                G2 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.25 * L)
                H1 = 0.0
                H2 = 0.0
            elif i == j2:
                G1 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.25 * L)
                G2 = (-1.0 / (2.0 * np.pi)) * (0.5 * L * np.log(L) - 0.75 * L)
                H1 = 0.0
                H2 = 0.0
            else:
                G1 = 0.0; G2 = 0.0; H1 = 0.0; H2 = 0.0
                for k in range(len(pts)):
                    xi = pts[k]
                    w  = wts[k]
                    N1 = 0.5 * (1.0 - xi)
                    N2 = 0.5 * (1.0 + xi)
                    yx = N1 * nodes_x[j1] + N2 * nodes_x[j2]
                    yy = N1 * nodes_y[j1] + N2 * nodes_y[j2]
                    rx = nodes_x[i] - yx
                    ry = nodes_y[i] - yy
                    r2 = rx * rx + ry * ry
                    G_val = -1.0 / (4.0 * np.pi) * np.log(r2)
                    H_val =  1.0 / (2.0 * np.pi * r2) * (rx * ex[e] + ry * ey[e])
                    wL2 = w * L * 0.5
                    G1 += G_val * N1 * wL2
                    G2 += G_val * N2 * wL2
                    H1 += H_val * N1 * wL2
                    H2 += H_val * N2 * wL2
            if j1 < N_D:                       # Dirichlet: H·u_known → RHS
                row_val += H1 * u_known[j1]
            else:                               # Neumann:   G·q_known → RHS (with sign flip)
                row_val -= G1 * q_known[j1]
            if j2 < N_D:
                row_val += H2 * u_known[j2]
            else:
                row_val -= G2 * q_known[j2]
        rhs[i] = row_val
    return rhs

# ---------------------------------------------------------------------------
# Interior representation formula:  u(x) = Σ_j [G_j·q_j − H_j·u_j]
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def eval_interior(ix_pts, iy_pts, u_all, q_all, nodes_x, nodes_y, elems, ex, ey, lengths, pts, wts):
    M       = len(ix_pts)
    N_elems = elems.shape[0]
    u_int   = np.zeros(M)
    for i in prange(M):
        val = 0.0
        for e in range(N_elems):
            j1 = elems[e, 0]
            j2 = elems[e, 1]
            L  = lengths[e]
            G1 = 0.0; G2 = 0.0; H1 = 0.0; H2 = 0.0
            for k in range(len(pts)):
                xi = pts[k]
                w  = wts[k]
                N1 = 0.5 * (1.0 - xi)
                N2 = 0.5 * (1.0 + xi)
                yx = N1 * nodes_x[j1] + N2 * nodes_x[j2]
                yy = N1 * nodes_y[j1] + N2 * nodes_y[j2]
                rx = ix_pts[i] - yx
                ry = iy_pts[i] - yy
                r2 = rx * rx + ry * ry
                G_val = -1.0 / (4.0 * np.pi) * np.log(r2)
                H_val =  1.0 / (2.0 * np.pi * r2) * (rx * ex[e] + ry * ey[e])
                wL2 = w * L * 0.5
                G1 += G_val * N1 * wL2
                G2 += G_val * N2 * wL2
                H1 += H_val * N1 * wL2
                H2 += H_val * N2 * wL2
            val += (G1 * q_all[j1] + G2 * q_all[j2]) - (H1 * u_all[j1] + H2 * u_all[j2])
        u_int[i] = val
    return u_int

class IterationCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1

# ---------------------------------------------------------------------------
# Main solver
#
# Rectangle Ω = [0,2]×[0,1],  N panels per unit length  →  h = 1/N
#
# ┌──────────── Node layout ────────────────────────────────────────────────┐
# │ Dirichlet nodes (indices 0 .. 2N+1 = N_D−1)  — u known, q unknown      │
# │   Right side (x=2): 0 .. N       (N+1 nodes, y: 0 → 1)                 │
# │   Left  side (x=0): N+1 .. 2N+1  (N+1 nodes, y: 1 → 0)                 │
# │   Corner nodes at 0=(2,0), N=(2,1), N+1=(0,1), 2N+1=(0,0)              │
# │                                                                          │
# │ Neumann nodes (indices 2N+2 .. 6N−1)  — q known (=0), u unknown         │
# │   Bottom interior (y=0): 2N+2 .. 4N     (2N−1 nodes, x: 1/N → (2N−1)/N)│
# │   Top    interior (y=1): 4N+1 .. 6N−1   (2N−1 nodes, x: (2N−1)/N → 1/N)│
# │                                                                          │
# │ Total: N_D = 2(N+1),  N_N = 4N−2,  N_tot = 6N                          │
# └──────────────────────────────────────────────────────────────────────────┘
#
# CCW boundary traversal:
#   Bottom: (0,0) → (2,0)   [2N elements]
#   Right:  (2,0) → (2,1)   [N  elements]
#   Top:    (2,1) → (0,1)   [2N elements]
#   Left:   (0,1) → (0,0)   [N  elements]
#
# Outward normal for CCW traversal: n = (dy/L, −dx/L)
#
# Free-term c_i:  0.5 for smooth boundary nodes,
#                 0.25 (= π/2 / 2π) at right-angle corners
# ---------------------------------------------------------------------------
def solve_bem(N):
    t_start = time.time()

    N_D   = 2 * (N + 1)          # Dirichlet node count
    N_N   = 4 * N - 2            # Neumann node count
    N_tot = N_D + N_N            # = 6N  total nodes
    N_elems = 6 * N              # = 6N  elements (closed polygon)

    nodes_x = np.zeros(N_tot)
    nodes_y = np.zeros(N_tot)

    # --- Dirichlet nodes ---
    # Right (x=2): y = 0, 1/N, 2/N, ..., 1  → indices 0..N
    for k in range(N + 1):
        nodes_x[k] = 2.0
        nodes_y[k] = k / N
    # Left (x=0): y = 1, (N-1)/N, ..., 0  → indices N+1..2N+1
    for k in range(N + 1):
        nodes_x[N + 1 + k] = 0.0
        nodes_y[N + 1 + k] = 1.0 - k / N

    # --- Neumann interior nodes ---
    # Bottom interior (y=0): x = 1/N, 2/N, ..., (2N-1)/N  → indices 2N+2..4N
    for k in range(2 * N - 1):
        nodes_x[2 * N + 2 + k] = (k + 1) / N
        nodes_y[2 * N + 2 + k] = 0.0
    # Top interior (y=1): x = (2N-1)/N, ..., 1/N  → indices 4N+1..6N-1
    for k in range(2 * N - 1):
        nodes_x[4 * N + 1 + k] = 2.0 - (k + 1) / N
        nodes_y[4 * N + 1 + k] = 1.0

    # --- Element connectivity (CCW traversal) ---
    elems = np.zeros((N_elems, 2), dtype=np.int32)

    # Bottom: 2N elements, (0,0)=2N+1 → … → (2,0)=0
    for k in range(2 * N):
        elems[k, 0] = 2 * N + 1 + k
        elems[k, 1] = (2 * N + 2 + k) if k < 2 * N - 1 else 0

    # Right: N elements, (2,0)=0 → (2,1)=N
    for k in range(N):
        elems[2 * N + k, 0] = k
        elems[2 * N + k, 1] = k + 1

    # Top: 2N elements, (2,1)=N → … → (0,1)=N+1
    for k in range(2 * N):
        elems[3 * N + k, 0] = N if k == 0 else 4 * N + k
        elems[3 * N + k, 1] = (4 * N + 1 + k) if k < 2 * N - 1 else N + 1

    # Left: N elements, (0,1)=N+1 → (0,0)=2N+1
    for k in range(N):
        elems[5 * N + k, 0] = N + 1 + k
        elems[5 * N + k, 1] = N + 2 + k

    # --- Element lengths and outward normals ---
    # For CCW traversal: n = (dy, −dx) / L  (rotate tangent 90° clockwise)
    ex      = np.zeros(N_elems)
    ey      = np.zeros(N_elems)
    lengths = np.zeros(N_elems)
    for e in range(N_elems):
        j1, j2 = elems[e, 0], elems[e, 1]
        dx = nodes_x[j2] - nodes_x[j1]
        dy = nodes_y[j2] - nodes_y[j1]
        L  = np.sqrt(dx * dx + dy * dy)
        lengths[e] = L
        ex[e] =  dy / L   # outward normal x
        ey[e] = -dx / L   # outward normal y

    # --- Free-term coefficients ---
    # Smooth boundary: c = 0.5;  right-angle corners: c = π/2/(2π) = 0.25
    c_free = np.full(N_tot, 0.5)
    c_free[0]           = 0.25   # corner (2, 0)
    c_free[N]           = 0.25   # corner (2, 1)
    c_free[N + 1]       = 0.25   # corner (0, 1)
    c_free[2 * N + 1]   = 0.25   # corner (0, 0)

    # --- Known boundary data ---
    # Dirichlet (right: u=cos(πy), left: u=0); Neumann (q=0 everywhere on top/bottom)
    u_known = np.array([exact_u(nodes_x[i], nodes_y[i]) for i in range(N_tot)])
    q_known = np.zeros(N_tot)    # q=0 on Neumann sides (∂u/∂y=0 on bottom and top)

    pts, wts = get_gauss_points()

    rhs = build_rhs(u_known, q_known, nodes_x, nodes_y, elems, ex, ey, lengths,
                    np.int64(N_D), c_free, pts, wts)
    t_setup = time.time() - t_start

    # --- Matrix-free GMRES ---
    t_solve_start = time.time()
    def matvec(v):
        return bem_matvec(v, nodes_x, nodes_y, elems, ex, ey, lengths,
                          np.int64(N_D), c_free, pts, wts)
    LO      = LinearOperator((N_tot, N_tot), matvec=matvec)
    counter = IterationCounter()
    sol, info = gmres(LO, rhs, callback=counter, rtol=1e-12, maxiter=N_tot,
                      callback_type='legacy')
    t_solve = time.time() - t_solve_start

    # --- Reconstruct full boundary solution ---
    u_all = np.zeros(N_tot)
    q_all = np.zeros(N_tot)
    for i in range(N_tot):
        if i < N_D:       # Dirichlet: u known, q solved
            u_all[i] = u_known[i]
            q_all[i] = sol[i]
        else:              # Neumann: q known (=0), u solved
            q_all[i] = q_known[i]
            u_all[i] = sol[i]

    # --- Interior evaluation ---
    t_eval_start = time.time()
    ngrid_x = 80; ngrid_y = 40
    _gx = np.linspace(0.01, 1.99, ngrid_x)
    _gy = np.linspace(0.01, 0.99, ngrid_y)
    gx, gy = np.meshgrid(_gx, _gy)
    gx = gx.flatten(); gy = gy.flatten()
    ix_pts = gx; iy_pts = gy

    u_interior  = eval_interior(ix_pts, iy_pts, u_all, q_all,
                                nodes_x, nodes_y, elems, ex, ey, lengths, pts, wts)
    u_exact_int = np.array([exact_u(px, py) for px, py in zip(ix_pts, iy_pts)])
    rel_l2      = np.linalg.norm(u_interior - u_exact_int) / np.linalg.norm(u_exact_int)
    t_eval = time.time() - t_eval_start

    return {
        "N":      N,
        "tot_N":  N_tot,
        "iters":  counter.niter,
        "error":  rel_l2,
        "t_setup":  t_setup,
        "t_solve":  t_solve,
        "t_eval":   t_eval,
        "t_total":  t_setup + t_solve + t_eval,
    }

# ---------------------------------------------------------------------------
# Refinement study
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Warm-up: trigger Numba JIT compilation (excluded from timing)
    _ = solve_bem(10)

    N_values = [20, 40, 80, 160, 320]
    results  = []

    print(f"{'N':>5} | {'DOFs':>6} | {'Iters':>6} | {'L2 Error':>12} | {'Rate':>8} | "
          f"{'Setup(s)':>8} | {'Solve(s)':>8} | {'Eval(s)':>8} | {'Total(s)':>8}")
    print("-" * 100)

    for n in N_values:
        res = solve_bem(n)
        results.append(res)

    for i, res in enumerate(results):
        if i == 0:
            rate_str = "     ---"
        else:
            prev = results[i - 1]
            rate = np.log(prev['error'] / res['error']) / np.log(res['N'] / prev['N'])
            rate_str = f"{rate:8.2f}"
        print(f"{res['N']:5d} | {res['tot_N']:6d} | {res['iters']:6d} | {res['error']:12.6e} | "
              f"{rate_str} | {res['t_setup']:8.3f} | {res['t_solve']:8.3f} | "
              f"{res['t_eval']:8.3f} | {res['t_total']:8.3f}")

    if len(results) > 1:
        overall_rate = (np.log(results[0]['error'] / results[-1]['error']) /
                        np.log(results[-1]['N']    / results[0]['N']))
        print("-" * 100)
        print(f"Overall Convergence Order: {overall_rate:.2f}")