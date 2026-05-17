import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange

# ---------------------------------------------------------------------------
# Gauss–Legendre quadrature (8-point, same as reference)
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
#   u(x,y)  = sinh(pi*x) / sinh(2*pi) * cos(pi*y)
#   du/dx   = pi*cosh(pi*x) / sinh(2*pi) * cos(pi*y)
#   du/dy   = -pi*sinh(pi*x) / sinh(2*pi) * sin(pi*y)
#   q = du/dn = (du/dx)*nx + (du/dy)*ny
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
# Matrix-free matvec  (same kernel/singular treatment as reference)
#
# Unknowns vector v holds:
#   v[j] = q_j   if bc_type[j] == 0  (Dirichlet panel → unknown flux)
#   v[j] = u_j   if bc_type[j] == 1  (Neumann panel  → unknown potential)
#
# The BIE row for collocation point i is:
#   0.5*u_i + sum_j H_ij*u_j = sum_j G_ij*q_j
#
# Rearranged so that the unknown v appears on the left:
#   For Dirichlet panel j : coefficient of v[j]=q[j]  in row i is  +G_ij
#   For Neumann  panel j  : coefficient of v[j]=u[j]  in row i is  -H_ij
#                           (and H_ii contributes 0.5 to the diagonal)
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def bem_matvec(v, mid_x, mid_y, norm_x, norm_y, lengths, bc_type, pts, wts):
    N_tot = len(mid_x)
    out = np.zeros(N_tot)
    for i in prange(N_tot):
        row_val = 0.0
        for j in range(N_tot):
            L = lengths[j]
            if i == j:
                # Analytic diagonal: G_ii = (L/(2pi))*(1 - ln(L/2)), H_ii = 0.5
                G_ij = (L / (2.0 * np.pi)) * (1.0 - np.log(L / 2.0))
                H_ij = 0.5
            else:
                g_val = 0.0
                h_val = 0.0
                tx = -norm_y[j]   # tangent direction along panel j
                ty =  norm_x[j]
                for k in range(len(pts)):
                    pos = 0.5 * pts[k] * L          # local coordinate in [-L/2, L/2]
                    yx = mid_x[j] + pos * tx
                    yy = mid_y[j] + pos * ty
                    rx = mid_x[i] - yx
                    ry = mid_y[i] - yy
                    r2 = rx*rx + ry*ry
                    # G  = -1/(2pi) * ln(r)  = -1/(4pi) * ln(r^2)
                    g_val += -1.0 / (4.0 * np.pi) * np.log(r2) * wts[k]
                    # H  = 1/(2pi) * (r·n)/r^2
                    h_val += 1.0 / (2.0 * np.pi * r2) * (rx * norm_x[j] + ry * norm_y[j]) * wts[k]
                G_ij = g_val * 0.5 * L
                H_ij = h_val * 0.5 * L

            if bc_type[j] == 0:          # Dirichlet → q unknown
                row_val += G_ij * v[j]
            else:                         # Neumann  → u unknown
                row_val -= H_ij * v[j]
        out[i] = row_val
    return out

# ---------------------------------------------------------------------------
# Build RHS from the known boundary data
#
# RHS_i = sum_{j: Neumann}  G_ij*q_known[j]
#        -sum_{j: Dirichlet} H_ij*u_known[j]
#        + 0.5 * u_i   (the free term is included via H_ii = 0.5 above;
#                        but on the RHS we move the known side over)
#
# Concretely (mirrors reference exactly):
#   for Dirichlet panel j : known quantity is u[j]  → adds  +H_ij*u_known[j]
#   for Neumann  panel j  : known quantity is q[j]  → adds  -G_ij*q_known[j]
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def build_rhs(u_known, q_known, mid_x, mid_y, norm_x, norm_y, lengths, bc_type, pts, wts):
    N_tot = len(mid_x)
    rhs = np.zeros(N_tot)
    for i in prange(N_tot):
        row_val = 0.0
        for j in range(N_tot):
            L = lengths[j]
            if i == j:
                G_ij = (L / (2.0 * np.pi)) * (1.0 - np.log(L / 2.0))
                H_ij = 0.5
            else:
                g_val = 0.0
                h_val = 0.0
                tx = -norm_y[j]
                ty =  norm_x[j]
                for k in range(len(pts)):
                    pos = 0.5 * pts[k] * L
                    yx = mid_x[j] + pos * tx
                    yy = mid_y[j] + pos * ty
                    rx = mid_x[i] - yx
                    ry = mid_y[i] - yy
                    r2 = rx*rx + ry*ry
                    g_val += -1.0 / (4.0 * np.pi) * np.log(r2) * wts[k]
                    h_val += 1.0 / (2.0 * np.pi * r2) * (rx * norm_x[j] + ry * norm_y[j]) * wts[k]
                G_ij = g_val * 0.5 * L
                H_ij = h_val * 0.5 * L

            if bc_type[j] == 0:          # Dirichlet panel: u known → move H*u to RHS
                row_val += H_ij * u_known[j]
            else:                         # Neumann panel:  q known → move G*q to RHS
                row_val -= G_ij * q_known[j]
        rhs[i] = row_val
    return rhs

# ---------------------------------------------------------------------------
# Interior evaluation via representation formula
#   u(x) = sum_j [ G_ij*q_j - H_ij*u_j ]
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def eval_interior(ix_pts, iy_pts, u_all, q_all, mid_x, mid_y, norm_x, norm_y, lengths, pts, wts):
    M = len(ix_pts)
    N_tot = len(mid_x)
    u_int = np.zeros(M)
    for i in prange(M):
        val = 0.0
        for j in range(N_tot):
            L = lengths[j]
            g_val = 0.0
            h_val = 0.0
            tx = -norm_y[j]
            ty =  norm_x[j]
            for k in range(len(pts)):
                pos = 0.5 * pts[k] * L
                yx = mid_x[j] + pos * tx
                yy = mid_y[j] + pos * ty
                rx = ix_pts[i] - yx
                ry = iy_pts[i] - yy
                r2 = rx*rx + ry*ry
                g_val += -1.0 / (4.0 * np.pi) * np.log(r2) * wts[k]
                h_val += 1.0 / (2.0 * np.pi * r2) * (rx * norm_x[j] + ry * norm_y[j]) * wts[k]
            G_ij = g_val * 0.5 * L
            H_ij = h_val * 0.5 * L
            val += G_ij * q_all[j] - H_ij * u_all[j]
        u_int[i] = val
    return u_int

# ---------------------------------------------------------------------------
# GMRES iteration counter (same as reference)
# ---------------------------------------------------------------------------
class IterationCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1

# ---------------------------------------------------------------------------
# Main solver
#
# Rectangle  Ω = [0,2] × [0,1]
#
# Boundary conditions
#   Left  (x=0): Dirichlet  u = 0              bc_type = 0
#   Right (x=2): Dirichlet  u = cos(pi*y)      bc_type = 0
#   Bottom(y=0): Neumann    du/dy = 0           bc_type = 1
#   Top   (y=1): Neumann    du/dy = 0           bc_type = 1
#
# Panel distribution: N panels per unit length so the mesh is isotropic.
#   Bottom / Top : 2*N panels each  (length 2)
#   Left  / Right:   N panels each  (length 1)
#   Total DOFs = 6*N
# ---------------------------------------------------------------------------
def solve_bem(N):
    t_start = time.time()

    # --- Bottom side: y=0, x in [0,2], outward normal (0,-1) ---
    Nb = 2 * N                          # panels proportional to length
    xb = np.linspace(0.0, 2.0, Nb + 1)
    xb_mid = 0.5 * (xb[:-1] + xb[1:])
    yb_mid = np.zeros(Nb)
    nb_x   = np.zeros(Nb)
    nb_y   = np.full(Nb, -1.0)
    lb     = np.full(Nb, 2.0 / Nb)
    bc_b   = np.ones(Nb, dtype=np.int64)    # Neumann

    # --- Right side: x=2, y in [0,1], outward normal (1,0) ---
    Nr = N
    yr = np.linspace(0.0, 1.0, Nr + 1)
    yr_mid = 0.5 * (yr[:-1] + yr[1:])
    xr_mid = np.full(Nr, 2.0)
    nr_x   = np.ones(Nr)
    nr_y   = np.zeros(Nr)
    lr     = np.full(Nr, 1.0 / Nr)
    bc_r   = np.zeros(Nr, dtype=np.int64)   # Dirichlet

    # --- Top side: y=1, x in [2,0] (traversed right→left), outward normal (0,1) ---
    Nt = 2 * N
    xt = np.linspace(2.0, 0.0, Nt + 1)
    xt_mid = 0.5 * (xt[:-1] + xt[1:])
    yt_mid = np.ones(Nt)
    nt_x   = np.zeros(Nt)
    nt_y   = np.ones(Nt)
    lt     = np.full(Nt, 2.0 / Nt)
    bc_t   = np.ones(Nt, dtype=np.int64)    # Neumann

    # --- Left side: x=0, y in [1,0] (traversed top→bottom), outward normal (-1,0) ---
    Nl = N
    yl = np.linspace(1.0, 0.0, Nl + 1)
    yl_mid = 0.5 * (yl[:-1] + yl[1:])
    xl_mid = np.zeros(Nl)
    nl_x   = np.full(Nl, -1.0)
    nl_y   = np.zeros(Nl)
    ll     = np.full(Nl, 1.0 / Nl)
    bc_l   = np.zeros(Nl, dtype=np.int64)   # Dirichlet

    # Assemble global arrays
    mid_x  = np.concatenate([xb_mid, xr_mid, xt_mid, xl_mid])
    mid_y  = np.concatenate([yb_mid, yr_mid, yt_mid, yl_mid])
    norm_x = np.concatenate([nb_x,   nr_x,   nt_x,   nl_x  ])
    norm_y = np.concatenate([nb_y,   nr_y,   nt_y,   nl_y  ])
    lengths= np.concatenate([lb,     lr,     lt,     ll    ])
    bc_type= np.concatenate([bc_b,   bc_r,   bc_t,   bc_l  ])

    N_tot = len(mid_x)   # = 6*N

    # Known boundary data (exact solution)
    u_known = np.array([exact_u(mid_x[j], mid_y[j]) for j in range(N_tot)])
    q_known = np.array([exact_q(mid_x[j], mid_y[j], norm_x[j], norm_y[j]) for j in range(N_tot)])

    pts, wts = get_gauss_points()

    # Build RHS
    rhs = build_rhs(u_known, q_known, mid_x, mid_y, norm_x, norm_y, lengths, bc_type, pts, wts)

    t_setup = time.time() - t_start

    # --- Matrix-free GMRES ---
    t_solve_start = time.time()

    def matvec(v):
        return bem_matvec(v, mid_x, mid_y, norm_x, norm_y, lengths, bc_type, pts, wts)

    LO = LinearOperator((N_tot, N_tot), matvec=matvec)
    counter = IterationCounter()
    sol, info = gmres(LO, rhs, callback=counter, rtol=1e-12, maxiter=N_tot,
                      callback_type='legacy')

    t_solve = time.time() - t_solve_start

    # Reconstruct full boundary solution
    u_all = np.zeros(N_tot)
    q_all = np.zeros(N_tot)
    for j in range(N_tot):
        if bc_type[j] == 0:      # Dirichlet: u known, q solved
            q_all[j] = sol[j]
            u_all[j] = u_known[j]
        else:                     # Neumann: q known, u solved
            u_all[j] = sol[j]
            q_all[j] = q_known[j]

    # --- Interior evaluation ---
    t_eval_start = time.time()
    ngrid_x = 80; ngrid_y = 40
    _gx = np.linspace(0.01, 1.99, ngrid_x)
    _gy = np.linspace(0.01, 0.99, ngrid_y)
    gx, gy = np.meshgrid(_gx, _gy)
    gx = gx.flatten(); gy = gy.flatten()
    ix_pts = gx; iy_pts = gy

    u_interior = eval_interior(ix_pts, iy_pts, u_all, q_all,
                               mid_x, mid_y, norm_x, norm_y, lengths, pts, wts)

    u_exact_int = np.array([exact_u(px, py) for px, py in zip(ix_pts, iy_pts)])
    rel_l2 = np.linalg.norm(u_interior - u_exact_int) / np.linalg.norm(u_exact_int)

    t_eval = time.time() - t_eval_start

    return {
        "N": N_tot,
        "iters": counter.niter,
        "error": rel_l2,
        "t_setup": t_setup,
        "t_solve": t_solve,
        "t_eval": t_eval,
        "t_total": t_setup + t_solve + t_eval
    }

# ---------------------------------------------------------------------------
# Refinement study
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    N_values = [160, 320, 640]   # N panels per unit length → 6N total DOFs

    # Warm-up run to compile Numba JIT kernels (excluded from timing)
    _ = solve_bem(5)

    print(f"{'N':>6} | {'DOFs':>6} | {'Asm(s)':>8} | {'Solve(s)':>9} | "
          f"{'Eval(s)':>8} | {'Total(s)':>9} | {'RelL2':>12} | {'Rate':>7}")
    print("-" * 95)

    results = []
    for n in N_values:
        res = solve_bem(n)
        results.append(res)

    for i, res in enumerate(results):
        if i == 0:
            rate_str = "    ---"
        else:
            prev = results[i - 1]
            rate = (np.log(prev['error'] / res['error']) /
                    np.log(res['N'] / prev['N']))
            rate_str = f"{rate:7.2f}"

        print(f"{N_values[i]:6d} | {res['N']:6d} | {res['t_setup']:8.3f} | "
              f"{res['t_solve']:9.3f} | {res['t_eval']:8.3f} | "
              f"{res['t_total']:9.3f} | {res['error']:12.6e} | {rate_str}")

    if len(results) > 1:
        log_slope = (np.log(results[0]['error'] / results[-1]['error']) /
                     np.log(results[-1]['N']    / results[0]['N']))
        print("-" * 95)
        print(f"Overall Convergence Slope (log-log fit, error vs DOFs): {log_slope:.3f}")