import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange

@njit(cache=True)
def get_gauss_points():
    pts = np.array([-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
                     0.1834346424956498,  0.5255324099163290,  0.7966664774136267,  0.9602898564975363])
    wts = np.array([ 0.1012285362903763,  0.2223810344533745,  0.3137066458778873,  0.3626837833783620,
                     0.3626837833783620,  0.3137066458778873,  0.2223810344533745,  0.1012285362903763])
    return pts, wts

@njit(cache=True)
def exact_u(x, y):
    return (np.sinh(np.pi * x) / np.sinh(2.0 * np.pi)) * np.cos(np.pi * y)

@njit(cache=True)
def exact_q(x, y, nx, ny):
    ux = np.pi * (np.cosh(np.pi * x) / np.sinh(2.0 * np.pi)) * np.cos(np.pi * y)
    uy = -np.pi * (np.sinh(np.pi * x) / np.sinh(2.0 * np.pi)) * np.sin(np.pi * y)
    return ux * nx + uy * ny

@njit(parallel=True, fastmath=True, cache=True)
def bem_matvec(v, mid_x, mid_y, norm_x, norm_y, lengths, bc_type, pts, wts):
    N_tot = len(mid_x)
    out = np.zeros(N_tot)
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
                ty = norm_x[j]
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

            if bc_type[j] == 0:
                row_val += G_ij * v[j]
            else:
                row_val -= H_ij * v[j]
        out[i] = row_val
    return out

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
                ty = norm_x[j]
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

            if bc_type[j] == 0:
                row_val += H_ij * u_known[j]
            else:
                row_val -= G_ij * q_known[j]
        rhs[i] = row_val
    return rhs

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
            ty = norm_x[j]
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

class IterationCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1

def solve_bem(N):
    t_start = time.time()

    # Boundary Discretization (N panels per unit length)

    # Bottom side: y=0, x in [0,2], Neumann BC (q=0 -> bc_type=1)
    b_l = 2.0 / (2*N)
    b_x = np.linspace(0, 2, 2*N, endpoint=False) + b_l/2.0
    b_y = np.zeros(2*N)
    b_nx = np.zeros(2*N)
    b_ny = np.full(2*N, -1.0)
    b_len = np.full(2*N, b_l)
    b_bc = np.full(2*N, 1.0)

    # Right side: x=2, y in [0,1], Dirichlet BC (u=cos(pi*y) -> bc_type=0)
    r_l = 1.0 / N
    r_x = np.full(N, 2.0)
    r_y = np.linspace(0, 1, N, endpoint=False) + r_l/2.0
    r_nx = np.full(N, 1.0)
    r_ny = np.zeros(N)
    r_len = np.full(N, r_l)
    r_bc = np.zeros(N)

    # Top side: y=1, x in [2,0] (reverse direction), Neumann BC (q=0 -> bc_type=1)
    t_l = 2.0 / (2*N)
    t_x = np.linspace(2, 0, 2*N, endpoint=False) - t_l/2.0
    t_y = np.full(2*N, 1.0)
    t_nx = np.zeros(2*N)
    t_ny = np.full(2*N, 1.0)
    t_len = np.full(2*N, t_l)
    t_bc = np.full(2*N, 1.0)

    # Left side: x=0, y in [1,0] (reverse direction), Dirichlet BC (u=0 -> bc_type=0)
    l_l = 1.0 / N
    l_x = np.zeros(N)
    l_y = np.linspace(1, 0, N, endpoint=False) - l_l/2.0
    l_nx = np.full(N, -1.0)
    l_ny = np.zeros(N)
    l_len = np.full(N, l_l)
    l_bc = np.zeros(N)

    mid_x = np.concatenate([b_x, r_x, t_x, l_x])
    mid_y = np.concatenate([b_y, r_y, t_y, l_y])
    norm_x = np.concatenate([b_nx, r_nx, t_nx, l_nx])
    norm_y = np.concatenate([b_ny, r_ny, t_ny, l_ny])
    lengths = np.concatenate([b_len, r_len, t_len, l_len])
    bc_type = np.concatenate([b_bc, r_bc, t_bc, l_bc])

    N_tot = len(mid_x) # Total 6*N DOFs

    u_known = np.zeros(N_tot)
    q_known = np.zeros(N_tot)
    for i in range(N_tot):
        u_known[i] = exact_u(mid_x[i], mid_y[i])
        q_known[i] = exact_q(mid_x[i], mid_y[i], norm_x[i], norm_y[i])

    pts, wts = get_gauss_points()

    rhs = build_rhs(u_known, q_known, mid_x, mid_y, norm_x, norm_y, lengths, bc_type, pts, wts)

    t_setup = time.time() - t_start

    # Matrix-free Solve
    t_solve_start = time.time()

    def matvec(v):
        return bem_matvec(v, mid_x, mid_y, norm_x, norm_y, lengths, bc_type, pts, wts)

    LO = LinearOperator((N_tot, N_tot), matvec=matvec)
    counter = IterationCounter()

    sol, info = gmres(LO, rhs, callback=counter, rtol=1e-12, maxiter=N_tot, callback_type='legacy')

    t_solve = time.time() - t_solve_start

    u_all = np.zeros(N_tot)
    q_all = np.zeros(N_tot)
    for j in range(N_tot):
        if bc_type[j] == 0:
            q_all[j] = sol[j]
            u_all[j] = u_known[j]
        else:
            u_all[j] = sol[j]
            q_all[j] = q_known[j]

    # Interior Evaluation
    t_eval_start = time.time()
    ngrid_x, ngrid_y = 80, 40
    _grid_x = np.linspace(0.01, 1.99, ngrid_x)
    _grid_y = np.linspace(0.01, 0.99, ngrid_y)
    gx, gy = np.meshgrid(_grid_x, _grid_y)
    ix_pts = gx.flatten()
    iy_pts = gy.flatten()

    u_interior = eval_interior(ix_pts, iy_pts, u_all, q_all, mid_x, mid_y, norm_x, norm_y, lengths, pts, wts)

    u_exact_int = np.array([exact_u(px, py) for px, py in zip(ix_pts, iy_pts)])
    rel_l2 = np.linalg.norm(u_interior - u_exact_int) / np.linalg.norm(u_exact_int)

    t_eval = time.time() - t_eval_start

    return {
        "N": N,
        "DOFs": N_tot,
        "iters": counter.niter,
        "error": rel_l2,
        "t_setup": t_setup,
        "t_solve": t_solve,
        "t_eval": t_eval,
        "t_total": t_setup + t_solve + t_eval
    }

if __name__ == "__main__":
    N_values = [160, 320, 640]

    # Dummy run to compile Numba JIT functions for accurate timing
    _ = solve_bem(5)

    print(f"{'N':>5} | {'DOFs':>6} | {'Asm(s)':>8} | {'Solve(s)':>8} | {'Eval(s)':>8} | {'Total(s)':>8} | {'RelL2':>12} | {'Rate':>8}")
    print("-" * 85)

    results = []
    for n in N_values:
        res = solve_bem(n)
        results.append(res)

    for i, res in enumerate(results):
        if i == 0:
            rate_str = "     ---"
        else:
            prev = results[i-1]
            rate = np.log(prev['error'] / res['error']) / np.log(res['DOFs'] / prev['DOFs'])
            rate_str = f"{rate:8.2f}"

        print(f"{res['N']:5d} | {res['DOFs']:6d} | {res['t_setup']:8.3f} | {res['t_solve']:8.3f} | {res['t_eval']:8.3f} | {res['t_total']:8.3f} | {res['error']:12.6e} | {rate_str}")

    if len(results) > 1:
        overall_rate = np.log(results[0]['error'] / results[-1]['error']) / np.log(results[-1]['DOFs'] / results[0]['DOFs'])
        print("-" * 85)
        print(f"Overall Convergence Order: {overall_rate:.2f}")