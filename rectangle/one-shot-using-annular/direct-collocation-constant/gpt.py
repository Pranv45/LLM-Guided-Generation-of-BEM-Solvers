# Single Jupyter notebook cell
# Rectangle BEM solver (preserving methodology from the annulus reference implementation)
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
    A = np.sinh(np.pi * x) / np.sinh(2.0 * np.pi)
    Ax = (np.pi * np.cosh(np.pi * x)) / np.sinh(2.0 * np.pi)
    uy = - np.pi * A * np.sin(np.pi * y)
    ux = Ax * np.cos(np.pi * y)
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


def solve_bem(N_per_side):

    t_start = time.time()

    N = N_per_side

    ys_v = (np.arange(N) + 0.5) * (1.0 / N)

    xs_left = np.zeros(N)
    ys_left = ys_v.copy()
    nx_left = np.full(N, -1.0)
    ny_left = np.zeros(N)
    len_left = np.full(N, 1.0 / N)

    xs_right = np.full(N, 2.0)
    ys_right = ys_v.copy()
    nx_right = np.full(N, 1.0)
    ny_right = np.zeros(N)
    len_right = np.full(N, 1.0 / N)

    xs_bot = (np.arange(N) + 0.5) * (2.0 / N)
    ys_bot = np.zeros(N)
    nx_bot = np.zeros(N)
    ny_bot = np.full(N, -1.0)
    len_bot = np.full(N, 2.0 / N)

    xs_top = (np.arange(N) + 0.5) * (2.0 / N)
    ys_top = np.full(N, 1.0)
    nx_top = np.zeros(N)
    ny_top = np.full(N, 1.0)
    len_top = np.full(N, 2.0 / N)

    mid_x = np.concatenate([xs_left, xs_right, xs_bot, xs_top])
    mid_y = np.concatenate([ys_left, ys_right, ys_bot, ys_top])
    norm_x = np.concatenate([nx_left, nx_right, nx_bot, nx_top])
    norm_y = np.concatenate([ny_left, ny_right, ny_bot, ny_top])
    lengths = np.concatenate([len_left, len_right, len_bot, len_top])

    N_tot = len(mid_x)

    bc_type = np.concatenate([np.zeros(N), np.zeros(N), np.ones(N), np.ones(N)])

    u_known = np.zeros(N_tot)
    q_known = np.zeros(N_tot)

    for i in range(N_tot):
        u_known[i] = exact_u(mid_x[i], mid_y[i])
        q_known[i] = exact_q(mid_x[i], mid_y[i], norm_x[i], norm_y[i])

    pts, wts = get_gauss_points()

    rhs = build_rhs(u_known, q_known, mid_x, mid_y, norm_x, norm_y, lengths, bc_type, pts, wts)

    t_setup = time.time() - t_start

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

    # ============================================================
    # ONLY MODIFICATION: evaluation grid replaced with requested grid
    # ============================================================

    t_eval_start = time.time()

    ngrid_x = 80
    ngrid_y = 40

    _gx = np.linspace(0.01, 1.99, ngrid_x)
    _gy = np.linspace(0.01, 0.99, ngrid_y)

    gx, gy = np.meshgrid(_gx, _gy)

    gx = gx.flatten()
    gy = gy.flatten()

    ix_pts = gx
    iy_pts = gy

    u_interior = eval_interior(ix_pts, iy_pts, u_all, q_all, mid_x, mid_y, norm_x, norm_y, lengths, pts, wts)

    u_exact_int = np.array([exact_u(px, py) for px, py in zip(ix_pts, iy_pts)])

    rel_l2 = np.linalg.norm(u_interior - u_exact_int) / np.linalg.norm(u_exact_int)

    t_eval = time.time() - t_eval_start

    return {
        "N_per_side": N,
        "DOFs": N_tot,
        "iters": counter.niter,
        "error": rel_l2,
        "t_setup": t_setup,
        "t_solve": t_solve,
        "t_eval": t_eval,
        "t_total": t_setup + t_solve + t_eval
    }


if __name__ == "__main__":

    N_values = [40, 80, 160, 320, 640]

    _ = solve_bem(4)

    header = f"{'N':>5} | {'DOFs':>5} | {'Iters':>6} | {'Asm(s)':>8} | {'Solve(s)':>8} | {'Eval(s)':>8} | {'Total(s)':>8} | {'RelL2':>12}"
    print(header)
    print("-" * len(header))

    results = []

    for N in N_values:

        res = solve_bem(N)
        results.append(res)

        print(f"{res['N_per_side']:5d} | {res['DOFs']:5d} | {res['iters']:6d} | {res['t_setup']:8.3f} | {res['t_solve']:8.3f} | {res['t_eval']:8.3f} | {res['t_total']:8.3f} | {res['error']:12.6e}")

    dofs = np.array([r['DOFs'] for r in results])
    errs = np.array([r['error'] for r in results])

    valid = (errs > 0)

    if np.sum(valid) >= 2:

        p = np.polyfit(np.log(dofs[valid]), np.log(errs[valid]), 1)
        slope = p[0]

        print("-" * len(header))
        print(f"Observed convergence slope (error vs DOFs): {slope:.3f} (power law: error ~ DOFs^{slope:.3f})")

    else:

        print("Not enough valid error data to compute convergence slope.")

