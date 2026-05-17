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
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (r**3 + r**(-3)) * np.cos(3 * theta)

@njit(cache=True)
def exact_q(x, y, nx, ny):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    dudr = (3 * r**2 - 3 * r**(-4)) * np.cos(3 * theta)
    dudt = -3 * (r**3 + r**(-3)) * np.sin(3 * theta)
    ux = dudr * np.cos(theta) - (1.0/r) * dudt * np.sin(theta)
    uy = dudr * np.sin(theta) + (1.0/r) * dudt * np.cos(theta)
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

    # Boundary Discretization
    theta_i = np.linspace(0, 2*np.pi, N, endpoint=False)
    dtheta = 2*np.pi / N
    m_theta = theta_i + dtheta/2.0

    # Inner (r=1)
    ix = 1.0 * np.cos(m_theta)
    iy = 1.0 * np.sin(m_theta)
    inx = -np.cos(m_theta)
    iny = -np.sin(m_theta)
    il = np.full(N, 1.0 * dtheta)

    # Outer (r=2)
    ox = 2.0 * np.cos(m_theta)
    oy = 2.0 * np.sin(m_theta)
    onx = np.cos(m_theta)
    ony = np.sin(m_theta)
    ol = np.full(N, 2.0 * dtheta)

    mid_x = np.concatenate([ix, ox])
    mid_y = np.concatenate([iy, oy])
    norm_x = np.concatenate([inx, onx])
    norm_y = np.concatenate([iny, ony])
    lengths = np.concatenate([il, ol])

    bc_type = np.concatenate([np.zeros(N), np.ones(N)])

    u_known = np.zeros(2*N)
    q_known = np.zeros(2*N)
    for i in range(2*N):
        u_known[i] = exact_u(mid_x[i], mid_y[i])
        q_known[i] = exact_q(mid_x[i], mid_y[i], norm_x[i], norm_y[i])

    pts, wts = get_gauss_points()

    rhs = build_rhs(u_known, q_known, mid_x, mid_y, norm_x, norm_y, lengths, bc_type, pts, wts)

    t_setup = time.time() - t_start

    # Matrix-free Solve
    t_solve_start = time.time()

    def matvec(v):
        return bem_matvec(v, mid_x, mid_y, norm_x, norm_y, lengths, bc_type, pts, wts)

    LO = LinearOperator((2*N, 2*N), matvec=matvec)
    counter = IterationCounter()

    sol, info = gmres(LO, rhs, callback=counter, rtol=1e-12, maxiter=2*N, callback_type= 'legacy')

    t_solve = time.time() - t_solve_start

    u_all = np.zeros(2*N)
    q_all = np.zeros(2*N)
    for j in range(2*N):
        if bc_type[j] == 0:
            q_all[j] = sol[j]
            u_all[j] = u_known[j]
        else:
            u_all[j] = sol[j]
            q_all[j] = q_known[j]

    # Interior Evaluation
    t_eval_start = time.time()
    ngrid = 60
    _grid = np.linspace(-1.9, 1.9, ngrid)
    gx, gy = np.meshgrid(_grid, _grid)
    gx = gx.flatten()
    gy = gy.flatten()

    r_val = np.sqrt(gx**2 + gy**2)
    mask = (r_val > 1.0) & (r_val < 2.0)
    ix_pts = gx[mask]
    iy_pts = gy[mask]

    u_interior = eval_interior(ix_pts, iy_pts, u_all, q_all, mid_x, mid_y, norm_x, norm_y, lengths, pts, wts)

    u_exact_int = np.array([exact_u(px, py) for px, py in zip(ix_pts, iy_pts)])
    rel_l2 = np.linalg.norm(u_interior - u_exact_int) / np.linalg.norm(u_exact_int)

    t_eval = time.time() - t_eval_start

    return {
        "N": 2*N,
        "iters": counter.niter,
        "error": rel_l2,
        "t_setup": t_setup,
        "t_solve": t_solve,
        "t_eval": t_eval,
        "t_total": t_setup + t_solve + t_eval
    }

if __name__ == "__main__":
    N_values = [160, 320, 640, 1280, 2560]

    # Dummy run to compile Numba JIT functions for accurate timing
    _ = solve_bem(10)

    print(f"{'N':>5} | {'Iters':>6} | {'L2 Error':>12} | {'Rate':>8} | {'Setup(s)':>8} | {'Solve(s)':>8} | {'Eval(s)':>8} | {'Total(s)':>8}")
    print("-" * 90)

    results = []
    for n in N_values:
        res = solve_bem(n)
        results.append(res)

    for i, res in enumerate(results):
        if i == 0:
            rate_str = "     ---"
        else:
            prev = results[i-1]
            rate = np.log(prev['error'] / res['error']) / np.log(res['N'] / prev['N'])
            rate_str = f"{rate:8.2f}"

        print(f"{res['N']:5d} | {res['iters']:6d} | {res['error']:12.6e} | {rate_str} | {res['t_setup']:8.3f} | {res['t_solve']:8.3f} | {res['t_eval']:8.3f} | {res['t_total']:8.3f}")

    if len(results) > 1:
        overall_rate = np.log(results[0]['error'] / results[-1]['error']) / np.log(results[-1]['N'] / results[0]['N'])
        print("-" * 90)
        print(f"Overall Convergence Order: {overall_rate:.2f}")