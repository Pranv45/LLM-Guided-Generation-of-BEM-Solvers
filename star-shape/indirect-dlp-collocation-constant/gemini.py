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
def r_func(theta):
    return 1.0 + 0.3 * np.cos(5.0 * theta)

@njit(cache=True)
def r_prime(theta):
    return -1.5 * np.sin(5.0 * theta)

@njit(cache=True)
def exact_u(x, y):
    return x**3 - 3.0 * x * y**2

@njit(parallel=True, fastmath=True, cache=True)
def bem_matvec(mu, theta_mid, dtheta, pts, wts):
    N = len(mu)
    out = np.zeros(N)
    for i in prange(N):
        val = 0.0
        ti = theta_mid[i]
        ri = r_func(ti)
        xi = ri * np.cos(ti)
        yi = ri * np.sin(ti)

        for j in range(N):
            if i == j:
                continue

            tj_mid = theta_mid[j]
            hij = 0.0
            for k in range(len(pts)):
                xi_q = pts[k]
                w = wts[k]
                t_q = tj_mid + xi_q * dtheta / 2.0

                r_q = r_func(t_q)
                rp_q = r_prime(t_q)

                x_q = r_q * np.cos(t_q)
                y_q = r_q * np.sin(t_q)

                xp_q = rp_q * np.cos(t_q) - r_q * np.sin(t_q)
                yp_q = rp_q * np.sin(t_q) + r_q * np.cos(t_q)

                dx = x_q - xi
                dy = y_q - yi
                r2 = dx*dx + dy*dy

                kernel = (dx * yp_q - dy * xp_q) / (2.0 * np.pi * r2)
                hij += kernel * w

            hij *= (dtheta / 2.0)
            val += hij * (mu[j] - mu[i])

        out[i] = mu[i] + val
    return out

@njit(parallel=True, fastmath=True, cache=True)
def eval_interior(ix_pts, iy_pts, mu, theta_mid, dtheta, pts, wts):
    M = len(ix_pts)
    N = len(mu)
    u_int = np.zeros(M)
    for i in prange(M):
        val = 0.0
        xi = ix_pts[i]
        yi = iy_pts[i]

        for j in range(N):
            tj_mid = theta_mid[j]
            hij = 0.0
            for k in range(len(pts)):
                xi_q = pts[k]
                w = wts[k]
                t_q = tj_mid + xi_q * dtheta / 2.0

                r_q = r_func(t_q)
                rp_q = r_prime(t_q)

                x_q = r_q * np.cos(t_q)
                y_q = r_q * np.sin(t_q)

                xp_q = rp_q * np.cos(t_q) - r_q * np.sin(t_q)
                yp_q = rp_q * np.sin(t_q) + r_q * np.cos(t_q)

                dx = x_q - xi
                dy = y_q - yi
                r2 = dx*dx + dy*dy

                kernel = (dx * yp_q - dy * xp_q) / (2.0 * np.pi * r2)
                hij += kernel * w

            hij *= (dtheta / 2.0)
            val += hij * mu[j]
        u_int[i] = val
    return u_int

class IterationCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, pr_norm):
        self.niter += 1

def solve_bem(N):
    t_start = time.time()

    dtheta = 2.0 * np.pi / N
    theta_mid = np.array([(i + 0.5) * dtheta for i in range(N)])

    pts, wts = get_gauss_points()

    f = np.zeros(N)
    for i in range(N):
        r_i = r_func(theta_mid[i])
        x_i = r_i * np.cos(theta_mid[i])
        y_i = r_i * np.sin(theta_mid[i])
        f[i] = exact_u(x_i, y_i)

    t_setup = time.time() - t_start

    t_solve_start = time.time()
    def matvec(v):
        return bem_matvec(v, theta_mid, dtheta, pts, wts)

    LO = LinearOperator((N, N), matvec=matvec)
    counter = IterationCounter()

    sol, info = gmres(LO, f, callback=counter, rtol=1e-12, atol=1e-12, maxiter=N, callback_type='pr_norm')
    t_solve = time.time() - t_solve_start

    t_eval_start = time.time()
    ngrid = 60
    _grid = np.linspace(-1.5, 1.5, ngrid)
    gx, gy = np.meshgrid(_grid, _grid)
    gx = gx.flatten()
    gy = gy.flatten()

    r_val = np.sqrt(gx**2 + gy**2)
    theta_val = np.arctan2(gy, gx)
    r_bound = r_func(theta_val)

    mask = r_val < r_bound - 0.1
    ix_pts = gx[mask]
    iy_pts = gy[mask]

    u_interior = eval_interior(ix_pts, iy_pts, sol, theta_mid, dtheta, pts, wts)
    u_exact_int = np.array([exact_u(px, py) for px, py in zip(ix_pts, iy_pts)])

    rel_l2 = np.linalg.norm(u_interior - u_exact_int) / np.linalg.norm(u_exact_int)
    t_eval = time.time() - t_eval_start

    return {
        "N": N,
        "Unknowns": N,
        "GMRES": counter.niter,
        "Error": rel_l2,
        "Setup": t_setup,
        "Solve": t_solve,
        "Eval": t_eval,
        "Total": t_setup + t_solve + t_eval
    }

if __name__ == "__main__":
    _ = solve_bem(10)

    N_values = [320, 640, 1280, 2560, 5120]
    results = []

    print(f"{'N':<5} {'Unknowns':<10} {'GMRES':<7} {'Rel L2 Error':<13} {'Conv Rate':<10} {'Setup':<9} {'Solve':<9} {'Eval':<9} {'Total':<9}")
    print("-" * 90)

    for n in N_values:
        res = solve_bem(n)
        results.append(res)

    for i, res in enumerate(results):
        if i == 0:
            rate_str = "      ---"
        else:
            prev = results[i-1]
            rate = np.log(prev['Error'] / res['Error']) / np.log(res['N'] / prev['N'])
            rate_str = f"{rate:9.2f}"

        print(f"{res['N']:<5d} {res['Unknowns']:<10d} {res['GMRES']:<7d} {res['Error']:13.6e} {rate_str:10s} {res['Setup']:<9.3f} {res['Solve']:<9.3f} {res['Eval']:<9.3f} {res['Total']:<9.3f}")

    if len(results) > 1:
        log_h = np.log([1.0 / res['N'] for res in results])
        log_err = np.log([res['Error'] for res in results])
        A = np.vstack([log_h, np.ones(len(log_h))]).T
        m, c = np.linalg.lstsq(A, log_err, rcond=None)[0]
        print("-" * 90)
        print(f"Final Convergence Order: {m:.2f}")