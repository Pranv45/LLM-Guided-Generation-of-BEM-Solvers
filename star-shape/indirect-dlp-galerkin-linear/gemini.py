import numpy as np
import scipy.sparse.linalg as spla
from numba import njit, prange
import time
import warnings

warnings.filterwarnings('ignore')

@njit(fastmath=True)
def r_func(theta):
    return 1.0 + 0.3 * np.cos(5.0 * theta)

@njit(fastmath=True)
def u_exact(x, y):
    return x**3 - 3.0 * x * y**2

@njit(fastmath=True)
def curve_props(theta):
    r = 1.0 + 0.3 * np.cos(5.0 * theta)
    rp = -1.5 * np.sin(5.0 * theta)
    rpp = -7.5 * np.cos(5.0 * theta)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    xp = rp * np.cos(theta) - r * np.sin(theta)
    yp = rp * np.sin(theta) + r * np.cos(theta)

    xpp = rpp * np.cos(theta) - 2.0 * rp * np.sin(theta) - r * np.cos(theta)
    ypp = rpp * np.sin(theta) + 2.0 * rp * np.cos(theta) - r * np.sin(theta)

    J = np.sqrt(xp**2 + yp**2)
    nx = yp / J
    ny = -xp / J
    kappa = (xp * ypp - yp * xpp) / (J**3)

    return x, y, nx, ny, J, kappa

def build_interior_grid():
    ngrid = 60
    _grid = np.linspace(-1.5, 1.5, ngrid)
    gx, gy = np.meshgrid(_grid, _grid)

    gx = gx.flatten()
    gy = gy.flatten()

    r_val = np.sqrt(gx**2 + gy**2)
    theta_val = np.arctan2(gy, gx)
    r_bound = r_func(theta_val)

    mask = r_val < r_bound - 0.1

    return gx[mask], gy[mask]

@njit(parallel=True, fastmath=True)
def precompute_geometry(N, dtheta, gauss_nodes, gauss_weights):
    """Precomputes exact geometry at all quadrature points to eliminate trig overhead."""
    Nq = len(gauss_nodes)
    px_q = np.zeros((N, Nq))
    py_q = np.zeros((N, Nq))
    nx_q = np.zeros((N, Nq))
    ny_q = np.zeros((N, Nq))
    w_scaled_q = np.zeros((N, Nq))
    kappa_q = np.zeros((N, Nq))

    phi1 = 0.5 * (1.0 - gauss_nodes)
    phi2 = 0.5 * (1.0 + gauss_nodes)

    for e in prange(N):
        th1 = e * dtheta
        th2 = (e + 1) * dtheta
        for q in range(Nq):
            th = th1 * phi1[q] + th2 * phi2[q]
            x, y, nx, ny, J, kappa = curve_props(th)
            px_q[e, q] = x
            py_q[e, q] = y
            nx_q[e, q] = nx
            ny_q[e, q] = ny
            w_scaled_q[e, q] = gauss_weights[q] * J * dtheta / 2.0
            kappa_q[e, q] = kappa

    return px_q, py_q, nx_q, ny_q, w_scaled_q, kappa_q

@njit(parallel=True, fastmath=True)
def compute_matvec(v, gauss_nodes, px_q, py_q, nx_q, ny_q, w_scaled_q, kappa_q):
    N, Nq = px_q.shape
    out_elem = np.zeros((N, 2))

    phi1 = 0.5 * (1.0 - gauss_nodes)
    phi2 = 0.5 * (1.0 + gauss_nodes)

    for e_x in prange(N):
        out_i = 0.0
        out_j = 0.0

        v_x1 = v[e_x]
        v_x2 = v[(e_x + 1) % N]

        for q_x in range(Nq):
            px = px_q[e_x, q_x]
            py = py_q[e_x, q_x]
            wx = w_scaled_q[e_x, q_x]

            v_x = v_x1 * phi1[q_x] + v_x2 * phi2[q_x]
            val_x = -0.5 * v_x

            K_val = 0.0
            for e_y in range(N):
                v_y1 = v[e_y]
                v_y2 = v[(e_y + 1) % N]

                for q_y in range(Nq):
                    mu_y = v_y1 * phi1[q_y] + v_y2 * phi2[q_y]
                    wy = w_scaled_q[e_y, q_y]

                    if e_x == e_y and q_x == q_y:
                        kernel = -kappa_q[e_x, q_x] / (4.0 * np.pi)
                    else:
                        rx = px - px_q[e_y, q_y]
                        ry = py - py_q[e_y, q_y]
                        r2 = rx**2 + ry**2
                        kernel = (rx * nx_q[e_y, q_y] + ry * ny_q[e_y, q_y]) / (2.0 * np.pi * r2)

                    K_val += kernel * mu_y * wy

            total_val = val_x + K_val
            out_i += total_val * phi1[q_x] * wx
            out_j += total_val * phi2[q_x] * wx

        out_elem[e_x, 0] = out_i
        out_elem[e_x, 1] = out_j

    return out_elem

def matvec(v, N, gauss_nodes, px_q, py_q, nx_q, ny_q, w_scaled_q, kappa_q):
    out_elem = compute_matvec(v, gauss_nodes, px_q, py_q, nx_q, ny_q, w_scaled_q, kappa_q)
    out = np.zeros(N)
    for e in range(N):
        out[e] += out_elem[e, 0]
        out[(e + 1) % N] += out_elem[e, 1]
    return out

@njit(fastmath=True)
def assemble_rhs(N, gauss_nodes, px_q, py_q, w_scaled_q):
    b = np.zeros(N)
    Nq = len(gauss_nodes)
    phi1 = 0.5 * (1.0 - gauss_nodes)
    phi2 = 0.5 * (1.0 + gauss_nodes)

    for e_x in range(N):
        for q_x in range(Nq):
            px = px_q[e_x, q_x]
            py = py_q[e_x, q_x]
            wx = w_scaled_q[e_x, q_x]

            val = u_exact(px, py)

            b[e_x] += val * phi1[q_x] * wx
            b[(e_x + 1) % N] += val * phi2[q_x] * wx
    return b

@njit(parallel=True, fastmath=True)
def evaluate_interior(gx, gy, mu, gauss_nodes, px_q, py_q, nx_q, ny_q, w_scaled_q):
    N_int = len(gx)
    N, Nq = px_q.shape
    u_int = np.zeros(N_int)

    phi1 = 0.5 * (1.0 - gauss_nodes)
    phi2 = 0.5 * (1.0 + gauss_nodes)

    for i in prange(N_int):
        px = gx[i]
        py = gy[i]
        val = 0.0

        for e_y in range(N):
            mu_1 = mu[e_y]
            mu_2 = mu[(e_y + 1) % N]

            for q_y in range(Nq):
                wy = w_scaled_q[e_y, q_y]
                mu_val = mu_1 * phi1[q_y] + mu_2 * phi2[q_y]

                rx = px - px_q[e_y, q_y]
                ry = py - py_q[e_y, q_y]
                r2 = rx**2 + ry**2

                kernel = (rx * nx_q[e_y, q_y] + ry * ny_q[e_y, q_y]) / (2.0 * np.pi * r2)
                val += kernel * mu_val * wy

        u_int[i] = val
    return u_int

def main():
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(12)

    gx, gy = build_interior_grid()
    u_ex_int = u_exact(gx, gy)

    N_values = [1280, 2560, 5120]
    errors = []
    hs = []

    print(f"{'N':<5} {'Unknowns':<10} {'GMRES':<8} {'Rel L2 Error':<14} {'Conv Rate':<10} {'Setup':<10} {'Solve':<10} {'Eval':<10} {'Total':<10}")
    print("-" * 105)

    for i, N in enumerate(N_values):
        t_start = time.time()
        dtheta = 2.0 * np.pi / N

        t0 = time.time()

        # Geometry Precomputation
        px_q, py_q, nx_q, ny_q, w_scaled_q, kappa_q = precompute_geometry(N, dtheta, gauss_nodes, gauss_weights)
        b = assemble_rhs(N, gauss_nodes, px_q, py_q, w_scaled_q)

        # Clean JIT compilation on first loop
        if i == 0:
            _ = compute_matvec(np.zeros(N), gauss_nodes, px_q, py_q, nx_q, ny_q, w_scaled_q, kappa_q)
            _ = evaluate_interior(np.array([0.0]), np.array([0.0]), np.zeros(N), gauss_nodes, px_q, py_q, nx_q, ny_q, w_scaled_q)

        t_setup = time.time() - t0

        t0 = time.time()
        iter_count = [0]
        def gmres_callback(pr_norm):
            iter_count[0] += 1

        A_op = spla.LinearOperator((N, N), matvec=lambda v: matvec(v, N, gauss_nodes, px_q, py_q, nx_q, ny_q, w_scaled_q, kappa_q))
        mu_sol, info = spla.gmres(A_op, b, rtol=1e-8, atol=1e-10, callback=gmres_callback, callback_type='pr_norm')
        t_solve = time.time() - t0

        t0 = time.time()
        u_num_int = evaluate_interior(gx, gy, mu_sol, gauss_nodes, px_q, py_q, nx_q, ny_q, w_scaled_q)
        t_eval = time.time() - t0

        error_l2 = np.linalg.norm(u_num_int - u_ex_int) / np.linalg.norm(u_ex_int)
        errors.append(error_l2)
        h = 2 * np.pi / N
        hs.append(h)

        if i == 0:
            rate_str = "---"
        else:
            rate = np.log(errors[i] / errors[i-1]) / np.log(hs[i] / hs[i-1])
            rate_str = f"{rate:.4f}"

        t_total = time.time() - t_start

        print(f"{N:<5} {N:<10} {iter_count[0]:<8} {error_l2:<14.4e} {rate_str:<10} {t_setup:<10.4f} {t_solve:<10.4f} {t_eval:<10.4f} {t_total:<10.4f}")

    log_h = np.log(hs)
    log_e = np.log(errors)
    slope, _ = np.polyfit(log_h, log_e, 1)

    print("-" * 105)
    print(f"Final Convergence Order: {slope:.4f}")

if __name__ == "__main__":
    main()