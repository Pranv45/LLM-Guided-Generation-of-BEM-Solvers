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
def bem_matvec(v, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts):
    N_tot = len(nodes_x)
    N = N_tot // 2
    out = np.zeros(N_tot)

    for i in prange(N_tot):
        row_val = 0.0
        if i >= N:
            # Domain is inside the outer N-gon, interior angle is pi - 2pi/N
            row_val -= (0.5 - 1.0 / N) * v[i]

        for e in range(N_tot):
            j1 = elems[e, 0]
            j2 = elems[e, 1]
            L = lengths[e]

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
                    w = wts[k]
                    N1 = 0.5 * (1.0 - xi)
                    N2 = 0.5 * (1.0 + xi)
                    yx = N1 * nodes_x[j1] + N2 * nodes_x[j2]
                    yy = N1 * nodes_y[j1] + N2 * nodes_y[j2]

                    rx = nodes_x[i] - yx
                    ry = nodes_y[i] - yy
                    r2 = rx*rx + ry*ry

                    G_val = -1.0 / (4.0 * np.pi) * np.log(r2)
                    H_val = 1.0 / (2.0 * np.pi * r2) * (rx * nx[e] + ry * ny[e])

                    wL2 = w * L * 0.5
                    G1 += G_val * N1 * wL2
                    G2 += G_val * N2 * wL2
                    H1 += H_val * N1 * wL2
                    H2 += H_val * N2 * wL2

            if j1 < N:
                row_val += G1 * v[j1]
            else:
                row_val -= H1 * v[j1]

            if j2 < N:
                row_val += G2 * v[j2]
            else:
                row_val -= H2 * v[j2]

        out[i] = row_val
    return out

@njit(parallel=True, fastmath=True, cache=True)
def build_rhs(u_known, q_known, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts):
    N_tot = len(nodes_x)
    N = N_tot // 2
    rhs = np.zeros(N_tot)

    for i in prange(N_tot):
        row_val = 0.0
        if i < N:
            # Domain is outside the inner N-gon, interior angle is pi + 2pi/N
            row_val += (0.5 + 1.0 / N) * u_known[i]

        for e in range(N_tot):
            j1 = elems[e, 0]
            j2 = elems[e, 1]
            L = lengths[e]

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
                    w = wts[k]
                    N1 = 0.5 * (1.0 - xi)
                    N2 = 0.5 * (1.0 + xi)
                    yx = N1 * nodes_x[j1] + N2 * nodes_x[j2]
                    yy = N1 * nodes_y[j1] + N2 * nodes_y[j2]

                    rx = nodes_x[i] - yx
                    ry = nodes_y[i] - yy
                    r2 = rx*rx + ry*ry

                    G_val = -1.0 / (4.0 * np.pi) * np.log(r2)
                    H_val = 1.0 / (2.0 * np.pi * r2) * (rx * nx[e] + ry * ny[e])

                    wL2 = w * L * 0.5
                    G1 += G_val * N1 * wL2
                    G2 += G_val * N2 * wL2
                    H1 += H_val * N1 * wL2
                    H2 += H_val * N2 * wL2

            if j1 < N:
                row_val += H1 * u_known[j1]
            else:
                row_val -= G1 * q_known[j1]

            if j2 < N:
                row_val += H2 * u_known[j2]
            else:
                row_val -= G2 * q_known[j2]

        rhs[i] = row_val
    return rhs

@njit(parallel=True, fastmath=True, cache=True)
def eval_interior(ix_pts, iy_pts, u_all, q_all, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts):
    M = len(ix_pts)
    N_tot = len(nodes_x)
    u_int = np.zeros(M)

    for i in prange(M):
        val = 0.0
        for e in range(N_tot):
            j1 = elems[e, 0]
            j2 = elems[e, 1]
            L = lengths[e]

            G1 = 0.0; G2 = 0.0; H1 = 0.0; H2 = 0.0
            for k in range(len(pts)):
                xi = pts[k]
                w = wts[k]
                N1 = 0.5 * (1.0 - xi)
                N2 = 0.5 * (1.0 + xi)
                yx = N1 * nodes_x[j1] + N2 * nodes_x[j2]
                yy = N1 * nodes_y[j1] + N2 * nodes_y[j2]

                rx = ix_pts[i] - yx
                ry = iy_pts[i] - yy
                r2 = rx*rx + ry*ry

                G_val = -1.0 / (4.0 * np.pi) * np.log(r2)
                H_val = 1.0 / (2.0 * np.pi * r2) * (rx * nx[e] + ry * ny[e])

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

def solve_bem(N):
    t_start = time.time()

    N_tot = 2 * N
    nodes_x = np.zeros(N_tot)
    nodes_y = np.zeros(N_tot)
    elems = np.zeros((N_tot, 2), dtype=np.int32)
    nx = np.zeros(N_tot)
    ny = np.zeros(N_tot)
    lengths = np.zeros(N_tot)

    theta_i = np.linspace(0, 2*np.pi, N, endpoint=False)

    for i in range(N):
        nodes_x[i] = np.cos(theta_i[i])
        nodes_y[i] = np.sin(theta_i[i])
        elems[i, 0] = i
        elems[i, 1] = (i + 1) % N

    for i in range(N):
        nodes_x[N + i] = 2.0 * np.cos(theta_i[i])
        nodes_y[N + i] = 2.0 * np.sin(theta_i[i])
        elems[N + i, 0] = N + i
        elems[N + i, 1] = N + ((i + 1) % N)

    for e in range(N_tot):
        j1 = elems[e, 0]
        j2 = elems[e, 1]
        dx = nodes_x[j2] - nodes_x[j1]
        dy = nodes_y[j2] - nodes_y[j1]
        L = np.sqrt(dx**2 + dy**2)
        lengths[e] = L
        if e < N:
            nx[e] = -dy / L
            ny[e] = dx / L
        else:
            nx[e] = dy / L
            ny[e] = -dx / L

    u_known = np.zeros(N_tot)
    q_known = np.zeros(N_tot)
    for i in range(N_tot):
        u_known[i] = exact_u(nodes_x[i], nodes_y[i])
        if i < N:
            nnx = -nodes_x[i]
            nny = -nodes_y[i]
        else:
            nnx = nodes_x[i] / 2.0
            nny = nodes_y[i] / 2.0
        q_known[i] = exact_q(nodes_x[i], nodes_y[i], nnx, nny)

    pts, wts = get_gauss_points()

    rhs = build_rhs(u_known, q_known, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts)
    t_setup = time.time() - t_start

    t_solve_start = time.time()
    def matvec(v):
        return bem_matvec(v, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts)

    LO = LinearOperator((N_tot, N_tot), matvec=matvec)
    counter = IterationCounter()
    sol, info = gmres(LO, rhs, callback=counter, rtol=1e-12, maxiter=N_tot, callback_type='legacy')
    t_solve = time.time() - t_solve_start

    u_all = np.zeros(N_tot)
    q_all = np.zeros(N_tot)
    for i in range(N_tot):
        if i < N:
            u_all[i] = u_known[i]
            q_all[i] = sol[i]
        else:
            q_all[i] = q_known[i]
            u_all[i] = sol[i]

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

    u_interior = eval_interior(ix_pts, iy_pts, u_all, q_all, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts)
    u_exact_int = np.array([exact_u(px, py) for px, py in zip(ix_pts, iy_pts)])
    rel_l2 = np.linalg.norm(u_interior - u_exact_int) / np.linalg.norm(u_exact_int)
    t_eval = time.time() - t_eval_start

    return {
        "N": N,
        "tot_N": N_tot,
        "iters": counter.niter,
        "error": rel_l2,
        "t_setup": t_setup,
        "t_solve": t_solve,
        "t_eval": t_eval,
        "t_total": t_setup + t_solve + t_eval
    }

if __name__ == "__main__":
    _ = solve_bem(10)

    N_values = [160, 320, 640, 1280, 2560]
    results = []

    print(f"{'N':>5} | {'Iters':>6} | {'L2 Error':>12} | {'Rate':>8} | {'Setup(s)':>8} | {'Solve(s)':>8} | {'Eval(s)':>8} | {'Total(s)':>8}")
    print("-" * 90)

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