import numpy as np
import time
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange


@njit(cache=False)
def get_gauss_points():
    pts = np.array([
        -0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
         0.1834346424956498,  0.5255324099163290,  0.7966664774136267,  0.9602898564975363
    ])
    wts = np.array([
         0.1012285362903763,  0.2223810344533745,  0.3137066458778873,  0.3626837833783620,
         0.3626837833783620,  0.3137066458778873,  0.2223810344533745,  0.1012285362903763
    ])
    return pts, wts


@njit(cache=False)
def exact_u(x, y):
    return (np.sinh(np.pi * x) / np.sinh(2.0 * np.pi)) * np.cos(np.pi * y)


@njit(cache=False)
def exact_q(x, y, nx, ny):
    ux = (np.pi * np.cosh(np.pi * x) / np.sinh(2.0 * np.pi)) * np.cos(np.pi * y)
    uy = (-np.pi * np.sinh(np.pi * x) / np.sinh(2.0 * np.pi)) * np.sin(np.pi * y)
    return ux * nx + uy * ny


@njit(parallel=True, fastmath=True, cache=False)
def bem_matvec(v, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts, is_dirichlet, ccoeff):
    N_tot = len(nodes_x)
    out = np.zeros(N_tot)

    for i in prange(N_tot):
        row_val = 0.0

        # Direct BEM convention used in the reference code:
        #   G q - H u - c u = 0
        # Therefore the c-term appears with a minus sign on Neumann rows
        # (rows whose unknown is u).
        if not is_dirichlet[i]:
            row_val -= ccoeff[i] * v[i]

        for e in range(N_tot):
            j1 = elems[e, 0]
            j2 = elems[e, 1]
            L = lengths[e]

            # Endpoint singular treatment, identical to the reference implementation
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
                G1 = 0.0
                G2 = 0.0
                H1 = 0.0
                H2 = 0.0
                for k in range(len(pts)):
                    xi = pts[k]
                    w = wts[k]
                    N1 = 0.5 * (1.0 - xi)
                    N2 = 0.5 * (1.0 + xi)

                    yx = N1 * nodes_x[j1] + N2 * nodes_x[j2]
                    yy = N1 * nodes_y[j1] + N2 * nodes_y[j2]

                    rx = nodes_x[i] - yx
                    ry = nodes_y[i] - yy
                    r2 = rx * rx + ry * ry

                    G_val = -1.0 / (4.0 * np.pi) * np.log(r2)
                    H_val = 1.0 / (2.0 * np.pi * r2) * (rx * nx[e] + ry * ny[e])

                    wL2 = w * L * 0.5
                    G1 += G_val * N1 * wL2
                    G2 += G_val * N2 * wL2
                    H1 += H_val * N1 * wL2
                    H2 += H_val * N2 * wL2

            # Unknown on Dirichlet nodes is q; unknown on Neumann nodes is u
            if is_dirichlet[j1]:
                row_val += G1 * v[j1]
            else:
                row_val -= H1 * v[j1]

            if is_dirichlet[j2]:
                row_val += G2 * v[j2]
            else:
                row_val -= H2 * v[j2]

        out[i] = row_val

    return out


@njit(parallel=True, fastmath=True, cache=False)
def build_rhs(u_known, q_known, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts, is_dirichlet, ccoeff):
    N_tot = len(nodes_x)
    rhs = np.zeros(N_tot)

    for i in prange(N_tot):
        row_val = 0.0

        # For Dirichlet rows, the c-term is known and moves to the RHS
        if is_dirichlet[i]:
            row_val += ccoeff[i] * u_known[i]

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
                G1 = 0.0
                G2 = 0.0
                H1 = 0.0
                H2 = 0.0
                for k in range(len(pts)):
                    xi = pts[k]
                    w = wts[k]
                    N1 = 0.5 * (1.0 - xi)
                    N2 = 0.5 * (1.0 + xi)

                    yx = N1 * nodes_x[j1] + N2 * nodes_x[j2]
                    yy = N1 * nodes_y[j1] + N2 * nodes_y[j2]

                    rx = nodes_x[i] - yx
                    ry = nodes_y[i] - yy
                    r2 = rx * rx + ry * ry

                    G_val = -1.0 / (4.0 * np.pi) * np.log(r2)
                    H_val = 1.0 / (2.0 * np.pi * r2) * (rx * nx[e] + ry * ny[e])

                    wL2 = w * L * 0.5
                    G1 += G_val * N1 * wL2
                    G2 += G_val * N2 * wL2
                    H1 += H_val * N1 * wL2
                    H2 += H_val * N2 * wL2

            if is_dirichlet[j1]:
                row_val += H1 * u_known[j1]
            else:
                row_val -= G1 * q_known[j1]

            if is_dirichlet[j2]:
                row_val += H2 * u_known[j2]
            else:
                row_val -= G2 * q_known[j2]

        rhs[i] = row_val

    return rhs


@njit(parallel=True, fastmath=True, cache=False)
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

            G1 = 0.0
            G2 = 0.0
            H1 = 0.0
            H2 = 0.0
            for k in range(len(pts)):
                xi = pts[k]
                w = wts[k]
                N1 = 0.5 * (1.0 - xi)
                N2 = 0.5 * (1.0 + xi)

                yx = N1 * nodes_x[j1] + N2 * nodes_x[j2]
                yy = N1 * nodes_y[j1] + N2 * nodes_y[j2]

                rx = ix_pts[i] - yx
                ry = iy_pts[i] - yy
                r2 = rx * rx + ry * ry

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
    """
    N = number of panels on each side of the rectangle.
    Total boundary elements / nodes = 4N.
    """
    t_start = time.time()

    N_tot = 4 * N
    nodes_x = np.zeros(N_tot)
    nodes_y = np.zeros(N_tot)
    elems = np.zeros((N_tot, 2), dtype=np.int32)
    nx = np.zeros(N_tot)
    ny = np.zeros(N_tot)
    lengths = np.zeros(N_tot)
    is_dirichlet = np.zeros(N_tot, dtype=np.bool_)
    ccoeff = np.full(N_tot, 0.5)

    # Boundary node ordering is counter-clockwise:
    # bottom: (0,0) -> (2,0)
    # right : (2,0) -> (2,1)
    # top   : (2,1) -> (0,1)
    # left  : (0,1) -> (0,0)
    #
    # Corners are assigned to one adjacent side only (no duplicate nodes),
    # which keeps the discretization compact and consistent.

    # Bottom side: Neumann q = du/dn = 0
    for i in range(N):
        idx = i
        t = i / N
        nodes_x[idx] = 2.0 * t
        nodes_y[idx] = 0.0
        elems[idx, 0] = idx
        elems[idx, 1] = (idx + 1) % N_tot
        is_dirichlet[idx] = False

    # Right side: Dirichlet u = cos(pi y)
    for i in range(N):
        idx = N + i
        t = i / N
        nodes_x[idx] = 2.0
        nodes_y[idx] = t
        elems[idx, 0] = idx
        elems[idx, 1] = (idx + 1) % N_tot
        is_dirichlet[idx] = True

    # Top side: Neumann q = du/dn = 0
    for i in range(N):
        idx = 2 * N + i
        t = i / N
        nodes_x[idx] = 2.0 * (1.0 - t)
        nodes_y[idx] = 1.0
        elems[idx, 0] = idx
        elems[idx, 1] = (idx + 1) % N_tot
        is_dirichlet[idx] = False

    # Left side: Dirichlet u = 0
    for i in range(N):
        idx = 3 * N + i
        t = i / N
        nodes_x[idx] = 0.0
        nodes_y[idx] = 1.0 - t
        elems[idx, 0] = idx
        elems[idx, 1] = (idx + 1) % N_tot
        is_dirichlet[idx] = True

    # Rectangle corners: interior angle = pi/2 => c = 1/4
    ccoeff[0] = 0.25
    ccoeff[N] = 0.25
    ccoeff[2 * N] = 0.25
    ccoeff[3 * N] = 0.25

    for e in range(N_tot):
        j1 = elems[e, 0]
        j2 = elems[e, 1]
        dx = nodes_x[j2] - nodes_x[j1]
        dy = nodes_y[j2] - nodes_y[j1]
        L = np.sqrt(dx * dx + dy * dy)
        lengths[e] = L

        # For the CCW boundary, the outward normal is (dy/L, -dx/L)
        nx[e] = dy / L
        ny[e] = -dx / L

    # Boundary conditions
    u_known = np.zeros(N_tot)
    q_known = np.zeros(N_tot)

    for i in range(N_tot):
        x = nodes_x[i]
        y = nodes_y[i]
        if is_dirichlet[i]:
            u_known[i] = exact_u(x, y)
        else:
            # Homogeneous Neumann on top and bottom
            q_known[i] = 0.0

    pts, wts = get_gauss_points()

    # RHS assembly
    rhs = build_rhs(u_known, q_known, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts, is_dirichlet, ccoeff)
    t_setup = time.time() - t_start

    # Matrix-free GMRES solve
    t_solve_start = time.time()

    def matvec(v):
        return bem_matvec(v, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts, is_dirichlet, ccoeff)

    LO = LinearOperator((N_tot, N_tot), matvec=matvec)
    counter = IterationCounter()

    # Same solver family as the reference implementation
    sol, info = gmres(
        LO,
        rhs,
        callback=counter,
        rtol=1e-12,
        maxiter=8 * N_tot,
        callback_type='legacy'
    )

    t_solve = time.time() - t_solve_start

    if info != 0:
        print(f"Warning: GMRES returned info = {info}")

    # Recover all boundary values
    u_all = np.zeros(N_tot)
    q_all = np.zeros(N_tot)
    for i in range(N_tot):
        if is_dirichlet[i]:
            u_all[i] = u_known[i]
            q_all[i] = sol[i]
        else:
            q_all[i] = q_known[i]
            u_all[i] = sol[i]

    # Interior error evaluation
    # t_eval_start = time.time()
    # ngrid = 60
    # gxv = np.linspace(0.0, 2.0, ngrid)
    # gyv = np.linspace(0.0, 1.0, ngrid)
    # gx, gy = np.meshgrid(gxv, gyv)
    # gx = gx.flatten()
    # gy = gy.flatten()

    # mask = (gx > 0.0) & (gx < 2.0) & (gy > 0.0) & (gy < 1.0)
    # ix_pts = gx[mask]
    # iy_pts = gy[mask]

    t_eval_start = time.time()
    ngrid_x = 80; ngrid_y = 40
    _gx = np.linspace(0.01, 1.99, ngrid_x)
    _gy = np.linspace(0.01, 0.99, ngrid_y)
    gx, gy = np.meshgrid(_gx, _gy)
    gx = gx.flatten(); gy = gy.flatten()
    ix_pts = gx; iy_pts = gy

    u_interior = eval_interior(ix_pts, iy_pts, u_all, q_all, nodes_x, nodes_y, elems, nx, ny, lengths, pts, wts)
    u_exact_int = np.array([exact_u(px, py) for px, py in zip(ix_pts, iy_pts)])
    rel_l2 = np.linalg.norm(u_interior - u_exact_int) / np.linalg.norm(u_exact_int)
    t_eval = time.time() - t_eval_start

    return {
        "N": N,
        "DOFs": N_tot,
        "Iters": counter.niter,
        "error": rel_l2,
        "t_setup": t_setup,
        "t_solve": t_solve,
        "t_eval": t_eval,
        "t_total": t_setup + t_solve + t_eval
    }


if __name__ == "__main__":
    # Warm-up compilation
    _ = solve_bem(10)

    N_values = [160, 320, 640]
    results = []

    print(f"{'N':>5} | {'DOFs':>6} | {'Asm(s)':>8} | {'Solve(s)':>8} | {'Eval(s)':>8} | {'Total(s)':>8} | {'RelL2':>12}")
    print("-" * 92)

    for n in N_values:
        res = solve_bem(n)
        results.append(res)

    for res in results:
        print(
            f"{res['N']:5d} | {res['DOFs']:6d} | {res['t_setup']:8.3f} | {res['t_solve']:8.3f} | "
            f"{res['t_eval']:8.3f} | {res['t_total']:8.3f} | {res['error']:12.6e}"
        )

    dofs = np.array([r["DOFs"] for r in results], dtype=np.float64)
    errs = np.array([r["error"] for r in results], dtype=np.float64)

    coeff = np.polyfit(np.log(dofs), np.log(errs), 1)
    slope = -coeff[0]

    print("-" * 92)
    print(f"Observed Convergence Slope (error vs DOFs): {slope:.2f}")