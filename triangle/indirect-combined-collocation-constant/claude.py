import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit
import time

# ── Parameters ────────────────────────────────────────────────────────────────
ALPHA = 0.1
NG    = 10

# ── Gauss quadrature on [-1,1] ────────────────────────────────────────────────
_GQ, _GW = np.polynomial.legendre.leggauss(NG)
GQ = _GQ.copy(); GW = _GW.copy()

# ── Exact solution ────────────────────────────────────────────────────────────
def u_exact(x, y):
    return np.sin(np.pi * x) * np.cosh(np.pi * y)

def dudn_exact(x, y, nx, ny):
    dudx =  np.pi * np.cos(np.pi * x) * np.cosh(np.pi * y)
    dudy =  np.pi * np.sin(np.pi * x) * np.sinh(np.pi * y)
    return dudx * nx + dudy * ny

# ── Mesh ──────────────────────────────────────────────────────────────────────
def build_mesh(N):
    """Cosine-spaced panels on each edge of right triangle (0,0)-(1,0)-(0,1)."""
    def cosine_pts(p0, p1, n):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        t = 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / n))
        return p0 + np.outer(t, p1 - p0)          # (n+1, 2)

    edges = [
        ([0., 0.], [1., 0.]),   # bottom  n=(0,-1)
        ([1., 0.], [0., 1.]),   # hyp
        ([0., 1.], [0., 0.]),   # left    n=(-1,0)
    ]

    A_list, B_list = [], []
    for p0, p1 in edges:
        pts = cosine_pts(p0, p1, N)
        A_list.append(pts[:-1])
        B_list.append(pts[1:])

    Apts = np.vstack(A_list)   # (3N, 2)
    Bpts = np.vstack(B_list)

    centers = 0.5 * (Apts + Bpts)
    d       = Bpts - Apts
    lengths = np.linalg.norm(d, axis=1)
    # outward normals: rotate tangent 90° CW (CCW boundary → outward = CW rotation)
    normals = np.stack([d[:, 1] / lengths, -d[:, 0] / lengths], axis=1)

    return centers, normals, lengths, Apts, Bpts

# ── Numba matvec ──────────────────────────────────────────────────────────────
@njit(cache=True)
def _matvec_numba(mu, cx, cy, nx, ny, lengths,
                  ax, ay, bx, by,
                  gq, gw, ng, alpha):
    """
    Compute y = (0.5 I + K + alpha*V) mu.
    All arrays are 1-D / 2-D NumPy arrays passed in.
    """
    M  = len(mu)
    y  = np.zeros(M)

    for i in range(M):
        xi = cx[i]; yi_c = cy[i]
        nxi = nx[i]; nyi = ny[i]
        val = 0.5 * mu[i]   # jump term

        for j in range(M):
            Lj  = lengths[j]
            hj  = Lj / 2.0
            # map reference interval to physical segment j
            axj = ax[j]; ayj = ay[j]
            bxj = bx[j]; byj = by[j]
            nxj = nx[j]; nyj = ny[j]
            muj = mu[j]

            for k in range(ng):
                s  = gq[k]; w = gw[k]
                # physical quad pt on panel j
                t  = (1.0 + s) / 2.0
                qx = axj + t * (bxj - axj)
                qy = ayj + t * (byj - ayj)

                rx = xi - qx
                ry = yi_c - qy
                r2 = rx*rx + ry*ry
                if r2 < 1e-28:
                    continue
                r  = r2 ** 0.5

                # DLP kernel: -(1/2pi)(r.n_y)/r^2
                rdotny = rx*nxj + ry*nyj
                Hk = -1.0/(2.0*3.141592653589793) * rdotny / r2

                # SLP kernel: -(1/2pi) log(r)
                Gk = -1.0/(2.0*3.141592653589793) * 0.5*np.log(r2)  # log(r)=0.5*log(r^2)

                contrib = (Hk + alpha * Gk) * muj * w * hj
                val    += contrib

        y[i] = val
    return y

# ── RHS ───────────────────────────────────────────────────────────────────────
def build_rhs(centers, normals):
    """
    RHS = boundary data from exact solution.
    For an indirect formulation on a Dirichlet problem the RHS is
    f_i = u_exact(x_i)  (Dirichlet condition, enforced at collocation pts).
    """
    return u_exact(centers[:, 0], centers[:, 1])

# ── Interior evaluation ───────────────────────────────────────────────────────
@njit(cache=True)
def _eval_interior_numba(px, py, mu,
                          cx, cy, nx, ny, lengths,
                          ax, ay, bx, by,
                          gq, gw, ng, alpha):
    Np = len(px)
    M  = len(mu)
    u  = np.zeros(Np)
    PI2 = 2.0 * 3.141592653589793

    for p in range(Np):
        xi = px[p]; yi_c = py[p]
        val = 0.0
        for j in range(M):
            Lj  = lengths[j]; hj = Lj / 2.0
            axj = ax[j]; ayj = ay[j]
            bxj = bx[j]; byj = by[j]
            nxj = nx[j]; nyj = ny[j]
            muj = mu[j]
            for k in range(ng):
                s  = gq[k]; w = gw[k]
                t  = (1.0 + s) / 2.0
                qx = axj + t*(bxj - axj)
                qy = ayj + t*(byj - ayj)
                rx = xi - qx; ry = yi_c - qy
                r2 = rx*rx + ry*ry
                if r2 < 1e-28:
                    continue
                rdotny = rx*nxj + ry*nyj
                Hk = -rdotny / (PI2 * r2)
                Gk = -0.5 * np.log(r2) / PI2
                val += (Hk + alpha*Gk) * muj * w * hj
        u[p] = val
    return u

# ── Warm-up Numba ─────────────────────────────────────────────────────────────
def warmup_numba(centers, normals, lengths, Apts, Bpts):
    mu0 = np.ones(len(centers))
    _matvec_numba(mu0,
                  centers[:,0].copy(), centers[:,1].copy(),
                  normals[:,0].copy(), normals[:,1].copy(),
                  lengths.copy(),
                  Apts[:,0].copy(), Apts[:,1].copy(),
                  Bpts[:,0].copy(), Bpts[:,1].copy(),
                  GQ, GW, NG, ALPHA)

# ── Solve ─────────────────────────────────────────────────────────────────────
def solve_bem(centers, normals, lengths, Apts, Bpts, rhs):
    M   = len(centers)
    cx  = centers[:,0].copy(); cy  = centers[:,1].copy()
    nxv = normals[:,0].copy(); nyv = normals[:,1].copy()
    ax  = Apts[:,0].copy();    ay  = Apts[:,1].copy()
    bx  = Bpts[:,0].copy();    by  = Bpts[:,1].copy()

    iters = [0]
    def callback(xk):
        iters[0] += 1

    def matvec(mu):
        return _matvec_numba(mu.astype(np.float64),
                             cx, cy, nxv, nyv, lengths,
                             ax, ay, bx, by,
                             GQ, GW, NG, ALPHA)

    A = LinearOperator((M, M), matvec=matvec, dtype=np.float64)
    mu, info = gmres(A, rhs, atol=1e-10, rtol=1e-10,
                     maxiter=2000, callback=callback, callback_type='legacy')
    return mu, iters[0], info

# ── Refinement study ──────────────────────────────────────────────────────────
N_values = [25, 50, 100, 200, 400]

# Interior grid (fixed)
ngrid = 60
xg = np.linspace(0.02, 0.98, ngrid)
yg = np.linspace(0.02, 0.98, ngrid)
XXg, YYg = np.meshgrid(xg, yg)
mask = (XXg + YYg) < 1.0
px   = XXg[mask].copy()
py   = YYg[mask].copy()
u_ex = u_exact(px, py)

print(f"{'N':>5} {'DOFs':>6} {'Iters':>6} {'L2 err':>12} "
      f"{'Setup(s)':>10} {'Solve(s)':>10} {'Eval(s)':>9} {'Total(s)':>10}")
print("─" * 80)

# Warm up Numba with smallest N to avoid timing JIT compilation
_c, _n, _l, _a, _b = build_mesh(N_values[0])
warmup_numba(_c, _n, _l, _a, _b)

prev_err = None
for N in N_values:
    t0 = time.perf_counter()

    # Setup
    ts = time.perf_counter()
    centers, normals, lengths, Apts, Bpts = build_mesh(N)
    rhs = build_rhs(centers, normals)
    t_setup = time.perf_counter() - ts

    # Solve
    ts = time.perf_counter()
    mu, niters, info = solve_bem(centers, normals, lengths, Apts, Bpts, rhs)
    t_solve = time.perf_counter() - ts

    # Evaluate interior
    ts = time.perf_counter()
    u_int = _eval_interior_numba(px, py, mu,
                                  centers[:,0].copy(), centers[:,1].copy(),
                                  normals[:,0].copy(), normals[:,1].copy(),
                                  lengths.copy(),
                                  Apts[:,0].copy(), Apts[:,1].copy(),
                                  Bpts[:,0].copy(), Bpts[:,1].copy(),
                                  GQ, GW, NG, ALPHA)
    t_eval = time.perf_counter() - ts

    t_tot  = time.perf_counter() - t0
    l2_err = np.linalg.norm(u_int - u_ex) / np.linalg.norm(u_ex)

    rate = ""
    if prev_err is not None:
        rate = f"  rate={np.log2(prev_err/l2_err):.2f}"
    print(f"{N:>5} {len(centers):>6} {niters:>6} {l2_err:>12.4e} "
          f"{t_setup:>10.4f} {t_solve:>10.4f} {t_eval:>9.4f} {t_tot:>10.4f}{rate}")
    prev_err = l2_err