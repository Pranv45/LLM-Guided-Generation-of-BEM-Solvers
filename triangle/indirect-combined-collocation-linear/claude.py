import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit
import time

# ── Gauss quadrature ──────────────────────────────────────────────────────────
NG = 10
_GQ, _GW = np.polynomial.legendre.leggauss(NG)
GQ = _GQ.copy()
GW = _GW.copy()
PHI1_Q = (1.0 - GQ) / 2.0   # shape fn φ1 at quad pts
PHI2_Q = (1.0 + GQ) / 2.0   # shape fn φ2 at quad pts

# ── Exact solution ─────────────────────────────────────────────────────────────
def u_exact_fn(x, y):
    return np.sin(np.pi * x) * np.cosh(np.pi * y)

# ── Mesh ───────────────────────────────────────────────────────────────────────
def build_mesh(N):
    """
    Cosine-spaced nodes on each edge.  No duplicate corner nodes.
    3N nodes total, 3N elements (closed polygon).
    Returns
    -------
    nodes   : (3N, 2)
    elems   : (3N, 2)  node indices per element
    normals : (3N, 2)  outward unit normals per element
    lengths : (3N,)    element arc lengths
    """
    def cosine_nodes(p0, p1, n):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        t = 0.5 * (1.0 - np.cos(np.pi * np.arange(n) / n))
        return p0 + np.outer(t, p1 - p0)          # (n, 2) — excludes endpoint

    edges = [
        ([0., 0.], [1., 0.]),   # bottom
        ([1., 0.], [0., 1.]),   # hypotenuse
        ([0., 1.], [0., 0.]),   # left
    ]
    parts = [cosine_nodes(p0, p1, N) for p0, p1 in edges]
    nodes = np.vstack(parts)     # (3N, 2)
    Nn    = 3 * N

    # element e connects node e → node (e+1)%Nn
    idx   = np.arange(Nn)
    elems = np.stack([idx, (idx + 1) % Nn], axis=1)   # (3N, 2)

    d       = nodes[elems[:, 1]] - nodes[elems[:, 0]]
    lengths = np.linalg.norm(d, axis=1)
    normals = np.stack([d[:, 1] / lengths, -d[:, 0] / lengths], axis=1)

    return nodes, elems, normals, lengths

# ── Numba matvec  y = (1/2 I + K) μ ──────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _matvec_nb(mu,
               node_x, node_y,
               e0, e1,
               nx, ny, lengths,
               gq, gw, phi1q, phi2q, ng):
    """
    Collocation at boundary nodes; linear elements; DLP kernel.
    For collocation node i on element e (as endpoint), the singular integral
    is treated via the free-term (1/2); the principal-value self-contribution
    of K on a straight element is zero.
    """
    Nn  = len(mu)
    Ne  = len(e0)
    PI2 = 6.283185307179586   # 2π
    out = np.empty(Nn)

    for i in range(Nn):
        xi = node_x[i]
        yi = node_y[i]
        acc = 0.5 * mu[i]       # jump term

        for e in range(Ne):
            n0 = e0[e]; n1 = e1[e]
            hj  = lengths[e] * 0.5      # Jacobian (half element length)
            nxe = nx[e]; nye = ny[e]
            ax  = node_x[n0]; ay = node_y[n0]
            bx  = node_x[n1]; by = node_y[n1]
            mu0 = mu[n0]; mu1 = mu[n1]

            # Skip if collocation node is either endpoint (singular; K PV = 0)
            if n0 == i or n1 == i:
                continue

            for k in range(ng):
                s  = gq[k]
                p1 = phi1q[k]; p2 = phi2q[k]
                # physical quad pt
                qx = ax * p1 + bx * p2 + 0.0   # = ax+(1+s)/2*(bx-ax)
                qy = ay * p1 + by * p2
                # Gaussian weight × Jacobian
                w  = gw[k] * hj

                rx  = xi - qx
                ry  = yi - qy
                r2  = rx*rx + ry*ry
                if r2 < 1.0e-28:
                    continue

                # DLP kernel: ∂G/∂n_y = -(r·n_y)/(2π r²)
                rdotny = rx*nxe + ry*nye
                Hk     = -rdotny / (PI2 * r2)

                mu_q = mu0*p1 + mu1*p2
                acc  += Hk * mu_q * w

        out[i] = acc
    return out

# ── Warmup ─────────────────────────────────────────────────────────────────────
def _warmup(nodes, elems, normals, lengths):
    mu0 = np.ones(len(nodes))
    _matvec_nb(mu0,
               nodes[:, 0].copy(), nodes[:, 1].copy(),
               elems[:, 0].copy(), elems[:, 1].copy(),
               normals[:, 0].copy(), normals[:, 1].copy(),
               lengths.copy(),
               GQ, GW, PHI1_Q, PHI2_Q, NG)

# ── Interior evaluation ────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _eval_int_nb(px, py, mu,
                 node_x, node_y,
                 e0, e1,
                 nx, ny, lengths,
                 gq, gw, phi1q, phi2q, ng):
    Np  = len(px)
    Ne  = len(e0)
    PI2 = 6.283185307179586
    u   = np.zeros(Np)

    for p in range(Np):
        xi = px[p]; yi = py[p]
        acc = 0.0
        for e in range(Ne):
            n0 = e0[e]; n1 = e1[e]
            hj  = lengths[e] * 0.5
            nxe = nx[e]; nye = ny[e]
            ax  = node_x[n0]; ay = node_y[n0]
            bx  = node_x[n1]; by = node_y[n1]
            mu0 = mu[n0]; mu1 = mu[n1]

            for k in range(ng):
                s  = gq[k]
                p1 = phi1q[k]; p2 = phi2q[k]
                qx = ax*p1 + bx*p2
                qy = ay*p1 + by*p2
                w  = gw[k] * hj

                rx  = xi - qx; ry = yi - qy
                r2  = rx*rx + ry*ry
                if r2 < 1.0e-28:
                    continue

                rdotny = rx*nxe + ry*nye
                Hk     = -rdotny / (PI2 * r2)

                acc += Hk * (mu0*p1 + mu1*p2) * w
        u[p] = acc
    return u

# ── Main refinement study ──────────────────────────────────────────────────────
N_values = [25, 50, 100, 200, 400]

# Fixed interior grid
ngrid = 60
xg = np.linspace(0.02, 0.98, ngrid)
yg = np.linspace(0.02, 0.98, ngrid)
XXg, YYg = np.meshgrid(xg, yg)
mask = (XXg + YYg) < 1.0
px   = XXg[mask].ravel().copy()
py   = YYg[mask].ravel().copy()
u_ex = u_exact_fn(px, py)

# Warm up Numba JIT with smallest problem
_nodes0, _elems0, _nrm0, _len0 = build_mesh(N_values[0])
_warmup(_nodes0, _elems0, _nrm0, _len0)

print(f"{'N':>5} {'DOFs':>6} {'Iters':>7} {'L2 err':>12} "
      f"{'Setup(s)':>10} {'Solve(s)':>10} {'Eval(s)':>9} {'Total(s)':>10}")
print("─" * 82)

prev_err = None
for N in N_values:
    t0 = time.perf_counter()

    # ── Setup ──────────────────────────────────────────────────────────────
    ts = time.perf_counter()
    nodes, elems, normals, lengths = build_mesh(N)
    Nn = len(nodes)

    # Flatten for Numba
    node_x = nodes[:, 0].copy(); node_y = nodes[:, 1].copy()
    e0     = elems[:, 0].copy(); e1     = elems[:, 1].copy()
    nxv    = normals[:, 0].copy(); nyv  = normals[:, 1].copy()
    lenv   = lengths.copy()

    # RHS: Dirichlet data at nodes
    rhs = u_exact_fn(node_x, node_y)
    t_setup = time.perf_counter() - ts

    # ── GMRES solve ─────────────────────────────────────────────────────────
    iters = [0]
    def callback(xk): iters[0] += 1

    def matvec(mu):
        return _matvec_nb(mu.astype(np.float64),
                          node_x, node_y, e0, e1,
                          nxv, nyv, lenv,
                          GQ, GW, PHI1_Q, PHI2_Q, NG)

    A  = LinearOperator((Nn, Nn), matvec=matvec, dtype=np.float64)
    ts = time.perf_counter()
    mu, info = gmres(A, rhs, atol=1e-10, rtol=1e-10,
                     maxiter=3000, callback=callback)
    t_solve = time.perf_counter() - ts

    # ── Interior evaluation ─────────────────────────────────────────────────
    ts = time.perf_counter()
    u_int = _eval_int_nb(px, py, mu,
                         node_x, node_y, e0, e1,
                         nxv, nyv, lenv,
                         GQ, GW, PHI1_Q, PHI2_Q, NG)
    t_eval = time.perf_counter() - ts

    t_tot  = time.perf_counter() - t0
    l2_err = np.linalg.norm(u_int - u_ex) / np.linalg.norm(u_ex)

    rate = ""
    if prev_err is not None:
        rate = f"  rate={np.log2(prev_err / l2_err):.2f}"
    print(f"{N:>5} {Nn:>6} {iters[0]:>7} {l2_err:>12.4e} "
          f"{t_setup:>10.4f} {t_solve:>10.4f} {t_eval:>9.4f} {t_tot:>10.4f}{rate}")
    prev_err = l2_err