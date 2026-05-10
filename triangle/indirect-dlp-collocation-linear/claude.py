import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit
import math, time

# ── Constants ─────────────────────────────────────────────────────────────────
ETA   = 1.0
_PI   = math.pi

# ── Gauss quadrature ──────────────────────────────────────────────────────────
GQ_PTS, GQ_WTS = np.polynomial.legendre.leggauss(12)
GQ_PTS = np.ascontiguousarray(GQ_PTS)
GQ_WTS = np.ascontiguousarray(GQ_WTS)

# ── Exact solution ────────────────────────────────────────────────────────────
def u_exact(x, y):
    return np.sin(_PI * x) * np.cosh(_PI * y)

# ── Interior grid ─────────────────────────────────────────────────────────────
ngrid = 60
_xg = np.linspace(0.02, 0.98, ngrid)
_yg = np.linspace(0.02, 0.98, ngrid)
_Xg, _Yg = np.meshgrid(_xg, _yg)
_Xf, _Yf = _Xg.ravel(), _Yg.ravel()
_msk  = _Xf + _Yf < 1.0
X_int = np.ascontiguousarray(np.stack([_Xf[_msk], _Yf[_msk]], axis=1))
u_ref = u_exact(_Xf[_msk], _Yf[_msk])

# ── Boundary discretization ───────────────────────────────────────────────────
def cosine_nodes(n, a, b):
    """n+1 cosine-spaced nodes from a to b (clustered at endpoints)."""
    k = np.arange(n + 1)
    t = 0.5*(1.0 - np.cos(_PI * k / n))
    return a + t*(b - a)

def make_boundary(N):
    """
    Right triangle: (0,0)→(1,0)→(0,1)→(0,0).
    Each edge gets N elements (N+1 nodes, shared corners merged).
    Cosine spacing on each edge.

    Returns:
      nodes    : (Nn, 2)
      conn     : (Ne, 2)  element node indices
      normals  : (Ne, 2)  outward unit normals
      lengths  : (Ne,)    element lengths
    """
    # Three edges
    edges = [
        (np.array([0.0, 0.0]), np.array([1.0, 0.0])),  # bottom
        (np.array([1.0, 0.0]), np.array([0.0, 1.0])),  # hypotenuse
        (np.array([0.0, 1.0]), np.array([0.0, 0.0])),  # left
    ]
    # Outward normals for each edge
    edge_normals = [
        np.array([ 0.0, -1.0]),   # bottom  → downward
        np.array([ 1.0,  1.0]) / math.sqrt(2.0),  # hypotenuse → upper-right
        np.array([-1.0,  0.0]),   # left    → leftward
    ]

    all_nodes  = []
    all_conn   = []
    all_normals= []
    all_lengths= []
    offset = 0

    for (A, B), en in zip(edges, edge_normals):
        t_vals = cosine_nodes(N, 0.0, 1.0)   # N+1 values in [0,1]
        pts = np.outer(1.0 - t_vals, A) + np.outer(t_vals, B)  # (N+1, 2)

        # Store nodes (skip last: shared with next edge)
        all_nodes.append(pts[:-1])

        for k in range(N):
            i1 = offset + k
            i2 = offset + (k + 1) % N   # wraps within this edge's nodes
            # Last element connects last local node back to next edge's first node
            # Handle with raw indices after all nodes merged
            p1 = pts[k]; p2 = pts[k+1]
            L  = np.linalg.norm(p2 - p1)
            all_conn.append([offset + k, -1])   # placeholder, fix below
            all_lengths.append(L)
            all_normals.append(en.copy())

        offset += N   # N nodes per edge (last node shared)

    # Build proper node array and connectivity
    nodes = np.vstack(all_nodes)   # (3N, 2)
    Nn = nodes.shape[0]            # = 3N
    Ne = 3 * N

    conn    = np.zeros((Ne, 2), dtype=np.int32)
    normals = np.array(all_normals)
    lengths = np.array(all_lengths)

    for e in range(Ne):
        conn[e, 0] = e % Nn
        conn[e, 1] = (e + 1) % Nn

    # Recompute lengths and normals from actual node positions
    for e in range(Ne):
        i1, i2 = conn[e]
        p1 = nodes[i1]; p2 = nodes[i2]
        d  = p2 - p1
        L  = np.linalg.norm(d)
        lengths[e] = L
        # Normal already set per edge; keep as is (correct outward direction)

    return (np.ascontiguousarray(nodes),
            np.ascontiguousarray(conn),
            np.ascontiguousarray(normals),
            np.ascontiguousarray(lengths))

# ── Numba matvec ──────────────────────────────────────────────────────────────
@njit(cache=True)
def _matvec(sigma, nodes, conn, normals, lengths, gq_pts, gq_wts, eta):
    """
    Computes: y = A σ  where
    A[i] = SLP[σ](x_i) + η * DLP[σ](x_i) - (η/2) σ(x_i)

    SLP kernel : G(x,y)      = -(1/2π) log|x-y|
    DLP kernel : ∂G/∂n_y    =  (1/2π) (x-y)·n_y / |x-y|²

    For coincident integrals (collocation node on source element):
      SLP: weakly singular → handled by log-weighted quadrature subtraction
      DLP: CPV → computed via regularisation (subtract and add back)

    For the SLP self-element contribution with constant approximation the
    integral has a known closed form; here we use the standard approach of
    computing all off-element integrals with Gauss and handling the singular
    element by the identity:  ∫_Γ ∂G/∂n_y ds_y = -1/2  (interior)
    → we compute DLP self via: -1/2 * phi_j(x_i) integrated ... see below.
    """
    inv2pi = 1.0 / (2.0 * math.pi)
    Nn = nodes.shape[0]
    Ne = conn.shape[0]
    Nq = gq_pts.shape[0]
    res = np.zeros(Nn)

    for ci in range(Nn):
        xc0 = nodes[ci, 0]; xc1 = nodes[ci, 1]

        slp_val = 0.0
        dlp_val = 0.0

        for e in range(Ne):
            i1 = conn[e, 0]; i2 = conn[e, 1]
            p10 = nodes[i1, 0]; p11 = nodes[i1, 1]
            p20 = nodes[i2, 0]; p21 = nodes[i2, 1]
            L   = lengths[e]
            en0 = normals[e, 0]; en1 = normals[e, 1]
            hL  = 0.5 * L

            # Check if collocation node is an endpoint of this element
            on_elem = (i1 == ci) or (i2 == ci)

            if not on_elem:
                # ── Regular Gauss quadrature ──────────────────────────────
                for q in range(Nq):
                    s    = gq_pts[q]; w = gq_wts[q]
                    ph1  = 0.5*(1.0 - s); ph2 = 0.5*(1.0 + s)
                    yy0  = p10*ph1 + p20*ph2
                    yy1  = p11*ph1 + p21*ph2
                    sig_q = sigma[i1]*ph1 + sigma[i2]*ph2
                    dx = xc0 - yy0; dy = xc1 - yy1
                    r2 = dx*dx + dy*dy
                    if r2 < 1e-28: continue
                    jac = w * hL
                    # SLP
                    slp_val += -inv2pi*0.5*math.log(r2) * sig_q * jac
                    # DLP
                    dlp_val +=  inv2pi*(dx*en0 + dy*en1)/r2 * sig_q * jac

            else:
                # ── Singular element: collocation node is i1 or i2 ───────
                # Map so that singular point is at s = -1 (local node 1).
                # If ci == i1: singular at s=-1, use substitution s = -1 + t²  (Duffy)
                # If ci == i2: singular at s=+1, flip element

                if ci == i1:
                    # singular at s = -1
                    # x(s) = p1*(1-s)/2 + p2*(1+s)/2, singularity at s=-1 (y=p1=x_c)
                    # Use Duffy: s = -1 + t, t ∈ [0,2], or split [−1,1] = [-1,0]+[0,1]
                    # and use s = -1 + t on each half with Gauss on t.
                    for half_idx in range(2):
                        for q in range(Nq):
                            # Map Gauss pt to t ∈ [0,1]
                            t_ref = 0.5*(1.0 + gq_pts[q])  # t in [0,1]
                            wt    = 0.5 * gq_wts[q]
                            if half_idx == 0:
                                # s = -1 + t, t ∈ [0,1] → s ∈ [-1,0]
                                s = -1.0 + t_ref
                                duffy_jac = 1.0   # ds/dt = 1, but log |s+1| = log t
                            else:
                                # s = t, t ∈ [0,1] → s ∈ [0,1]; no singularity
                                s = t_ref
                                duffy_jac = 1.0

                            ph1 = 0.5*(1.0 - s); ph2 = 0.5*(1.0 + s)
                            yy0 = p10*ph1 + p20*ph2
                            yy1 = p11*ph1 + p21*ph2
                            sig_q = sigma[i1]*ph1 + sigma[i2]*ph2
                            dx = xc0 - yy0; dy = xc1 - yy1
                            r2 = dx*dx + dy*dy
                            if r2 < 1e-28: continue
                            jac = wt * hL * duffy_jac
                            slp_val += -inv2pi*0.5*math.log(r2) * sig_q * jac
                            dlp_val +=  inv2pi*(dx*en0+dy*en1)/r2 * sig_q * jac
                else:
                    # ci == i2, singular at s = +1; flip: use s' = -s
                    for half_idx in range(2):
                        for q in range(Nq):
                            t_ref = 0.5*(1.0 + gq_pts[q])
                            wt    = 0.5 * gq_wts[q]
                            if half_idx == 0:
                                s = 1.0 - t_ref   # s ∈ [0,1]
                            else:
                                s = -(t_ref)       # s ∈ [-1,0]

                            ph1 = 0.5*(1.0 - s); ph2 = 0.5*(1.0 + s)
                            yy0 = p10*ph1 + p20*ph2
                            yy1 = p11*ph1 + p21*ph2
                            sig_q = sigma[i1]*ph1 + sigma[i2]*ph2
                            dx = xc0 - yy0; dy = xc1 - yy1
                            r2 = dx*dx + dy*dy
                            if r2 < 1e-28: continue
                            jac = wt * hL
                            slp_val += -inv2pi*0.5*math.log(r2) * sig_q * jac
                            dlp_val +=  inv2pi*(dx*en0+dy*en1)/r2 * sig_q * jac

        # Combined operator: SLP + η*DLP - (η/2)*σ(x)
        res[ci] = slp_val + eta*dlp_val - (eta*0.5)*sigma[ci]

    return res


@njit(cache=True)
def _eval_interior(x_pts, nodes, conn, normals, lengths, sigma,
                   gq_pts, gq_wts, eta):
    """u(x) = SLP[σ](x) + η * DLP[σ](x)  for x in interior."""
    inv2pi = 1.0 / (2.0 * math.pi)
    Np = x_pts.shape[0]; Ne = conn.shape[0]; Nq = gq_pts.shape[0]
    u  = np.zeros(Np)

    for p in range(Np):
        xc0 = x_pts[p, 0]; xc1 = x_pts[p, 1]
        val = 0.0
        for e in range(Ne):
            i1 = conn[e, 0]; i2 = conn[e, 1]
            p10 = nodes[i1,0]; p11 = nodes[i1,1]
            p20 = nodes[i2,0]; p21 = nodes[i2,1]
            L   = lengths[e]; hL = 0.5*L
            en0 = normals[e,0]; en1 = normals[e,1]
            for q in range(Nq):
                s    = gq_pts[q]; w = gq_wts[q]
                ph1  = 0.5*(1.0-s); ph2 = 0.5*(1.0+s)
                yy0  = p10*ph1 + p20*ph2
                yy1  = p11*ph1 + p21*ph2
                sig_q = sigma[i1]*ph1 + sigma[i2]*ph2
                dx = xc0-yy0; dy = xc1-yy1
                r2 = dx*dx+dy*dy
                if r2 < 1e-28: continue
                jac = w*hL
                slp_k = -inv2pi*0.5*math.log(r2)
                dlp_k =  inv2pi*(dx*en0+dy*en1)/r2
                val  += (slp_k + eta*dlp_k)*sig_q*jac
        u[p] = val
    return u

# ── Warmup ────────────────────────────────────────────────────────────────────
def _warmup():
    nd, co, nm, le = make_boundary(4)
    s0 = np.ones(nd.shape[0])
    _matvec(s0, nd, co, nm, le, GQ_PTS, GQ_WTS, ETA)
    _eval_interior(X_int[:2].copy(), nd, co, nm, le, s0, GQ_PTS, GQ_WTS, ETA)

_warmup()

# ── Refinement study ──────────────────────────────────────────────────────────
N_values = [100, 200, 400, 800]
errors = []; hs = []
prev_err = None; prev_h = None

print(f"{'N':<6}{'Unknowns':<12}{'GMRES':<9}{'Rel L2 Error':<16}"
      f"{'Conv Rate':<14}{'Setup':>8}{'Solve':>9}{'Eval':>9}{'Total':>9}")
print("-"*100)

for N in N_values:
    t0 = time.perf_counter()

    nodes, conn, normals, lengths = make_boundary(N)
    Nn = nodes.shape[0]   # = 3N

    rhs = u_exact(nodes[:, 0], nodes[:, 1])

    _nd = np.ascontiguousarray(nodes)
    _co = np.ascontiguousarray(conn)
    _nm = np.ascontiguousarray(normals)
    _le = np.ascontiguousarray(lengths)

    t_setup = time.perf_counter() - t0

    iters = [0]
    def _cb(rk): iters[0] += 1

    def mv(sigma):
        return _matvec(sigma, _nd, _co, _nm, _le, GQ_PTS, GQ_WTS, ETA)

    A = LinearOperator((Nn, Nn), matvec=mv, dtype=np.float64)

    t1 = time.perf_counter()
    sigma_sol, info = gmres(A, rhs, rtol=1e-10, atol=1e-12,
                             maxiter=500, restart=100,
                             callback=_cb, callback_type='pr_norm')
    t_solve = time.perf_counter() - t1

    t2 = time.perf_counter()
    u_num = _eval_interior(X_int, _nd, _co, _nm, _le,
                           np.ascontiguousarray(sigma_sol),
                           GQ_PTS, GQ_WTS, ETA)
    t_eval = time.perf_counter() - t2

    rel_err = np.linalg.norm(u_num - u_ref) / np.linalg.norm(u_ref)
    h = 1.0 / N
    errors.append(rel_err); hs.append(h)

    rate_str = "N/A"
    if prev_err is not None and rel_err > 0 and prev_err > 0:
        rate = math.log(prev_err/rel_err) / math.log(prev_h/h)
        rate_str = f"{rate:.2f}"

    t_tot = t_setup + t_solve + t_eval
    print(f"{N:<6}{Nn:<12}{iters[0]:<9}{rel_err:<16.4e}"
          f"{rate_str:<14}{t_setup:>7.2f}s{t_solve:>8.2f}s"
          f"{t_eval:>8.2f}s{t_tot:>8.2f}s")

    prev_err = rel_err; prev_h = h

_p = np.polyfit(np.log(np.array(hs)), np.log(np.array(errors)), 1)
print(f"\nObserved convergence order (least-squares fit): {_p[0]:.3f}")