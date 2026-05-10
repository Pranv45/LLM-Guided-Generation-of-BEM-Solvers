import numpy as np
from scipy.sparse.linalg import gmres
import time

# ── Gauss quadrature ─────────────────────────────────────────────────────────
NG = 10
GQ, GW = np.polynomial.legendre.leggauss(NG)
PHI1 = (1.0 - GQ) / 2.0
PHI2 = (1.0 + GQ) / 2.0

# Double quadrature grid (NG² points) via meshgrid
GQ1, GQ2 = np.meshgrid(GQ, GQ, indexing='ij')
GW1, GW2 = np.meshgrid(GW, GW, indexing='ij')
GQ1f = GQ1.ravel(); GQ2f = GQ2.ravel()
GW1f = GW1.ravel(); GW2f = GW2.ravel()
WW   = (GW1f * GW2f)          # (NG²,) double weights

P1_s = (1.0 - GQ1f) / 2.0    # test  φ₁ at outer quad pts
P2_s = (1.0 + GQ1f) / 2.0    # test  φ₂
P1_t = (1.0 - GQ2f) / 2.0    # trial φ₁ at inner quad pts
P2_t = (1.0 + GQ2f) / 2.0    # trial φ₂

# ── Mesh ─────────────────────────────────────────────────────────────────────
def build_mesh(M):
    def side_nodes(p0, p1):
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        t  = np.arange(M) / M
        return p0 + np.outer(t, p1 - p0)

    bot = side_nodes([0., 0.], [2., 0.])
    rgt = side_nodes([2., 0.], [2., 1.])
    top = side_nodes([2., 1.], [0., 1.])
    lft = side_nodes([0., 1.], [0., 0.])
    nodes = np.vstack([bot, rgt, top, lft])   # (4M, 2) no duplicate corners
    Nn    = 4 * M

    elems = np.stack([np.arange(Nn), (np.arange(Nn) + 1) % Nn], axis=1)

    d  = nodes[elems[:, 1]] - nodes[elems[:, 0]]
    L  = np.linalg.norm(d, axis=1)
    normals = np.stack([d[:, 1] / L, -d[:, 0] / L], axis=1)  # outward

    bc     = np.empty(Nn, dtype=object)
    bc_val = np.zeros(Nn)
    bc[:M]      = 'N'                                                   # bottom
    bc[M:2*M]   = 'D'; bc_val[M:2*M]   = np.cos(np.pi * nodes[M:2*M, 1])  # right
    bc[2*M:3*M] = 'N'                                                   # top
    bc[3*M:4*M] = 'D'                                                   # left u=0

    return nodes, elems, normals, L, bc, bc_val

# ── Singular self-element G integral via Duffy transformation ────────────────
def singular_G_element(h):
    """
    Compute 2×2 matrix  I[a,b] = ∫_{-1}^{1}∫_{-1}^{1}
        -1/(2π) ln(h|s-t|) φ_a(s) φ_b(t) h² ds dt
    for a straight element of half-length h using the Duffy trick.
    Split [−1,1]² into lower (s>t) and upper (s<t) triangles;
    by symmetry integrate lower and double.
    Duffy map on lower triangle: s = -1+η, t = -1+η·ξ, η∈[0,2], ξ∈[0,1]
    Jacobian = η,  |s-t| = η(1-ξ)  (never zero for ξ<1).
    """
    NG2 = 20
    pu, wu = np.polynomial.legendre.leggauss(NG2)
    pv, wv = np.polynomial.legendre.leggauss(NG2)

    # map pu → η ∈ [0,2],  pv → ξ ∈ [0,1]
    eta = 1.0 + pu          # ∈ [0,2],  weight factor ×1  (half-interval /2 × full 2)
    xi  = (1.0 + pv) / 2.0 # ∈ [0,1],  weight factor ×1/2
    weta = wu               # already scaled for [−1,1]→[0,2] (factor 1 since range=2)
    wxi  = wv / 2.0

    ETA, XI  = np.meshgrid(eta, xi,  indexing='ij')
    WE,  WX  = np.meshgrid(weta, wxi, indexing='ij')
    ETA = ETA.ravel(); XI = XI.ravel()
    Wduff = (WE * WX).ravel() * ETA   # include Jacobian η

    s_d = -1.0 + ETA
    t_d = -1.0 + ETA * XI
    dist = ETA * (1.0 - XI)           # = |s - t|, always > 0

    log_val = np.log(h * dist)        # ln(h·|s-t|)
    Gval    = -1.0 / (2.0 * np.pi) * log_val   # kernel value

    phi = np.array([(1.0 - s_d) / 2.0, (1.0 + s_d) / 2.0])  # (2, Nq)
    psi = np.array([(1.0 - t_d) / 2.0, (1.0 + t_d) / 2.0])  # (2, Nq)

    Iab = np.zeros((2, 2))
    for a in range(2):
        for b in range(2):
            # lower triangle × 2 (symmetry with upper)
            Iab[a, b] = 2.0 * np.sum(Gval * phi[a] * psi[b] * Wduff) * h**2

    return Iab

# ── Galerkin assembly ─────────────────────────────────────────────────────────
def assemble_galerkin(nodes, elems, normals, lengths):
    Nn = len(nodes)
    Ne = len(elems)

    x0 = nodes[elems[:, 0]]   # (Ne, 2)
    x1 = nodes[elems[:, 1]]   # (Ne, 2)
    dx = x1 - x0              # (Ne, 2)

    # Physical quad pts for every element: outer (s) and inner (t) variables
    # Shape (Ne, NG², 2)
    Xqs = x0[:, None, :] + P2_s[None, :, None] * dx[:, None, :]
    Xqt = x0[:, None, :] + P2_t[None, :, None] * dx[:, None, :]
    jacs = lengths / 2.0      # (Ne,) Jacobian of reference map

    H_mat = np.zeros((Nn, Nn))
    G_mat = np.zeros((Nn, Nn))

    # Precompute singular G blocks for each element
    sing_G = [singular_G_element(lengths[e] / 2.0) for e in range(Ne)]

    for ei in range(Ne):
        ni0, ni1 = elems[ei]
        js   = jacs[ei]
        xs_e = Xqs[ei]          # (NG², 2) — test (x) quad pts for element ei

        for ej in range(Ne):
            nj0, nj1 = elems[ej]
            jt   = jacs[ej]
            ny_e = normals[ej]  # (2,) outward normal of trial element

            if ei == ej:
                # G: analytic Duffy result; H self-term = 0 (straight element)
                Iab = sing_G[ej]
                G_mat[ni0, nj0] += Iab[0, 0]
                G_mat[ni0, nj1] += Iab[0, 1]
                G_mat[ni1, nj0] += Iab[1, 0]
                G_mat[ni1, nj1] += Iab[1, 1]
                continue

            # Off-diagonal double Gauss quadrature
            # diff[k] = x(GQ1[k]) - y(GQ2[k]) for the k-th tensor pair
            diff = xs_e - Xqt[ej]                    # (NG², 2)
            r2   = diff[:, 0]**2 + diff[:, 1]**2     # (NG²,)
            r    = np.sqrt(r2)

            with np.errstate(divide='ignore', invalid='ignore'):
                Gk = np.where(r > 0, -1.0 / (2.0 * np.pi) * np.log(r), 0.0)
                rdotn = diff[:, 0] * ny_e[0] + diff[:, 1] * ny_e[1]
                Hk = np.where(r2 > 0, -1.0 / (2.0 * np.pi) * rdotn / r2, 0.0)

            w  = WW * js * jt          # (NG²,) combined weight + Jacobians
            wG = Gk * w
            wH = Hk * w

            G_mat[ni0, nj0] += np.dot(wG, P1_s * P1_t)
            G_mat[ni0, nj1] += np.dot(wG, P1_s * P2_t)
            G_mat[ni1, nj0] += np.dot(wG, P2_s * P1_t)
            G_mat[ni1, nj1] += np.dot(wG, P2_s * P2_t)

            H_mat[ni0, nj0] += np.dot(wH, P1_s * P1_t)
            H_mat[ni0, nj1] += np.dot(wH, P1_s * P2_t)
            H_mat[ni1, nj0] += np.dot(wH, P2_s * P1_t)
            H_mat[ni1, nj1] += np.dot(wH, P2_s * P2_t)

    # ── Galerkin mass matrix for ½⟨φᵢ, u⟩ jump term ─────────────────────────
    # ∫ φ_a(x) φ_b(x) ds per element:  h/3 (a=b), h/6 (a≠b)
    M_jmp = np.zeros((Nn, Nn))
    for e in range(Ne):
        n0, n1 = elems[e]
        h = lengths[e]
        M_jmp[n0, n0] += h / 3.0
        M_jmp[n0, n1] += h / 6.0
        M_jmp[n1, n0] += h / 6.0
        M_jmp[n1, n1] += h / 3.0

    H_mat += 0.5 * M_jmp

    return H_mat, G_mat

# ── Block solve ───────────────────────────────────────────────────────────────
def block_solve(H_mat, G_mat, bc, bc_val):
    Nn    = len(bc)
    D_idx = np.where(bc == 'D')[0]
    N_idx = np.where(bc == 'N')[0]
    nN    = len(N_idx)

    A = np.empty((Nn, Nn))
    A[:, :nN] =  H_mat[:, N_idx]
    A[:, nN:] = -G_mat[:, D_idx]

    rhs = G_mat[:, N_idx] @ bc_val[N_idx] - H_mat[:, D_idx] @ bc_val[D_idx]

    x, info = gmres(A, rhs, atol=1e-13, rtol=1e-11, maxiter=5000)
    if info != 0:
        print(f"  [GMRES info={info}]")

    u_full = np.zeros(Nn); q_full = np.zeros(Nn)
    u_full[D_idx] = bc_val[D_idx]
    u_full[N_idx] = x[:nN]
    q_full[N_idx] = bc_val[N_idx]
    q_full[D_idx] = x[nN:]
    return u_full, q_full

# ── Interior evaluation ───────────────────────────────────────────────────────
def eval_interior(nodes, elems, normals, lengths, u_full, q_full):
    nxg, nyg = 40, 20
    xs = np.linspace(0.05, 1.95, nxg)
    ys = np.linspace(0.05, 0.95, nyg)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    Np  = len(pts)

    x0  = nodes[elems[:, 0]]
    x1  = nodes[elems[:, 1]]
    Xq  = x0[:, None, :] + PHI2[None, :, None] * (x1 - x0)[:, None, :]
    jac = lengths / 2.0
    n0a, n1a = elems[:, 0], elems[:, 1]

    u_int = np.zeros(Np)
    BATCH = 40

    for b in range(0, Np, BATCH):
        pb   = pts[b:b+BATCH]; B = len(pb)
        diff = pb[:, None, None, :] - Xq[None, :, :, :]   # (B,Ne,NG,2)
        r2   = diff[..., 0]**2 + diff[..., 1]**2
        r    = np.sqrt(r2)
        with np.errstate(divide='ignore', invalid='ignore'):
            Gk    = np.where(r > 0, -1.0/(2*np.pi)*np.log(r), 0.0)
            rdotn = (diff[..., 0]*normals[None, :, None, 0]
                   + diff[..., 1]*normals[None, :, None, 1])
            Hk    = np.where(r2 > 0, -1.0/(2*np.pi)*rdotn/r2, 0.0)

        wG = Gk  * GW[None, None, :] * jac[None, :, None]
        wH = Hk  * GW[None, None, :] * jac[None, :, None]

        g1 = np.sum(wG * PHI1[None, None, :], axis=2)
        g2 = np.sum(wG * PHI2[None, None, :], axis=2)
        h1 = np.sum(wH * PHI1[None, None, :], axis=2)
        h2 = np.sum(wH * PHI2[None, None, :], axis=2)

        u_int[b:b+B] = (
            np.sum(h1*u_full[None, n0a] + h2*u_full[None, n1a], axis=1)
          - np.sum(g1*q_full[None, n0a] + g2*q_full[None, n1a], axis=1)
        )

    u_num   = u_int.reshape(nyg, nxg)
    u_exact = (np.sinh(np.pi*XX) / np.sinh(2*np.pi)) * np.cos(np.pi*YY)
    l2_err  = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)
    return u_num, l2_err

# ── Refinement study ──────────────────────────────────────────────────────────
print(f"{'M/side':>7} {'DOFs':>6} {'Assemble(s)':>12} {'Solve(s)':>10} "
      f"{'Eval(s)':>8} {'Total(s)':>9} {'L2 err':>12} {'Rate':>6}")
print("─" * 80)

prev_err = None
for M in [10, 20, 40, 80, 160]:
    t0 = time.time()

    nodes, elems, normals, lengths, bc, bc_val = build_mesh(M)
    Nn = len(nodes)

    ta = time.time()
    H_mat, G_mat = assemble_galerkin(nodes, elems, normals, lengths)
    t_a = time.time() - ta

    ts = time.time()
    u_full, q_full = block_solve(H_mat, G_mat, bc, bc_val)
    t_s = time.time() - ts

    te = time.time()
    _, l2_err = eval_interior(nodes, elems, normals, lengths, u_full, q_full)
    t_e = time.time() - te

    t_tot = time.time() - t0
    rate  = f"{np.log2(prev_err/l2_err):.2f}" if prev_err is not None else "   —"
    print(f"{M:>7} {Nn:>6} {t_a:>12.3f} {t_s:>10.4f} {t_e:>8.4f} "
          f"{t_tot:>9.3f} {l2_err:>12.2e} {rate:>6}")
    prev_err = l2_err