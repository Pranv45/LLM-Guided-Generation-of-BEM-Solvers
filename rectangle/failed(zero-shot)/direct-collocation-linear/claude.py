import numpy as np
from scipy.sparse.linalg import gmres
import time

# ── Gauss quadrature on [-1,1] ──────────────────────────────────────────────
NG = 10
GQ, GW = np.polynomial.legendre.leggauss(NG)
PHI1 = (1.0 - GQ) / 2.0   # shape fn at node 0
PHI2 = (1.0 + GQ) / 2.0   # shape fn at node 1

# ── Build mesh (no duplicate corners) ───────────────────────────────────────
def build_mesh(M):
    """
    4 sides, M panels each → 4*M nodes (corners shared, no duplicates).
    Traversal: bottom(→), right(↑), top(←), left(↓)  — CCW gives inward normals;
    we flip to get outward normals in the kernel.
    """
    def linspace_open(p0, p1, M):
        t = np.arange(M) / M          # 0, 1/M, ..., (M-1)/M  — no endpoint
        return p0 + np.outer(t, p1 - p0)

    p = np.array
    # bottom y=0  x:0→2   Neumann  q=0
    bot = linspace_open(p([0.,0.]), p([2.,0.]), M)
    # right  x=2  y:0→1   Dirichlet u=cos(πy)
    rgt = linspace_open(p([2.,0.]), p([2.,1.]), M)
    # top    y=1  x:2→0   Neumann  q=0
    top = linspace_open(p([2.,1.]), p([0.,1.]), M)
    # left   x=0  y:1→0   Dirichlet u=0
    lft = linspace_open(p([0.,1.]), p([0.,0.]), M)

    nodes = np.vstack([bot, rgt, top, lft])   # (4M, 2) — no duplicates
    Nn = 4 * M

    # element e connects node e → node (e+1)%Nn
    elems = np.stack([np.arange(Nn), (np.arange(Nn) + 1) % Nn], axis=1)

    # outward normals: rotate tangent 90° clockwise (for CCW traversal → outward)
    d = nodes[elems[:,1]] - nodes[elems[:,0]]
    L = np.linalg.norm(d, axis=1)
    normals = np.stack([ d[:,1]/L, -d[:,0]/L], axis=1)

    # BC per node
    bc     = np.empty(Nn, dtype=object)
    bc_val = np.zeros(Nn)
    bc[:M]          = 'N';  bc_val[:M]          = 0.0                          # bottom
    bc[M:2*M]       = 'D';  bc_val[M:2*M]       = np.cos(np.pi*nodes[M:2*M,1]) # right
    bc[2*M:3*M]     = 'N';  bc_val[2*M:3*M]     = 0.0                          # top
    bc[3*M:4*M]     = 'D';  bc_val[3*M:4*M]     = 0.0                          # left

    return nodes, elems, normals, L, bc, bc_val

# ── Analytic singular G self-integrals ───────────────────────────────────────
# For element of half-length h = L/2, with a node at one end (s = ±1):
#
# Own-node (singular end):
#   ∫_{-1}^{1} -1/(2π) ln(h|s+1|) · (1-s)/2  ds   [node at s=-1, φ₁=(1-s)/2]
#   = 1/(2π) · h · (3/2 - ln h - ln 2 + ln 2)
#   Substituting u = s+1 ∈ [0,2]:
#   = -1/(4π) ∫_0^2 ln(h·u)(1-u/2+1/2... let's use the exact closed form:
#
# ∫_0^2 ln(h u) · (2-u)/2 · du/2  [φ₁ = (1-s)/2 = (2-u)/2, jac=1/2 absorbed in h]
# Full integral (hand-verified):
#   own   = h/(2π) · [3/2 - ln(2h)]
#   cross = h/(2π) · [1/2 - ln(2h) + 1]  =  h/(2π) · [3/2 - ln(2h) - ... ]
# Using substitution u=1+s (singular at u=0, node0=-1):
#   I_own   = ∫_0^2 -1/(2π) ln(h·u) · (2-u)/2 · du  = h/(2π)·(3/2 - ln(2h))
#   I_cross = ∫_0^2 -1/(2π) ln(h·u) · u/2      · du  = h/(2π)·(1/2 - ln(2h)) + h/(2π)
#           = h/(2π)·(3/2 - ln(2h) - 1 + 1) -- redo:
# ∫_0^2 -ln(hu)·u/2 du = -1/2·[u²/2·(ln(hu)-1/2)]_0^2
#                       = -1/2·[2(ln(2h)-1/2)] = -(ln(2h)-1/2) = 1/2 - ln(2h)
# So I_cross = 1/(2π)·h·[1/2 - ln(2h) + 1] ... NO. h comes from substitution:
# The physical integral has jac h, Gauss variable t, sub u = 1+t ∈ [0,2]:
#   G_own   = h · 1/(2π) · (3/2 - ln(2h))  [multiply by h for jac]  — WRONG:
# The kernel already has the jac in the quadrature weight. Let's be explicit:
# ∫_{elem} G(xi,y) φ_s(y) ds  where ds = h dt (t∈[-1,1])
#   = ∫_{-1}^{1} (-1/(2π)) ln(h|t - t_node|) · φ_s(t) · h dt
# node0 at t=-1: sub u=t+1
#   own   = -h/(2π) ∫_0^2 ln(hu) (1-u/2)  ... φ₁=(1-t)/2=(1-(u-1))/2=(2-u)/2
#         = -h/(2π) [I1 - I2/2]
#   I1 = ∫_0^2 ln(hu) du = [u ln(hu)-u]_0^2 = 2ln(2h)-2
#   I2 = ∫_0^2 u ln(hu) du = [u²/2 ln(hu) - u²/4]_0^2 = 2ln(2h)-1
#   own = -h/(2π) [(2ln(2h)-2) - (2ln(2h)-1)/2]
#       = -h/(2π) [2ln(2h)-2 - ln(2h)+1/2]
#       = -h/(2π) [ln(2h) - 3/2]
#       =  h/(2π) [3/2 - ln(2h)]
#
#   cross (φ₂=(1+t)/2=u/2):
#   = -h/(2π) ∫_0^2 ln(hu)·u/2 du = -h/(4π)·[u²/2 ln(hu)-u²/4]_0^2
#   = -h/(4π)·[2ln(2h)-1]
#   =  h/(4π)·[1 - 2ln(2h)]

def singular_G(h):
    lh = np.log(2.0 * h)
    own   = h / (2.0*np.pi) * (1.5 - lh)
    cross = h / (4.0*np.pi) * (1.0 - 2.0*lh)
    return own, cross

# ── Assemble H and G ─────────────────────────────────────────────────────────
def assemble(nodes, elems, normals, lengths):
    Nn = len(nodes)
    Ne = len(elems)

    x0 = nodes[elems[:,0]]                                        # (Ne,2)
    x1 = nodes[elems[:,1]]                                        # (Ne,2)
    # Physical quad points: (Ne,NG,2)
    Xq  = x0[:,None,:] + PHI2[None,:,None] * (x1-x0)[:,None,:]
    jac = lengths / 2.0                                            # (Ne,)

    H = np.zeros((Nn, Nn))
    G = np.zeros((Nn, Nn))

    for i in range(Nn):
        xi   = nodes[i]                                            # (2,)
        diff = xi[None,None,:] - Xq                               # (Ne,NG,2)
        r2   = diff[:,:,0]**2 + diff[:,:,1]**2                    # (Ne,NG)
        r    = np.sqrt(r2)

        # Off-diagonal: suppress singular points (handled analytically)
        sing_mask = (r < 1e-14)

        with np.errstate(divide='ignore', invalid='ignore'):
            Gk = np.where(sing_mask, 0.0, -1.0/(2*np.pi)*np.log(r))
        rdotn = diff[:,:,0]*normals[:,None,0] + diff[:,:,1]*normals[:,None,1]
        with np.errstate(divide='ignore', invalid='ignore'):
            Hk = np.where(sing_mask, 0.0, -1.0/(2*np.pi)*rdotn/r2)

        wG = Gk * GW[None,:] * jac[:,None]
        wH = Hk * GW[None,:] * jac[:,None]

        g1 = np.sum(wG * PHI1[None,:], axis=1)   # (Ne,)
        g2 = np.sum(wG * PHI2[None,:], axis=1)
        h1 = np.sum(wH * PHI1[None,:], axis=1)
        h2 = np.sum(wH * PHI2[None,:], axis=1)

        np.add.at(G[i], elems[:,0], g1)
        np.add.at(G[i], elems[:,1], g2)
        np.add.at(H[i], elems[:,0], h1)
        np.add.at(H[i], elems[:,1], h2)

    # ── Replace singular G contributions with analytic values ────────────────
    # Node i belongs to two elements: the one ending at i and the one starting at i.
    # For element e with nodes [n0, n1]:
    #   if i == n0: singular at left end (t=-1) → own→G[i,n0], cross→G[i,n1]
    #   if i == n1: singular at right end (t=+1) → own→G[i,n1], cross→G[i,n0]
    for e in range(Ne):
        n0, n1 = elems[e]
        h = lengths[e] / 2.0
        own, cross = singular_G(h)
        # collocation at n0: singular at left end of element e
        G[n0, n0] += own
        G[n0, n1] += cross
        # collocation at n1: singular at right end of element e
        G[n1, n1] += own
        G[n1, n0] += cross

    # ── Jump term c_i = 1/2 (smooth boundary) ────────────────────────────────
    H += 0.5 * np.eye(Nn)

    return H, G

# ── Block solve ───────────────────────────────────────────────────────────────
def block_solve(H, G, bc, bc_val):
    Nn    = len(bc)
    D_idx = np.where(bc == 'D')[0]
    N_idx = np.where(bc == 'N')[0]
    nD, nN = len(D_idx), len(N_idx)

    A = np.empty((Nn, Nn))
    A[:, :nN]  =  H[:, N_idx]
    A[:, nN:]  = -G[:, D_idx]

    rhs = G[:, N_idx] @ bc_val[N_idx] - H[:, D_idx] @ bc_val[D_idx]

    x, info = gmres(A, rhs, atol=1e-13, rtol=1e-11, maxiter=5000)
    if info != 0:
        print(f"  [GMRES did not converge, info={info}]")

    u_full = np.zeros(Nn)
    q_full = np.zeros(Nn)
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
    Ne  = len(elems)

    x0  = nodes[elems[:,0]]
    x1  = nodes[elems[:,1]]
    Xq  = x0[:,None,:] + PHI2[None,:,None]*(x1-x0)[:,None,:]
    jac = lengths / 2.0

    u_int = np.zeros(Np)
    BATCH = 50
    n0a, n1a = elems[:,0], elems[:,1]

    for b in range(0, Np, BATCH):
        pb   = pts[b:b+BATCH]
        B    = len(pb)
        diff = pb[:,None,None,:] - Xq[None,:,:,:]       # (B,Ne,NG,2)
        r2   = diff[:,:,:,0]**2 + diff[:,:,:,1]**2
        r    = np.sqrt(r2)

        Gk    = -1.0/(2*np.pi)*np.log(r)
        rdotn = (diff[:,:,:,0]*normals[None,:,None,0]
               + diff[:,:,:,1]*normals[None,:,None,1])
        Hk    = -1.0/(2*np.pi)*rdotn/r2

        wG = Gk * GW[None,None,:] * jac[None,:,None]
        wH = Hk * GW[None,None,:] * jac[None,:,None]

        g1 = np.sum(wG*PHI1[None,None,:], axis=2)   # (B,Ne)
        g2 = np.sum(wG*PHI2[None,None,:], axis=2)
        h1 = np.sum(wH*PHI1[None,None,:], axis=2)
        h2 = np.sum(wH*PHI2[None,None,:], axis=2)

        u_int[b:b+B] = (
            np.sum(h1*u_full[None,n0a] + h2*u_full[None,n1a], axis=1)
          - np.sum(g1*q_full[None,n0a] + g2*q_full[None,n1a], axis=1)
        )

    u_num   = u_int.reshape(nyg, nxg)
    u_exact = (np.sinh(np.pi*XX)/np.sinh(2*np.pi))*np.cos(np.pi*YY)
    l2_err  = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)
    return u_num, l2_err

# ── Refinement study ──────────────────────────────────────────────────────────
print(f"{'M/side':>7} {'DOFs':>6} {'Assemble(s)':>12} {'Solve(s)':>10} "
      f"{'Eval(s)':>8} {'Total(s)':>9} {'L2 err':>12} {'Rate':>6}")
print("─"*80)

prev_err = None
for M in [10, 20, 40, 80, 160]:
    t0 = time.time()

    nodes, elems, normals, lengths, bc, bc_val = build_mesh(M)
    Nn = len(nodes)

    ta = time.time()
    H, G = assemble(nodes, elems, normals, lengths)
    t_a  = time.time() - ta

    ts = time.time()
    u_full, q_full = block_solve(H, G, bc, bc_val)
    t_s  = time.time() - ts

    te = time.time()
    _, l2_err = eval_interior(nodes, elems, normals, lengths, u_full, q_full)
    t_e  = time.time() - te

    t_tot = time.time() - t0
    rate  = f"{np.log2(prev_err/l2_err):.2f}" if prev_err is not None else "   —"
    print(f"{M:>7} {Nn:>6} {t_a:>12.4f} {t_s:>10.4f} {t_e:>8.4f} "
          f"{t_tot:>9.4f} {l2_err:>12.2e} {rate:>6}")
    prev_err = l2_err