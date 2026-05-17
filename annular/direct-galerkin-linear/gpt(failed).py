# Single Jupyter cell: Improved Galerkin BEM (Option B -> corrected normals + refined near-handling)
# Requirements: numpy, scipy, numba
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit, prange
import time
from math import pi

# ---------------------------
# Utilities
# ---------------------------
def leggauss_01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    x01 = 0.5*(x + 1.0)
    w01 = 0.5*w
    return x01, w01

def u_exact_xy(xy):
    x = xy[...,0]; y = xy[...,1]
    r = np.sqrt(x*x + y*y)
    theta = np.arctan2(y, x)
    return (r**3 + r**-3) * np.cos(3*theta)

# ---------------------------
# Geometry: build boundaries (radial normals enforced)
# ---------------------------
def build_circle_boundary(radius, N, orientation='CCW'):
    thetas = np.linspace(0, 2*pi, N, endpoint=False)
    if orientation == 'CW':
        thetas = thetas[::-1]
    nodes = np.column_stack([radius * np.cos(thetas), radius * np.sin(thetas)])
    elems = np.column_stack([np.arange(N), (np.arange(N)+1)%N]).astype(np.int32)
    vecs = nodes[elems[:,1]] - nodes[elems[:,0]]
    lengths = np.linalg.norm(vecs, axis=1)
    mid = 0.5*(nodes[elems[:,0]] + nodes[elems[:,1]])
    # radial normals (exact): outward = mid / |mid|, inward = -mid/|mid|
    normals = mid.copy()
    norms = np.sqrt(normals[:,0]**2 + normals[:,1]**2)
    normals[:,0] /= norms; normals[:,1] /= norms
    return nodes, elems, normals, lengths, mid

# ---------------------------
# Numba-accelerated local integrator (Duffy for self, subdivide for near)
# ---------------------------
@njit(fastmath=True)
def compute_local_SD(Ax,Ay, Bx,By, Cx,Cy, Dx,Dy, nx_src, ny_src, len_test, len_src,
                     rq, wq, nq, rq01, wq01, n_duff, n_sub, near_tol):
    S_el = np.zeros((2,2))
    D_el = np.zeros((2,2))
    # detect identical element (endpoints match in either orientation)
    same = False
    if (abs(Ax-Cx)<1e-12 and abs(Ay-Cy)<1e-12 and abs(Bx-Dx)<1e-12 and abs(By-Dy)<1e-12) or \
       (abs(Ax-Dx)<1e-12 and abs(Ay-Dy)<1e-12 and abs(Bx-Cx)<1e-12 and abs(By-Cy)<1e-12):
        same = True
    # midpoint distance
    mx_test_x = 0.5*(Ax + Bx); mx_test_y = 0.5*(Ay + By)
    mx_src_x  = 0.5*(Cx + Dx); mx_src_y  = 0.5*(Cy + Dy)
    dxm = mx_test_x - mx_src_x; dym = mx_test_y - mx_src_y
    dist = (dxm*dxm + dym*dym)**0.5

    # SELF-ELEMENT: use Duffy transform (u in [0,1], v in [0,1], s = u*v)
    if same:
        for iu in range(n_duff):
            u = rq01[iu]; wu = wq01[iu]
            for iv in range(n_duff):
                v = rq01[iv]; wv = wq01[iv]
                r = u
                s = u * v
                phi0 = 1.0 - r; phi1 = r
                psi0 = 1.0 - s; psi1 = s
                Xx = Ax*phi0 + Bx*phi1
                Xy = Ay*phi0 + By*phi1
                Yx = Cx*psi0 + Dx*psi1
                Yy = Cy*psi0 + Dy*psi1
                dx = Xx - Yx; dy = Xy - Yy
                r2 = dx*dx + dy*dy
                if r2 < 1e-30:
                    r2 = 1e-30
                r_ = r2**0.5
                G = -1.0/(2.0*pi) * np.log(r_)
                dot = dx*nx_src + dy*ny_src
                dG = -1.0/(2.0*pi) * dot / r2
                J = len_test * len_src * u
                w = wu * wv * J
                S_el[0,0] += (phi0 * psi0) * G * w
                S_el[0,1] += (phi0 * psi1) * G * w
                S_el[1,0] += (phi1 * psi0) * G * w
                S_el[1,1] += (phi1 * psi1) * G * w
                D_el[0,0] += (phi0 * psi0) * dG * w
                D_el[0,1] += (phi0 * psi1) * dG * w
                D_el[1,0] += (phi1 * psi0) * dG * w
                D_el[1,1] += (phi1 * psi1) * dG * w
        return S_el, D_el

    # NEAR interaction: subdivide panels
    if dist < near_tol * max(len_test, len_src):
        for it in range(n_sub):
            rt0 = it / n_sub; rt1 = (it+1) / n_sub
            for js in range(n_sub):
                rs0 = js / n_sub; rs1 = (js+1) / n_sub
                for i in range(nq):
                    rref = rq[i]; wr = wq[i]
                    r = rt0 + (rt1 - rt0) * rref
                    jr = (rt1 - rt0)
                    phi0 = 1.0 - r; phi1 = r
                    Xx = Ax*phi0 + Bx*phi1
                    Xy = Ay*phi0 + By*phi1
                    for j in range(nq):
                        sref = rq[j]; ws = wq[j]
                        s = rs0 + (rs1 - rs0) * sref
                        ks = (rs1 - rs0)
                        psi0 = 1.0 - s; psi1 = s
                        Yx = Cx*psi0 + Dx*psi1
                        Yy = Cy*psi0 + Dy*psi1
                        dx = Xx - Yx; dy = Xy - Yy
                        r2 = dx*dx + dy*dy
                        if r2 < 1e-30:
                            r2 = 1e-30
                        r_ = r2**0.5
                        G = -1.0/(2.0*pi) * np.log(r_)
                        dot = dx*nx_src + dy*ny_src
                        dG = -1.0/(2.0*pi) * dot / r2
                        J = len_test * len_src * jr * ks
                        w = wr * ws * J
                        S_el[0,0] += (phi0 * psi0) * G * w
                        S_el[0,1] += (phi0 * psi1) * G * w
                        S_el[1,0] += (phi1 * psi0) * G * w
                        S_el[1,1] += (phi1 * psi1) * G * w
                        D_el[0,0] += (phi0 * psi0) * dG * w
                        D_el[0,1] += (phi0 * psi1) * dG * w
                        D_el[1,0] += (phi1 * psi0) * dG * w
                        D_el[1,1] += (phi1 * psi1) * dG * w
        return S_el, D_el

    # FAR interaction: regular tensor-product quadrature
    for i in range(nq):
        r = rq[i]; wr = wq[i]
        phi0 = 1.0 - r; phi1 = r
        Xx = Ax*phi0 + Bx*phi1
        Xy = Ay*phi0 + By*phi1
        for j in range(nq):
            s = rq[j]; ws = wq[j]
            psi0 = 1.0 - s; psi1 = s
            Yx = Cx*psi0 + Dx*psi1
            Yy = Cy*psi0 + Dy*psi1
            dx = Xx - Yx; dy = Xy - Yy
            r2 = dx*dx + dy*dy
            if r2 < 1e-30:
                r2 = 1e-30
            r_ = r2**0.5
            G = -1.0/(2.0*pi) * np.log(r_)
            dot = dx*nx_src + dy*ny_src
            dG = -1.0/(2.0*pi) * dot / r2
            J = len_test * len_src
            w = wr * ws * J
            S_el[0,0] += (phi0 * psi0) * G * w
            S_el[0,1] += (phi0 * psi1) * G * w
            S_el[1,0] += (phi1 * psi0) * G * w
            S_el[1,1] += (phi1 * psi1) * G * w
            D_el[0,0] += (phi0 * psi0) * dG * w
            D_el[0,1] += (phi0 * psi1) * dG * w
            D_el[1,0] += (phi1 * psi0) * dG * w
            D_el[1,1] += (phi1 * psi1) * dG * w
    return S_el, D_el

# ---------------------------
# Element-level assembly functions (numba)
# ---------------------------
@njit(parallel=True, fastmath=True)
def matvec_elements(n_elems, elems, nodes, normals_elem, lengths,
                    eqn_map, col_u_map, col_q_map,
                    unk_u_vec, unk_q_vec,
                    rq, wq, nq, rq01, wq01, n_duff, n_sub, near_tol):
    neq = eqn_map.shape[0]
    y = np.zeros(neq)
    for ie in prange(n_elems):
        n1 = elems[ie,0]; n2 = elems[ie,1]
        Ax = nodes[n1,0]; Ay = nodes[n1,1]
        Bx = nodes[n2,0]; By = nodes[n2,1]
        len_test = lengths[ie]
        for je in range(n_elems):
            m1 = elems[je,0]; m2 = elems[je,1]
            Cx = nodes[m1,0]; Cy = nodes[m1,1]
            Dx = nodes[m2,0]; Dy = nodes[m2,1]
            len_src = lengths[je]
            nx_src = normals_elem[je,0]; ny_src = normals_elem[je,1]
            S_el, D_el = compute_local_SD(Ax,Ay,Bx,By, Cx,Cy,Dx,Dy, nx_src,ny_src, len_test,len_src,
                                          rq, wq, nq, rq01, wq01, n_duff, n_sub, near_tol)
            for a in range(2):
                g_test = elems[ie,a]
                row = eqn_map[g_test]
                if row < 0:
                    continue
                # D * u_unknown
                for b in range(2):
                    g_src = elems[je,b]
                    cu = col_u_map[g_src]
                    if cu >= 0:
                        y[row] += D_el[a,b] * unk_u_vec[cu]
                    cq = col_q_map[g_src]
                    if cq >= 0:
                        y[row] -= S_el[a,b] * unk_q_vec[cq]
    return y

@njit(parallel=True, fastmath=True)
def compute_known_actions(n_elems, elems, nodes, normals_elem, lengths,
                          eqn_map,
                          u_known, q_known,
                          rq, wq, nq, rq01, wq01, n_duff, n_sub, near_tol):
    neq = eqn_map.shape[0]
    yD = np.zeros(neq)
    yS = np.zeros(neq)
    for ie in prange(n_elems):
        n1 = elems[ie,0]; n2 = elems[ie,1]
        Ax = nodes[n1,0]; Ay = nodes[n1,1]
        Bx = nodes[n2,0]; By = nodes[n2,1]
        len_test = lengths[ie]
        for je in range(n_elems):
            m1 = elems[je,0]; m2 = elems[je,1]
            Cx = nodes[m1,0]; Cy = nodes[m1,1]
            Dx = nodes[m2,0]; Dy = nodes[m2,1]
            len_src = lengths[je]
            nx_src = normals_elem[je,0]; ny_src = normals_elem[je,1]
            S_el, D_el = compute_local_SD(Ax,Ay,Bx,By, Cx,Cy,Dx,Dy, nx_src,ny_src, len_test,len_src,
                                          rq, wq, nq, rq01, wq01, n_duff, n_sub, near_tol)
            for a in range(2):
                g_test = elems[ie,a]
                row = eqn_map[g_test]
                if row < 0:
                    continue
                for b in range(2):
                    g_src = elems[je,b]
                    yD[row] += D_el[a,b] * u_known[g_src]
                    yS[row] += S_el[a,b] * q_known[g_src]
    return yD, yS

@njit(parallel=True, fastmath=True)
def evaluate_interior(n_pts, pts, n_elems, elems, nodes, normals_elem, lengths,
                      u_num, q_num, rq, wq, nq, rq01, wq01, n_duff, n_sub, near_tol):
    out = np.zeros(n_pts)
    for p in prange(n_pts):
        x = pts[p,0]; y = pts[p,1]
        val = 0.0
        for je in range(n_elems):
            m1 = elems[je,0]; m2 = elems[je,1]
            Cx = nodes[m1,0]; Cy = nodes[m1,1]
            Dx = nodes[m2,0]; Dy = nodes[m2,1]
            len_src = lengths[je]
            nx_src = normals_elem[je,0]; ny_src = normals_elem[je,1]
            mx_src_x = 0.5*(Cx + Dx); mx_src_y = 0.5*(Cy + Dy)
            dxm = x - mx_src_x; dym = y - mx_src_y
            dist = (dxm*dxm + dym*dym)**0.5
            if dist < near_tol * len_src:
                for js in range(n_sub):
                    rs0 = js / n_sub; rs1 = (js+1) / n_sub
                    for j in range(nq):
                        sref = rq[j]; ws = wq[j]
                        s = rs0 + (rs1 - rs0) * sref
                        psi0 = 1.0 - s; psi1 = s
                        Yx = Cx*psi0 + Dx*psi1
                        Yy = Cy*psi0 + Dy*psi1
                        dx0 = x - Yx; dy0 = y - Yy
                        r2 = dx0*dx0 + dy0*dy0
                        if r2 < 1e-30:
                            r2 = 1e-30
                        r_ = r2**0.5
                        G = -1.0/(2.0*pi) * np.log(r_)
                        dot = dx0*nx_src + dy0*ny_src
                        dG = -1.0/(2.0*pi) * dot / r2
                        J = len_src * (rs1 - rs0)
                        w = ws * J
                        qy = (psi0 * q_num[m1] + psi1 * q_num[m2])
                        uy = (psi0 * u_num[m1] + psi1 * u_num[m2])
                        val += G * qy * w - dG * uy * w
            else:
                for j in range(nq):
                    s = rq[j]; ws = wq[j]
                    psi0 = 1.0 - s; psi1 = s
                    Yx = Cx*psi0 + Dx*psi1
                    Yy = Cy*psi0 + Dy*psi1
                    dx0 = x - Yx; dy0 = y - Yy
                    r2 = dx0*dx0 + dy0*dy0
                    if r2 < 1e-30:
                        r2 = 1e-30
                    r_ = r2**0.5
                    G = -1.0/(2.0*pi) * np.log(r_)
                    dot = dx0*nx_src + dy0*ny_src
                    dG = -1.0/(2.0*pi) * dot / r2
                    J = len_src
                    w = ws * J
                    qy = (psi0 * q_num[m1] + psi1 * q_num[m2])
                    uy = (psi0 * u_num[m1] + psi1 * u_num[m2])
                    val += G * qy * w - dG * uy * w
        out[p] = val
    return out

# ---------------------------
# Main matrix-free solver (with corrected radial normals and tuned quadrature/subdivision)
# ---------------------------
def run_bem_matrixfree_fix(N, nq=16, n_duff=20, n_sub=6, near_tol=2.5):
    t0 = time.perf_counter()
    # build boundaries with radial normals explicitly set
    nodes_in, elems_in, normals_in, lengths_in, mid_in = build_circle_boundary(1.0, N, orientation='CW')
    nodes_out, elems_out, normals_out, lengths_out, mid_out = build_circle_boundary(2.0, N, orientation='CCW')
    # enforce radial normals exactly: outer = +mid/|mid|, inner = -mid/|mid|
    norms_in = np.sqrt(mid_in[:,0]**2 + mid_in[:,1]**2)
    normals_in = - np.column_stack([mid_in[:,0]/norms_in, mid_in[:,1]/norms_in])
    norms_out = np.sqrt(mid_out[:,0]**2 + mid_out[:,1]**2)
    normals_out = np.column_stack([mid_out[:,0]/norms_out, mid_out[:,1]/norms_out])
    # combine
    offset = nodes_in.shape[0]
    nodes = np.vstack([nodes_in, nodes_out])
    elems_out_off = elems_out + offset
    elems = np.vstack([elems_in, elems_out_off]).astype(np.int32)
    normals_elem = np.vstack([normals_in, normals_out])
    lengths = np.concatenate([lengths_in, lengths_out])
    n_nodes = nodes.shape[0]; n_elems = elems.shape[0]
    # boundary masks
    is_inner = np.zeros(n_nodes, dtype=np.bool_)
    is_inner[:N] = True
    is_outer = ~is_inner
    # known BCs
    thetas = np.arctan2(nodes[:,1], nodes[:,0])
    u_known = np.zeros(n_nodes); q_known = np.zeros(n_nodes)
    u_known[is_inner] = 2.0 * np.cos(3*thetas[is_inner])
    q_known[is_outer] = (189.0/16.0) * np.cos(3*thetas[is_outer])
    # unknown indices
    unk_q_idx = np.where(is_inner)[0].astype(np.int32)
    unk_u_idx = np.where(is_outer)[0].astype(np.int32)
    nu = unk_u_idx.size; nq_u = unk_q_idx.size
    unknowns_total = nu + nq_u
    # equation nodes (unknown DOFs positions)
    eqn_nodes = np.concatenate((unk_u_idx, unk_q_idx)).astype(np.int32)
    neq = eqn_nodes.size
    eqn_map = -1 * np.ones(n_nodes, dtype=np.int32)
    for i in range(neq):
        eqn_map[eqn_nodes[i]] = i
    col_u_map = -1 * np.ones(n_nodes, dtype=np.int32)
    for i in range(nu):
        col_u_map[unk_u_idx[i]] = i
    col_q_map = -1 * np.ones(n_nodes, dtype=np.int32)
    for i in range(nq_u):
        col_q_map[unk_q_idx[i]] = i

    # quadrature on [0,1]
    rq, wq = leggauss_01(nq)
    rq01, wq01 = leggauss_01(n_duff)

    # compute known actions
    setup_t0 = time.perf_counter()
    Dknown_full, Sknown_full = compute_known_actions(n_elems, elems, nodes, normals_elem, lengths,
                                                    eqn_map, u_known, q_known,
                                                    rq, wq, nq, rq01, wq01, n_duff, n_sub, near_tol)
    C_known = np.zeros(neq)
    for r in range(neq):
        g = eqn_nodes[r]
        C_known[r] = 0.5 * u_known[g]
    b = - (C_known + Dknown_full - Sknown_full)
    setup_t1 = time.perf_counter()

    # matvec wrapper
    @njit(fastmath=True)
    def matvec_njit(x):
        # split
        u_part = np.zeros(nu)
        q_part = np.zeros(nq_u)
        for i in range(nu):
            u_part[i] = x[i]
        for i in range(nq_u):
            q_part[i] = x[nu + i]
        y = matvec_elements(n_elems, elems, nodes, normals_elem, lengths,
                            eqn_map, col_u_map, col_q_map,
                            u_part, q_part,
                            rq, wq, nq, rq01, wq01, n_duff, n_sub, near_tol)
        # add Cuu contributions (0.5 u for eqn rows corresponding to u unknowns)
        for i in range(nu):
            g = unk_u_idx[i]
            row = eqn_map[g]
            if row >= 0:
                y[row] += 0.5 * u_part[i]
        return y

    def matvec_wrapper(x):
        return matvec_njit(x)

    Aop = LinearOperator((neq, neq), matvec=matvec_wrapper, dtype=float)

    # GMRES solve
    iter_count = {'n':0}
    def gmres_cb(resnorm):
        iter_count['n'] += 1

    solve_t0 = time.perf_counter()
    if unknowns_total == 0:
        x_unknown = np.zeros(0); solve_t1 = time.perf_counter(); gmres_iters = 0
    else:
        x0 = np.zeros(unknowns_total)
        res = gmres(Aop, b, x0=x0, callback=gmres_cb, callback_type='pr_norm', rtol=1e-6, atol=0)
        x_unknown = res[0]; solve_t1 = time.perf_counter(); gmres_iters = iter_count['n']

    # recover u and q
    u_num = u_known.copy(); q_num = q_known.copy()
    if nu>0:
        for i in range(nu):
            u_num[unk_u_idx[i]] = x_unknown[i]
    if nq_u>0:
        for i in range(nq_u):
            q_num[unk_q_idx[i]] = x_unknown[nu + i]

    setup_time = setup_t1 - setup_t0
    solve_time = solve_t1 - solve_t0

    # interior evaluation (fixed grid)
    eval_t0 = time.perf_counter()
    ngrid = 60
    xs = np.linspace(-1.9, 1.9, ngrid)
    ys = np.linspace(-1.9, 1.9, ngrid)
    Xg, Yg = np.meshgrid(xs, ys, indexing='xy')
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    rs = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
    mask = (rs>1.0) & (rs<2.0)
    pts_in = pts[mask]
    npts = pts_in.shape[0]
    u_eval = evaluate_interior(npts, pts_in, n_elems, elems, nodes, normals_elem, lengths,
                               u_num, q_num, rq, wq, nq, rq01, wq01, n_duff, n_sub, near_tol)
    eval_t1 = time.perf_counter()
    u_ex = u_exact_xy(pts_in)
    rel_l2 = np.linalg.norm(u_eval - u_ex) / np.linalg.norm(u_ex)

    total_time = time.perf_counter() - t0
    return {
        'N': N,
        'Unknowns': unknowns_total,
        'GMRES_iters': gmres_iters,
        'RelL2': rel_l2,
        'Setup': setup_time,
        'Solve': solve_time,
        'Eval': eval_t1 - eval_t0,
        'Total': total_time
    }

# ---------------------------
# Refinement study and output (exact format)
# ---------------------------
N_values = [160, 320, 640, 1280, 2560]
results = []
for N in N_values:
    # tuned quadrature/subdivision parameters (higher accuracy)
    nq = 16
    n_duff = 20
    n_sub = 6
    near_tol = 2.5
    res = run_bem_matrixfree_fix(N, nq=nq, n_duff=n_duff, n_sub=n_sub, near_tol=near_tol)
    results.append(res)

hs = np.array([1.0/N for N in N_values])
errs = np.array([r['RelL2'] for r in results])
conv_rates = [None]
for i in range(1, len(errs)):
    rate = np.log(errs[i-1]/errs[i]) / np.log(hs[i-1]/hs[i])
    conv_rates.append(rate)
mask = errs>0
if mask.sum() >= 2:
    p = np.polyfit(np.log(hs[mask]), np.log(errs[mask]), 1)
    order = -p[0]
else:
    order = np.nan

print("N   Unknowns    GMRES    Rel L2 Error  Conv Rate     Setup     Solve      Eval     Total")
for i, r in enumerate(results):
    conv = f"{conv_rates[i]:.3f}" if conv_rates[i] is not None else "   -  "
    print(f"{r['N']:<4}{r['Unknowns']:<12}{r['GMRES_iters']:<9}{r['RelL2']:<13.6e}{conv:<13}{r['Setup']:<9.4f}{r['Solve']:<10.4f}{r['Eval']:<9.4f}{r['Total']:<.4f}")

print(f"\nFinal observed convergence order (least squares fit): {order:.4f}")