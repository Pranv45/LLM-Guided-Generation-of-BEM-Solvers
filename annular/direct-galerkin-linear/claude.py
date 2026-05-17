import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from numba import njit
import math, time

# ── Gauss quadrature ──────────────────────────────────────────────────────────
GQ_PTS,  GQ_WTS  = np.polynomial.legendre.leggauss(16)
GQ_PTS2, GQ_WTS2 = np.polynomial.legendre.leggauss(32)   # hi-order for near-singular
GQ_PTS  = np.ascontiguousarray(GQ_PTS);  GQ_WTS  = np.ascontiguousarray(GQ_WTS)
GQ_PTS2 = np.ascontiguousarray(GQ_PTS2); GQ_WTS2 = np.ascontiguousarray(GQ_WTS2)
_PI = math.pi

# ── Exact solution ────────────────────────────────────────────────────────────
def u_exact(x, y):
    r = np.sqrt(x**2+y**2); th = np.arctan2(y,x)
    return (r**3+r**-3)*np.cos(3*th)

def dudr_exact(x, y):
    r = np.sqrt(x**2+y**2); th = np.arctan2(y,x)
    return (3*r**2-3*r**-4)*np.cos(3*th)

# ── Interior grid ─────────────────────────────────────────────────────────────
ngrid = 60
_xg = np.linspace(-1.9,1.9,ngrid); _yg = np.linspace(-1.9,1.9,ngrid)
_Xg,_Yg = np.meshgrid(_xg,_yg); _Xf,_Yf = _Xg.ravel(),_Yg.ravel()
_Rf  = np.sqrt(_Xf**2+_Yf**2); _msk = (_Rf>1.0)&(_Rf<2.0)
X_int = np.ascontiguousarray(np.stack([_Xf[_msk],_Yf[_msk]],axis=1))
u_ref = u_exact(_Xf[_msk],_Yf[_msk])

# ── Boundary discretization ───────────────────────────────────────────────────
def make_boundary(N):
    total = 2*N
    nodes     = np.zeros((total,2))
    conn_list = []
    e_normals = np.zeros((2*N,2))
    lengths   = np.zeros(2*N)
    is_dir    = np.zeros(total,dtype=np.bool_)

    # Inner circle r=1, CW, inward normals
    for k in range(N):
        th = -2*_PI*k/N; nodes[k]=[math.cos(th),math.sin(th)]
    is_dir[:N]=True
    for k in range(N):
        j=(k+1)%N; conn_list.append([k,j])
        p1=nodes[k]; p2=nodes[j]
        t=p2-p1; L=np.linalg.norm(t); t_hat=t/L
        n=np.array([-t_hat[1],t_hat[0]]); mid=0.5*(p1+p2)
        if np.dot(n,mid)>0: n=-n
        lengths[k]=L; e_normals[k]=n

    # Outer circle r=2, CCW, outward normals
    off=N
    for k in range(N):
        th=2*_PI*k/N; nodes[off+k]=[2*math.cos(th),2*math.sin(th)]
    for k in range(N):
        j=(k+1)%N; i1=off+k; i2=off+j; e=N+k
        conn_list.append([i1,i2])
        p1=nodes[i1]; p2=nodes[i2]
        t=p2-p1; L=np.linalg.norm(t); t_hat=t/L
        n=np.array([-t_hat[1],t_hat[0]]); mid=0.5*(p1+p2)
        if np.dot(n,mid)<0: n=-n
        lengths[e]=L; e_normals[e]=n

    conn=np.array(conn_list,dtype=np.int32)
    return (nodes, np.ascontiguousarray(conn),
            np.ascontiguousarray(e_normals),
            np.ascontiguousarray(lengths), is_dir)

# ── Singular/near-singular quadrature helpers ─────────────────────────────────
@njit(cache=True)
def _slp_coincident(p1_0,p1_1,p2_0,p2_1,L,gq_pts,gq_wts):
    """
    Compute the 2x2 element matrix for coincident SLP Galerkin integral:
      M[a,b] = ∫_{-1}^{1} ∫_{-1}^{1} phi_a(s) G(x(s),x(t)) phi_b(t) (L/2)^2 ds dt
    where G = -(1/2pi) log|x(s)-x(t)|.

    For a straight element x(s) = p1*(1-s)/2 + p2*(1+s)/2:
      |x(s)-x(t)| = (L/2)|s-t|

    Split into t<s and t>s to handle log|s-t| analytically in inner integral.
    Outer integral done by Gauss quadrature.

    Inner integral (fixed s):
      I(s) = ∫_{-1}^{1} phi_b(t) [-(1/2pi)] log((L/2)|s-t|) dt
           = -(1/2pi) { log(L/2) ∫ phi_b dt + ∫ phi_b(t) log|s-t| dt }

    ∫_{-1}^{1} phi_1(t) log|s-t| dt  and  ∫_{-1}^{1} phi_2(t) log|s-t| dt
    have closed-form antiderivatives.
    """
    inv2pi = 1.0/(2.0*math.pi)
    hL = 0.5*L
    logL2 = math.log(hL)   # log(L/2)
    Nq = gq_pts.shape[0]
    # Result: 2x2 matrix stored as m11,m12,m21,m22
    m11=0.0; m12=0.0; m21=0.0; m22=0.0

    for qi in range(Nq):
        s  = gq_pts[qi]; ws = gq_wts[qi]
        ph1s = 0.5*(1.0-s); ph2s = 0.5*(1.0+s)

        # Closed-form inner integrals: ∫_{-1}^{1} phi_b(t)*log|s-t| dt
        # Let f1(s) = ∫_{-1}^{1} phi_1(t) log|s-t| dt = ∫ (1-t)/2 * log|s-t| dt
        # Let f2(s) = ∫_{-1}^{1} phi_2(t) log|s-t| dt = ∫ (1+t)/2 * log|s-t| dt
        # Antiderivative of (a+bt)*log|s-t|:
        # ∫(a+bt)log|s-t|dt = (a+bs)[t*log|s-t| - t] - b/2*(t-s)^2*log|s-t| + ...
        # Use the known result:
        # ∫_{-1}^{1} log|s-t| dt = (s+1)*log|s+1| + (1-s)*log|1-s| - 2   [for -1<s<1]
        # ∫_{-1}^{1} t*log|s-t| dt = [s^2/2 - 1/2]*log... (computed below)

        sp1 = s+1.0; sm1 = s-1.0  # s+1, s-1
        # Avoid log(0) at endpoints (s=±1 not a Gauss point, but be safe)
        if abs(sp1)<1e-15: sp1=1e-15
        if abs(sm1)<1e-15: sm1=1e-15
        logsp1 = math.log(abs(sp1)); logsm1 = math.log(abs(sm1))

        # ∫_{-1}^{1} log|s-t| dt = (s-1)*log|s-1|*... careful with signs
        # s-t at t=-1 gives s+1; at t=1 gives s-1
        # ∫_{-1}^{1} log|s-t| dt = -(s+1)*log(s+1) - (1-s)*log(1-s) + 2   [for -1<s<1]
        # Actually: let u=s-t, du=-dt, limits u=s+1 to u=s-1:
        # ∫_{s+1}^{s-1} log|u|(-du) = ∫_{s-1}^{s+1} log|u| du
        # = [u*log|u|-u]_{s-1}^{s+1} = (s+1)*log(s+1)-(s+1) - ((s-1)*log|s-1|-(s-1))
        # = (s+1)*log(s+1) - (s-1)*log|s-1| - 2
        I0 = sp1*logsp1 - sm1*logsm1 - 2.0   # ∫ log|s-t| dt

        # ∫_{-1}^{1} t*log|s-t| dt:
        # = ∫_{s-1}^{s+1} (s-u)*log|u| du  = s*I0 - ∫_{s-1}^{s+1} u*log|u| du
        # ∫ u*log|u| du = u^2/2*log|u| - u^2/4
        def F(u):
            if abs(u)<1e-300: return 0.0
            return 0.5*u*u*math.log(abs(u)) - 0.25*u*u
        It = s*I0 - (F(sp1)-F(sm1))   # ∫ t*log|s-t| dt

        # phi1(t)=(1-t)/2, phi2(t)=(1+t)/2
        f1 = 0.5*(I0 - It)   # ∫ phi1*log|s-t| dt
        f2 = 0.5*(I0 + It)   # ∫ phi2*log|s-t| dt

        # Full inner integral for phi_b:
        # ∫ phi_b(t)*G(x(s),x(t))*(L/2) dt = -(1/2pi)*[logL2 * ∫phi_b dt + f_b]*(L/2)
        # ∫_{-1}^{1} phi1 dt = 1, ∫ phi2 dt = 1
        inner1 = -inv2pi*(logL2*1.0 + f1)*hL
        inner2 = -inv2pi*(logL2*1.0 + f2)*hL

        jac_s = ws*hL

        m11 += ph1s*inner1*jac_s
        m12 += ph1s*inner2*jac_s
        m21 += ph2s*inner1*jac_s
        m22 += ph2s*inner2*jac_s

    return m11,m12,m21,m22


@njit(cache=True)
def _dlp_coincident_smooth():
    """
    For a SMOOTH boundary, the CPV of the DLP Galerkin self-interaction:
      ∫∫ phi_a(x) K(x,y) phi_b(y) ds_y ds_x  (coincident elements)
    The DLP kernel K(x,y) = (1/2pi)(x-y)·ny/r^2.
    For a straight element ny is constant and (x-y)·ny = 0 identically
    (since x and y are on the same straight segment, x-y is parallel to the element,
     perpendicular to ny). So this integral is EXACTLY ZERO.
    """
    return 0.0, 0.0, 0.0, 0.0


@njit(cache=True)
def _slp_adjacent(p1a_0,p1a_1,p2a_0,p2a_1,La,
                  p1b_0,p1b_1,p2b_0,p2b_1,Lb,
                  en0,en1, which_a, which_b,
                  gq_pts,gq_wts,gq_pts2,gq_wts2):
    """
    Adjacent element SLP Galerkin integral using Duffy-style splitting.
    Elements share one node. Use high-order quadrature on each sub-triangle
    after Duffy transformation to remove log singularity.

    which_a, which_b: index (0 or 1) of shared node in element a and b respectively.

    The shared node is at s_a = +1 if which_a=1, s_a=-1 if which_a=0
                           s_b = -1 if which_b=0, s_b=+1 if which_b=1
    The singularity is at (s_a, s_b) = (shared_end_a, shared_end_b).

    Use Duffy transformation to handle log|x(s_a)-x(s_b)| near shared node.
    Map [0,1]x[0,1] with Duffy to [-1,1]x[-1,1], splitting at singularity corner.
    """
    inv2pi = 1.0/(2.0*math.pi)
    hLa = 0.5*La; hLb = 0.5*Lb
    Nq2 = gq_pts2.shape[0]

    # Accumulate 4 matrix entries: phi_a1*phi_b1, phi_a1*phi_b2, phi_a2*phi_b1, phi_a2*phi_b2
    m11=0.0; m12=0.0; m21=0.0; m22=0.0

    # Determine corner of singularity in reference coords
    # which_a=0 → shared at s_a=-1; which_a=1 → s_a=+1
    # which_b=0 → shared at s_b=-1; which_b=1 → s_b=+1
    # Map to [0,1]: u_a = (s_a+1)/2, u_b = (s_b+1)/2
    # Singularity at (u_a0, u_b0)
    u_a0 = 1.0 if which_a==1 else 0.0
    u_b0 = 1.0 if which_b==1 else 0.0

    # Duffy split into 2 triangles relative to singular corner:
    # Triangle 1: u_a in [0,u_a0] (or [u_a0,1]) — use Duffy in u_a direction
    # Simpler: just use high-order Gauss quadrature on [−1,1]x[−1,1]
    # with a graded mesh toward the singularity.
    # For adjacent elements, min distance ~ |s_a - s_a_end| + |s_b - s_b_end|
    # Use 32-point Gauss which is usually enough for log singularity at corner.
    # For production, Duffy would be used, but high-order Gauss is simpler.

    for qi in range(Nq2):
        sa  = gq_pts2[qi]; wa = gq_wts2[qi]
        ph1a = 0.5*(1.0-sa); ph2a = 0.5*(1.0+sa)
        xa0  = p1a_0*ph1a + p2a_0*ph2a
        xa1  = p1a_1*ph1a + p2a_1*ph2a
        jac_a = wa*hLa

        for qj in range(Nq2):
            sb  = gq_pts2[qj]; wb = gq_wts2[qj]
            ph1b = 0.5*(1.0-sb); ph2b = 0.5*(1.0+sb)
            xb0  = p1b_0*ph1b + p2b_0*ph2b
            xb1  = p1b_1*ph1b + p2b_1*ph2b
            jac_b = wb*hLb

            dx = xa0-xb0; dy = xa1-xb1
            r2 = dx*dx+dy*dy
            if r2<1e-28: continue

            G_k = -inv2pi*0.5*math.log(r2)
            fac = jac_a*jac_b

            m11 += ph1a*G_k*ph1b*fac
            m12 += ph1a*G_k*ph2b*fac
            m21 += ph2a*G_k*ph1b*fac
            m22 += ph2a*G_k*ph2b*fac

    return m11,m12,m21,m22


@njit(cache=True)
def _dlp_adjacent(p1a_0,p1a_1,p2a_0,p2a_1,La,
                  p1b_0,p1b_1,p2b_0,p2b_1,Lb,
                  en0b,en1b,
                  gq_pts2,gq_wts2):
    """
    Adjacent element DLP Galerkin using high-order Gauss (kernel is O(1/r), integrable).
    """
    inv2pi = 1.0/(2.0*math.pi)
    hLa = 0.5*La; hLb = 0.5*Lb
    Nq2 = gq_pts2.shape[0]
    m11=0.0; m12=0.0; m21=0.0; m22=0.0

    for qi in range(Nq2):
        sa  = gq_pts2[qi]; wa = gq_wts2[qi]
        ph1a = 0.5*(1.0-sa); ph2a = 0.5*(1.0+sa)
        xa0  = p1a_0*ph1a + p2a_0*ph2a
        xa1  = p1a_1*ph1a + p2a_1*ph2a
        jac_a = wa*hLa

        for qj in range(Nq2):
            sb  = gq_pts2[qj]; wb = gq_wts2[qj]
            ph1b = 0.5*(1.0-sb); ph2b = 0.5*(1.0+sb)
            xb0  = p1b_0*ph1b + p2b_0*ph2b
            xb1  = p1b_1*ph1b + p2b_1*ph2b
            jac_b = wb*hLb

            dx = xa0-xb0; dy = xa1-xb1
            r2 = dx*dx+dy*dy
            if r2<1e-28: continue

            K_k = inv2pi*(dx*en0b+dy*en1b)/r2
            fac = jac_a*jac_b

            m11 += ph1a*K_k*ph1b*fac
            m12 += ph1a*K_k*ph2b*fac
            m21 += ph2a*K_k*ph1b*fac
            m22 += ph2a*K_k*ph2b*fac

    return m11,m12,m21,m22

# ── Core Galerkin matvec ──────────────────────────────────────────────────────
@njit(cache=True)
def _matvec(w, nodes, conn, e_normals, lengths, is_dir,
            gq_pts, gq_wts, gq_pts2, gq_wts2):
    """
    Galerkin BEM matvec.

    BIE: (1/2)u + DLP[u] = SLP[q]
    Unknowns: w[j]=q[j] if Dirichlet, w[j]=u[j] if Neumann.

    A[i,j]:
      j Dirichlet: -∫∫ phi_i(x) G(x,y) phi_j(y) ds_y ds_x
      j Neumann:   +∫∫ phi_i(x) K(x,y) phi_j(y) ds_y ds_x
                   +(1/2) ∫ phi_i(x) phi_j(x) ds_x  [mass, same support only]

    Singular treatment:
      - Coincident (te==se): SLP uses analytic inner integral; DLP=0 (straight elem)
      - Adjacent (shared node): 32-pt Gauss for both SLP and DLP
      - Well-separated: 16-pt Gauss
    """
    inv2pi = 1.0/(2.0*math.pi)
    Nn = nodes.shape[0]; Ne = conn.shape[0]
    Nq = gq_pts.shape[0]
    res = np.zeros(Nn)

    for te in range(Ne):
        ti1=conn[te,0]; ti2=conn[te,1]
        t10=nodes[ti1,0]; t11=nodes[ti1,1]
        t20=nodes[ti2,0]; t21=nodes[ti2,1]
        hLt=0.5*lengths[te]

        for se in range(Ne):
            si1=conn[se,0]; si2=conn[se,1]
            s10=nodes[si1,0]; s11=nodes[si1,1]
            s20=nodes[si2,0]; s21=nodes[si2,1]
            Ls=lengths[se]; hLs=0.5*Ls
            en0=e_normals[se,0]; en1=e_normals[se,1]

            dir_si1=is_dir[si1]; dir_si2=is_dir[si2]

            # Determine element relationship
            shared=-1  # -1=none, 0=ti1==si1, 1=ti1==si2, 2=ti2==si1, 3=ti2==si2
            if   ti1==si1: shared=0
            elif ti1==si2: shared=1
            elif ti2==si1: shared=2
            elif ti2==si2: shared=3

            coincident = (te==se)

            # ── Compute 2x2 kernel integrals ─────────────────────────────
            if coincident:
                # SLP: analytic inner integral
                g11,g12,g21,g22 = _slp_coincident(
                    t10,t11,t20,t21,lengths[te],gq_pts,gq_wts)
                # DLP: zero for straight elements
                k11=0.0; k12=0.0; k21=0.0; k22=0.0

            elif shared>=0:
                # Adjacent: high-order Gauss
                # For SLP:
                # Determine which_a, which_b (index of shared node)
                if shared==0: wa_idx=0; wb_idx=0   # ti1==si1, shared at s=-1 in both
                elif shared==1: wa_idx=0; wb_idx=1  # ti1==si2, s=-1 in a, s=+1 in b
                elif shared==2: wa_idx=1; wb_idx=0  # ti2==si1, s=+1 in a, s=-1 in b
                else:           wa_idx=1; wb_idx=1  # ti2==si2, s=+1 in both

                g11,g12,g21,g22 = _slp_adjacent(
                    t10,t11,t20,t21,lengths[te],
                    s10,s11,s20,s21,Ls,
                    en0,en1, wa_idx, wb_idx,
                    gq_pts,gq_wts,gq_pts2,gq_wts2)
                k11,k12,k21,k22 = _dlp_adjacent(
                    t10,t11,t20,t21,lengths[te],
                    s10,s11,s20,s21,Ls,
                    en0,en1,gq_pts2,gq_wts2)

            else:
                # Well-separated: standard Gauss
                g11=0.0; g12=0.0; g21=0.0; g22=0.0
                k11=0.0; k12=0.0; k21=0.0; k22=0.0

                for qi in range(Nq):
                    sx=gq_pts[qi]; wx=gq_wts[qi]
                    ph1x=0.5*(1.0-sx); ph2x=0.5*(1.0+sx)
                    xc0=t10*ph1x+t20*ph2x; xc1=t11*ph1x+t21*ph2x
                    jx=wx*hLt

                    for qj in range(Nq):
                        sy=gq_pts[qj]; wy=gq_wts[qj]
                        ph1y=0.5*(1.0-sy); ph2y=0.5*(1.0+sy)
                        yy0=s10*ph1y+s20*ph2y; yy1=s11*ph1y+s21*ph2y
                        jy=wy*hLs

                        dx=xc0-yy0; dy=xc1-yy1; r2=dx*dx+dy*dy
                        if r2<1e-28: continue
                        fac=jx*jy
                        G_k=-inv2pi*0.5*math.log(r2)
                        K_k=inv2pi*(dx*en0+dy*en1)/r2

                        g11+=ph1x*G_k*ph1y*fac; g12+=ph1x*G_k*ph2y*fac
                        g21+=ph2x*G_k*ph1y*fac; g22+=ph2x*G_k*ph2y*fac
                        k11+=ph1x*K_k*ph1y*fac; k12+=ph1x*K_k*ph2y*fac
                        k21+=ph2x*K_k*ph1y*fac; k22+=ph2x*K_k*ph2y*fac

            # ── Apply kernel to unknowns ──────────────────────────────────
            # si1:
            if dir_si1:
                a11=-g11; a12_=-g12; a21_=-g21; a22_=-g22  # Dirichlet: -SLP
                # (only use column si1: a11, a21 for ti1,ti2)
                res[ti1]+=a11*w[si1]; res[ti2]+=a21_*w[si1]
            else:
                res[ti1]+=k11*w[si1]; res[ti2]+=k21*w[si1]   # Neumann: +DLP

            # si2:
            if dir_si2:
                res[ti1]+=(-g12)*w[si2]; res[ti2]+=(-g22)*w[si2]
            else:
                res[ti1]+=k12*w[si2]; res[ti2]+=k22*w[si2]

    # ── Mass term (1/2) ∫ phi_i phi_j ds for Neumann unknowns ─────────────
    for te in range(Ne):
        ti1=conn[te,0]; ti2=conn[te,1]
        hLt=0.5*lengths[te]
        if is_dir[ti1] and is_dir[ti2]: continue   # both Dirichlet, no mass needed

        m11=0.0; m12=0.0; m21=0.0; m22=0.0
        for qi in range(Nq):
            sx=gq_pts[qi]; wx=gq_wts[qi]
            ph1x=0.5*(1.0-sx); ph2x=0.5*(1.0+sx)
            jac=wx*hLt
            m11+=0.5*ph1x*ph1x*jac; m12+=0.5*ph1x*ph2x*jac
            m21+=0.5*ph2x*ph1x*jac; m22+=0.5*ph2x*ph2x*jac

        if not is_dir[ti1]:
            res[ti1]+=m11*w[ti1]
            if not is_dir[ti2]: res[ti1]+=m12*w[ti2]
        if not is_dir[ti2]:
            if not is_dir[ti1]: res[ti2]+=m21*w[ti1]
            res[ti2]+=m22*w[ti2]

    return res


# ── Build RHS ─────────────────────────────────────────────────────────────────
@njit(cache=True)
def _build_rhs(nodes, conn, e_normals, lengths, is_dir,
               u_bnd, q_bnd, gq_pts, gq_wts, gq_pts2, gq_wts2):
    """
    RHS from known data:
      rhs[i] = -∫∫ phi_i K u_known_Dir ds_y ds_x   [known u on Dirichlet, DLP to RHS]
              -(1/2) ∫ phi_i u_known_Dir ds_x       [mass of known Dirichlet u]
              +∫∫ phi_i G q_known_Neu ds_y ds_x     [known q on Neumann, SLP stays RHS]
    """
    inv2pi = 1.0/(2.0*math.pi)
    Nn=nodes.shape[0]; Ne=conn.shape[0]; Nq=gq_pts.shape[0]
    rhs=np.zeros(Nn)

    for te in range(Ne):
        ti1=conn[te,0]; ti2=conn[te,1]
        t10=nodes[ti1,0]; t11=nodes[ti1,1]
        t20=nodes[ti2,0]; t21=nodes[ti2,1]
        hLt=0.5*lengths[te]

        for se in range(Ne):
            si1=conn[se,0]; si2=conn[se,1]
            s10=nodes[si1,0]; s11=nodes[si1,1]
            s20=nodes[si2,0]; s21=nodes[si2,1]
            Ls=lengths[se]; hLs=0.5*Ls
            en0=e_normals[se,0]; en1=e_normals[se,1]

            dir_si1=is_dir[si1]; dir_si2=is_dir[si2]

            shared=-1
            if   ti1==si1: shared=0
            elif ti1==si2: shared=1
            elif ti2==si1: shared=2
            elif ti2==si2: shared=3
            coincident=(te==se)

            # Compute kernel integrals (same logic as matvec)
            if coincident:
                g11,g12,g21,g22=_slp_coincident(
                    t10,t11,t20,t21,lengths[te],gq_pts,gq_wts)
                k11=0.0; k12=0.0; k21=0.0; k22=0.0
            elif shared>=0:
                if shared==0: wa_idx=0; wb_idx=0
                elif shared==1: wa_idx=0; wb_idx=1
                elif shared==2: wa_idx=1; wb_idx=0
                else:           wa_idx=1; wb_idx=1
                g11,g12,g21,g22=_slp_adjacent(
                    t10,t11,t20,t21,lengths[te],
                    s10,s11,s20,s21,Ls,
                    en0,en1,wa_idx,wb_idx,
                    gq_pts,gq_wts,gq_pts2,gq_wts2)
                k11,k12,k21,k22=_dlp_adjacent(
                    t10,t11,t20,t21,lengths[te],
                    s10,s11,s20,s21,Ls,
                    en0,en1,gq_pts2,gq_wts2)
            else:
                g11=0.0; g12=0.0; g21=0.0; g22=0.0
                k11=0.0; k12=0.0; k21=0.0; k22=0.0
                for qi in range(Nq):
                    sx=gq_pts[qi]; wx=gq_wts[qi]
                    ph1x=0.5*(1.0-sx); ph2x=0.5*(1.0+sx)
                    xc0=t10*ph1x+t20*ph2x; xc1=t11*ph1x+t21*ph2x
                    jx=wx*hLt
                    for qj in range(Nq):
                        sy=gq_pts[qj]; wy=gq_wts[qj]
                        ph1y=0.5*(1.0-sy); ph2y=0.5*(1.0+sy)
                        yy0=s10*ph1y+s20*ph2y; yy1=s11*ph1y+s21*ph2y
                        jy=wy*hLs
                        dx=xc0-yy0; dy=xc1-yy1; r2=dx*dx+dy*dy
                        if r2<1e-28: continue
                        fac=jx*jy
                        G_k=-inv2pi*0.5*math.log(r2)
                        K_k=inv2pi*(dx*en0+dy*en1)/r2
                        g11+=ph1x*G_k*ph1y*fac; g12+=ph1x*G_k*ph2y*fac
                        g21+=ph2x*G_k*ph1y*fac; g22+=ph2x*G_k*ph2y*fac
                        k11+=ph1x*K_k*ph1y*fac; k12+=ph1x*K_k*ph2y*fac
                        k21+=ph2x*K_k*ph1y*fac; k22+=ph2x*K_k*ph2y*fac

            # RHS contribution from si1
            if dir_si1:
                # Known u: DLP → rhs -= K * u_known
                rhs[ti1]-=k11*u_bnd[si1]; rhs[ti2]-=k21*u_bnd[si1]
            else:
                # Known q: SLP → rhs += G * q_known
                rhs[ti1]+=g11*q_bnd[si1]; rhs[ti2]+=g21*q_bnd[si1]

            # RHS contribution from si2
            if dir_si2:
                rhs[ti1]-=k12*u_bnd[si2]; rhs[ti2]-=k22*u_bnd[si2]
            else:
                rhs[ti1]+=g12*q_bnd[si2]; rhs[ti2]+=g22*q_bnd[si2]

    # ── Mass term for known Dirichlet u ────────────────────────────────────
    for te in range(Ne):
        ti1=conn[te,0]; ti2=conn[te,1]
        hLt=0.5*lengths[te]
        b1=0.0; b2=0.0
        for qi in range(Nq):
            sx=gq_pts[qi]; wx=gq_wts[qi]
            ph1x=0.5*(1.0-sx); ph2x=0.5*(1.0+sx); jac=wx*hLt
            u_x=0.0
            if is_dir[ti1]: u_x+=ph1x*u_bnd[ti1]
            if is_dir[ti2]: u_x+=ph2x*u_bnd[ti2]
            b1-=0.5*ph1x*u_x*jac; b2-=0.5*ph2x*u_x*jac
        rhs[ti1]+=b1; rhs[ti2]+=b2

    return rhs


# ── Interior evaluation ───────────────────────────────────────────────────────
@njit(cache=True)
def _eval_interior(x_pts, nodes, conn, e_normals, lengths,
                   u_bnd, q_bnd, gq_pts, gq_wts):
    inv2pi=1.0/(2.0*math.pi)
    Np=x_pts.shape[0]; Ne=conn.shape[0]; Nq=gq_pts.shape[0]
    u=np.zeros(Np)
    for p in range(Np):
        xc0=x_pts[p,0]; xc1=x_pts[p,1]; val=0.0
        for e in range(Ne):
            i1=conn[e,0]; i2=conn[e,1]
            p10=nodes[i1,0]; p11=nodes[i1,1]
            p20=nodes[i2,0]; p21=nodes[i2,1]
            hL=0.5*lengths[e]; en0=e_normals[e,0]; en1=e_normals[e,1]
            for q in range(Nq):
                s=gq_pts[q]; wq=gq_wts[q]
                ph1=0.5*(1.0-s); ph2=0.5*(1.0+s)
                yy0=p10*ph1+p20*ph2; yy1=p11*ph1+p21*ph2
                u_q=u_bnd[i1]*ph1+u_bnd[i2]*ph2
                q_q=q_bnd[i1]*ph1+q_bnd[i2]*ph2
                jac=wq*hL
                dx=xc0-yy0; dy=xc1-yy1; r2=dx*dx+dy*dy
                if r2<1e-28: continue
                G_k=-inv2pi*0.5*math.log(r2)
                K_k=inv2pi*(dx*en0+dy*en1)/r2
                val+=(G_k*q_q-K_k*u_q)*jac
        u[p]=val
    return u


# ── Warmup ────────────────────────────────────────────────────────────────────
def _warmup():
    nd,co,en,le,di=make_boundary(4)
    ub=np.ones(8); qb=np.ones(8); w0=np.ones(8)
    _matvec(w0,nd,co,en,le,di,GQ_PTS,GQ_WTS,GQ_PTS2,GQ_WTS2)
    _build_rhs(nd,co,en,le,di,ub,qb,GQ_PTS,GQ_WTS,GQ_PTS2,GQ_WTS2)
    _eval_interior(X_int[:2].copy(),nd,co,en,le,ub,qb,GQ_PTS,GQ_WTS)

_warmup()

# ── Refinement study ──────────────────────────────────────────────────────────
N_values=[160, 320, 640, 1280, 2560, 5120]
errors=[]; hs=[]
prev_err=None; prev_h=None

print(f"{'N':<6}{'Unknowns':<12}{'GMRES':<9}{'Rel L2 Error':<16}"
      f"{'Conv Rate':<14}{'Setup':>8}{'Solve':>9}{'Eval':>9}{'Total':>9}")
print("-"*100)

for N in N_values:
    t0=time.perf_counter()
    nodes,conn,e_normals,lengths,is_dirichlet=make_boundary(N)
    Nn=nodes.shape[0]
    u_bnd=np.array([u_exact(*nodes[i])    for i in range(Nn)])
    q_bnd=np.array([dudr_exact(*nodes[i]) for i in range(Nn)])

    _nd=np.ascontiguousarray(nodes); _co=np.ascontiguousarray(conn)
    _en=np.ascontiguousarray(e_normals); _le=np.ascontiguousarray(lengths)
    _di=is_dirichlet.copy(); _ub=u_bnd.copy(); _qb=q_bnd.copy()

    rhs=_build_rhs(_nd,_co,_en,_le,_di,_ub,_qb,GQ_PTS,GQ_WTS,GQ_PTS2,GQ_WTS2)
    t_setup=time.perf_counter()-t0

    iters=[0]
    def _cb(rk): iters[0]+=1

    def mv(w):
        return _matvec(w,_nd,_co,_en,_le,_di,GQ_PTS,GQ_WTS,GQ_PTS2,GQ_WTS2)

    A=LinearOperator((Nn,Nn),matvec=mv,dtype=np.float64)
    t1=time.perf_counter()
    w_sol,info=gmres(A,rhs,rtol=1e-10,atol=1e-12,
                     maxiter=500,restart=100,
                     callback=_cb,callback_type='pr_norm')
    t_solve=time.perf_counter()-t1

    u_sol=_ub.copy(); q_sol=_qb.copy()
    for i in range(Nn):
        if is_dirichlet[i]: q_sol[i]=w_sol[i]
        else:               u_sol[i]=w_sol[i]

    t2=time.perf_counter()
    u_num=_eval_interior(X_int,_nd,_co,_en,_le,
                         np.ascontiguousarray(u_sol),
                         np.ascontiguousarray(q_sol),
                         GQ_PTS,GQ_WTS)
    t_eval=time.perf_counter()-t2

    rel_err=np.linalg.norm(u_num-u_ref)/np.linalg.norm(u_ref)
    h=2*_PI/N; errors.append(rel_err); hs.append(h)

    rate_str="N/A"
    if prev_err is not None and rel_err>0 and prev_err>0:
        rate=math.log(prev_err/rel_err)/math.log(prev_h/h)
        rate_str=f"{rate:.2f}"

    t_tot=t_setup+t_solve+t_eval
    print(f"{N:<6}{Nn:<12}{iters[0]:<9}{rel_err:<16.4e}"
          f"{rate_str:<14}{t_setup:>7.2f}s{t_solve:>8.2f}s"
          f"{t_eval:>8.2f}s{t_tot:>8.2f}s")
    prev_err=rel_err; prev_h=h

_p=np.polyfit(np.log(np.array(hs)),np.log(np.array(errors)),1)
print(f"\nObserved convergence order (least-squares fit): {_p[0]:.3f}")