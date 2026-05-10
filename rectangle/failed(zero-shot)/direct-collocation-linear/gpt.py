import numpy as np
import time
from scipy.sparse.linalg import gmres
from numpy.polynomial.legendre import leggauss

# ============================================================
# Exact solution
# ============================================================

def u_exact(x, y):
    return (np.sinh(np.pi * x) / np.sinh(2*np.pi)) * np.cos(np.pi * y)

def grad_u_exact(x, y):
    ux = (np.pi * np.cosh(np.pi * x) / np.sinh(2*np.pi)) * np.cos(np.pi * y)
    uy = -(np.sinh(np.pi * x) / np.sinh(2*np.pi)) * np.pi * np.sin(np.pi * y)
    return ux, uy

# ============================================================
# Quadrature (order 8)
# ============================================================

nq = 8
gp, gw = leggauss(nq)

phi1 = (1 - gp)/2
phi2 = (1 + gp)/2

# ============================================================
# Boundary mesh (linear elements)
# ============================================================

def build_mesh(M):
    nodes = []
    elems = []
    btypes = []  # 'D' or 'N'

    # bottom (Neumann)
    xs = np.linspace(0,2,M+1)
    for i in range(M):
        n1 = len(nodes); nodes.append([xs[i],0])
        if i==M-1:
            n2 = len(nodes); nodes.append([xs[i+1],0])
        else:
            n2 = n1+1
        elems.append((n1,n2))
        btypes.append('N')

    offset = len(nodes)-1
    # right (Dirichlet)
    ys = np.linspace(0,1,M+1)
    for i in range(M):
        n1 = offset+i
        if i==0:
            nodes[n1]=[2,0]
        if i==M-1:
            n2 = len(nodes); nodes.append([2,1])
        else:
            n2 = n1+1; nodes.append([2,ys[i+1]])
        elems.append((n1,n2))
        btypes.append('D')
    offset = len(nodes)-1

    # top (Neumann)
    xs = np.linspace(2,0,M+1)
    for i in range(M):
        n1 = offset+i
        if i==0:
            nodes[n1]=[2,1]
        if i==M-1:
            n2 = len(nodes); nodes.append([0,1])
        else:
            n2 = n1+1; nodes.append([xs[i+1],1])
        elems.append((n1,n2))
        btypes.append('N')
    offset = len(nodes)-1

    # left (Dirichlet)
    ys = np.linspace(1,0,M+1)
    for i in range(M):
        n1 = offset+i
        if i==0:
            nodes[n1]=[0,1]
        if i==M-1:
            n2 = 0
        else:
            n2 = n1+1; nodes.append([0,ys[i+1]])
        elems.append((n1,n2))
        btypes.append('D')

    return np.array(nodes), elems, btypes

# ============================================================
# Kernels
# ============================================================

def G(r):
    return -(1/(2*np.pi))*np.log(r)

def dGdn(dx,dy,r,nx,ny):
    return -(1/(2*np.pi))*(dx*nx+dy*ny)/(r**2)

# ============================================================
# BEM Solver
# ============================================================

def solve_bem(M):

    t0 = time.time()
    nodes, elems, btypes = build_mesh(M)
    Nn = len(nodes)

    # Determine BC per node
    node_type = np.zeros(Nn,dtype=int)  # 0=Dirichlet,1=Neumann
    u_known = np.zeros(Nn)
    q_known = np.zeros(Nn)

    for (n1,n2),bt in zip(elems,btypes):
        for n in (n1,n2):
            x,y = nodes[n]
            if bt=='D':
                node_type[n]=0
                u_known[n]=u_exact(x,y)
            else:
                node_type[n]=1
                q_known[n]=0.0

    A = np.zeros((Nn,Nn))
    rhs = np.zeros(Nn)

    # Precompute element geometry
    elem_data=[]
    for (n1,n2) in elems:
        x1,y1=nodes[n1]
        x2,y2=nodes[n2]
        tx = x2-x1; ty=y2-y1
        length=np.hypot(tx,ty)
        nx= ty/length; ny=-tx/length
        elem_data.append((n1,n2,x1,y1,x2,y2,nx,ny,length))

    # Assembly
    for i in range(Nn):
        xi,yi = nodes[i]
        for e,(n1,n2,x1,y1,x2,y2,nx,ny,L) in enumerate(elem_data):

            # mapped quadrature points
            xq = x1*phi1 + x2*phi2
            yq = y1*phi1 + y2*phi2
            dx = xi - xq
            dy = yi - yq
            r = np.hypot(dx,dy)

            # handle singular case
            if i==n1 or i==n2:
                r = np.maximum(r,1e-14)

            Gval = G(r)
            Hval = dGdn(dx,dy,r,nx,ny)

            J = L/2

            G1 = np.sum(Gval*phi1*gw)*J
            G2 = np.sum(Gval*phi2*gw)*J
            H1 = np.sum(Hval*phi1*gw)*J
            H2 = np.sum(Hval*phi2*gw)*J

            # local contributions
            for loc,(node,Gc,Hc) in enumerate([(n1,G1,H1),(n2,G2,H2)]):
                if node_type[node]==0:  # Dirichlet → unknown q
                    A[i,node] += -Gc
                    rhs[i] += Hc*u_known[node]
                else:  # Neumann → unknown u
                    A[i,node] += Hc
                    rhs[i] += -Gc*q_known[node]

        # jump term
        A[i,i] += 0.5

    t_asm = time.time()

    sol,_ = gmres(A,rhs,atol=1e-10,restart=200, callback_type= 'legacy')
    t_solve = time.time()

    u = np.zeros(Nn); q=np.zeros(Nn)
    for i in range(Nn):
        if node_type[i]==0:
            u[i]=u_known[i]
            q[i]=sol[i]
        else:
            u[i]=sol[i]
            q[i]=q_known[i]

    # Interior evaluation
    nxg,nyg=40,20
    xs=np.linspace(0.05,1.95,nxg)
    ys=np.linspace(0.05,0.95,nyg)
    XX,YY=np.meshgrid(xs,ys)
    u_num=np.zeros_like(XX)

    for k,(n1,n2,x1,y1,x2,y2,nx,ny,L) in enumerate(elem_data):
        xq = x1*phi1 + x2*phi2
        yq = y1*phi1 + y2*phi2
        J=L/2
        for i in range(nxg):
            for j in range(nyg):
                dx=XX[j,i]-xq
                dy=YY[j,i]-yq
                r=np.hypot(dx,dy)
                Gval=G(r)
                Hval=dGdn(dx,dy,r,nx,ny)
                val=np.sum((Hval*(u[n1]*phi1+u[n2]*phi2)
                           -Gval*(q[n1]*phi1+q[n2]*phi2))*gw)*J
                u_num[j,i]+=val

    u_ex=u_exact(XX,YY)
    relL2=np.linalg.norm(u_num-u_ex)/np.linalg.norm(u_ex)

    t_eval=time.time()

    return Nn, t_asm-t0, t_solve-t_asm, t_eval-t_solve, t_eval-t0, relL2


# ============================================================
# Refinement study
# ============================================================

Ms=[10,20,40,80, 160]
prev=None
print("M  DOFs  Asm(s) Solve(s) Eval(s) Total(s) RelL2   Rate")
for M in Ms:
    dof,ta,ts,te,tt,err=solve_bem(M)
    rate=np.log(prev/err)/np.log(2) if prev is not None else np.nan
    print(f"{M:<3}{dof:<6}{ta:7.3f}{ts:9.3f}{te:9.3f}{tt:9.3f}  {err:8.2e}{rate:7.2f}")
    prev=err