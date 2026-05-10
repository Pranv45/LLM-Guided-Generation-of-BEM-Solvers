import numpy as np
import pandas as pd
import time

# ── Eigenvalues of the DLP operator on the unit circle ───────────────────────
# For the double-layer potential on the unit circle, the Fourier modes
# are eigenfunctions. The operator K has eigenvalues:
#   λ_0 = -1/2  (mode 0, but handled by 1/2 I + K → 0, set separately)
#   λ_k = -1/2  * sign...
# Exact spectrum: for the interior DLP on unit circle,
#   (1/2 I + K) e^{ikθ} = λ_k e^{ikθ}
# where λ_k = 1/2 + κ_k and the DLP eigenvalues κ_k = -1/2 for k=0,
# and κ_k = sign convention below.
#
# Derivation: K[μ](x) = -(1/2π) ∫ (x-y)·n_y/|x-y|² μ(y) ds_y
# On unit circle, y=(cosφ,sinφ), n_y=y, x=(cosθ,sinθ):
#   K[e^{inφ}](θ) = κ_n e^{inθ}
# The known result for the unit circle:
#   κ_n = -1/2  for n=0  (interior)
#   κ_n = +|n|/2 ...
# Let us derive carefully.
# K(x,y) = -(1/2π)(x-y)·n_y/|x-y|²
# x=e^{iθ}, y=e^{iφ} (complex notation), n_y = e^{iφ}
# (x-y)·n_y = Re[(x-y) conj(n_y)] = Re[(e^{iθ}-e^{iφ})e^{-iφ}]
#           = Re[e^{i(θ-φ)} - 1]  = cos(θ-φ) - 1
# |x-y|² = 2 - 2cos(θ-φ)
# K(x,y) = -(1/2π)(cos(θ-φ)-1)/(2-2cos(θ-φ)) = (1/2π) * 1/2 = 1/(4π) ... wait
# More carefully: (cos α - 1)/(2 - 2cos α) = (cos α - 1)/(2(1-cos α)) = -1/2
# So K(x,y) = -(1/2π)*(-1/2) = 1/(4π)  — this is constant!
# Therefore K is a rank-1 operator: K[μ] = (1/4π) ∫ μ ds = (1/2) μ̂_0
# Eigenvalues: κ_0 = 1/2,  κ_k = 0 for k≠0
# So (1/2 I + K): eigenvalue for k=0 is 1/2+1/2=1, for k≠0 is 1/2+0=1/2
# Hence μ̂_k = f̂_k / (1/2) for k≠0,  μ̂_0 = f̂_0 / 1

def spectral_solve(f, M):
    """
    Solve (1/2 I + K) μ = f in Fourier space.
    Eigenvalues of (1/2 I + K) on unit circle:
      k=0  → 1   (from κ_0 = 1/2)
      k≠0  → 1/2 (from κ_k = 0)
    Returns μ (real, length M).
    """
    f_hat = np.fft.rfft(f)
    mu_hat = f_hat.copy().astype(complex)
    # k=0: divide by 1
    mu_hat[0] = f_hat[0] / 1.0
    # k>0: divide by 1/2  i.e. multiply by 2
    mu_hat[1:] = f_hat[1:] / 0.5
    mu = np.fft.irfft(mu_hat, n=M)
    return mu

def evaluate_interior(mu, theta_b, eval_pts, w):
    """
    u(x) = ∫_Γ K(x,y) μ(y) ds_y
         = Σ_j K(x, y_j) μ_j w
    K(x,y) = -(1/2π) (x-y)·n_y / |x-y|²
    y_j = (cos θ_j, sin θ_j),  n_j = y_j,  w = 2π/M
    Chunked over eval_pts to control memory.
    """
    N      = eval_pts.shape[0]
    result = np.zeros(N)
    inv2pi = 1.0 / (2.0 * np.pi)

    y  = np.stack([np.cos(theta_b), np.sin(theta_b)], axis=1)  # (M,2)
    ny = y.copy()

    # chunk over eval pts
    chunk = max(1, min(N, int(256*1024**2 / (len(theta_b)*8*4))))
    for s in range(0, N, chunk):
        e   = min(s+chunk, N)
        xp  = eval_pts[s:e]               # (C,2)
        dx  = xp[:,None,0] - y[None,:,0]  # (C,M)
        dy_ = xp[:,None,1] - y[None,:,1]
        dist2 = dx*dx + dy_*dy_
        dotn  = dx*ny[None,:,0] + dy_*ny[None,:,1]
        K     = -inv2pi * dotn / dist2     # (C,M)
        result[s:e] = (K * mu[None,:]).sum(axis=1) * w

    return result

def run_spectral_bem(M, n=3):
    t_total = time.perf_counter()

    # Boundary discretisation
    theta = 2.0*np.pi*np.arange(M)/M
    f     = np.cos(n*theta)
    w     = 2.0*np.pi/M

    # Spectral solve
    t_solve = time.perf_counter()
    mu      = spectral_solve(f, M)
    solve_time = time.perf_counter() - t_solve

    # Evaluation grid
    Nx = Ny = 120
    xx, yy = np.linspace(-0.9, 0.9, Nx), np.linspace(-0.9, 0.9, Ny)
    X, Y   = np.meshgrid(xx, yy)
    mask   = (X**2 + Y**2) < 0.9**2
    eval_pts = np.column_stack([X[mask], Y[mask]])

    t_eval = time.perf_counter()
    u_num  = evaluate_interior(mu, theta, eval_pts, w)
    eval_time = time.perf_counter() - t_eval

    r_  = np.sqrt(eval_pts[:,0]**2 + eval_pts[:,1]**2)
    th_ = np.arctan2(eval_pts[:,1], eval_pts[:,0])
    u_ex = r_**n * np.cos(n*th_)

    err   = u_num - u_ex
    rel_l2 = np.linalg.norm(err) / np.linalg.norm(u_ex)
    linf   = np.max(np.abs(err))

    return dict(
        M             = M,
        solve_time    = solve_time,
        eval_time     = eval_time,
        total_time    = time.perf_counter() - t_total,
        rel_L2_error  = rel_l2,
        Linf_error    = linf,
    )

# ── Convergence sweep ─────────────────────────────────────────────────────────
M_values = [100, 200, 400, 800, 1600]
rows = []
prev = None
for M in M_values:
    r = run_spectral_bem(M)
    r['rate'] = np.log2(prev/r['rel_L2_error']) if prev else float('nan')
    prev = r['rel_L2_error']
    rows.append(r)

df = pd.DataFrame(rows).set_index('M')
df.index = df.index.map(lambda x: f"{x:,}")
df.columns = ['Solve(s)', 'Eval(s)', 'Total(s)', 'Rel L2', 'Linf', 'Rate']

# format
fmt = {'Solve(s)':'{:.4f}','Eval(s)':'{:.3f}','Total(s)':'{:.3f}',
       'Rel L2':'{:.3e}','Linf':'{:.3e}','Rate':'{:.2f}'}
print("\n2D Spectral BEM — Interior Laplace on unit circle, u=cos(3θ)\n")
print(df.to_string(float_format=lambda x: f"{x:.3e}" if abs(x)<0.01 else f"{x:.4f}"))
print()