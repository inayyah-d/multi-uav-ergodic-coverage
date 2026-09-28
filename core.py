"""
core.py — Shared environment, GP, REMC, and simulation loop
============================================================
All simulation files import from here. Do not run directly.

Paper: Mendoza et al. (2026) — Decentralized Ergodic Coverage Control
       in Unknown Time-Varying Environments
"""

import numpy as np
from collections import defaultdict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# DEFAULT PARAMETERS
# ─────────────────────────────────────────────────────────

GRID_SIZE        = 8
N_UAVS           = 3
T_FINAL          = 600
TAU_GP           = 20
TAU_P            = 20
R_SENSE          = 1.5
R_COMM           = 2.5
SENSOR_NOISE_STD = 0.05
BETA             = 1.0
N_RUNS           = 5

ROI_CENTERS = [(2, 2), (2, 5), (5, 2), (5, 5)]
ROI_WEIGHT  = 5.0
BASE_WEIGHT = 1.0
NO_FLY      = {(3, 3), (3, 4), (4, 3), (4, 4)}


# ─────────────────────────────────────────────────────────
# ENVIRONMENT  (Section 3.1)
# ─────────────────────────────────────────────────────────

def region_to_xy(r, grid_size=GRID_SIZE):
    return (r // grid_size, r % grid_size)

def xy_to_region(x, y, grid_size=GRID_SIZE):
    return x * grid_size + y

def build_graph(grid_size=GRID_SIZE, no_fly=NO_FLY):
    """Build undirected graph G = (R, E). Returns neighbors dict and accessible list."""
    accessible = []
    for r in range(grid_size * grid_size):
        x, y = region_to_xy(r, grid_size)
        if (x, y) not in no_fly:
            accessible.append(r)

    accessible_set = set(accessible)
    neighbors = defaultdict(list)
    for r in accessible:
        x, y = region_to_xy(r, grid_size)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx_, ny_ = x+dx, y+dy
            if 0 <= nx_ < grid_size and 0 <= ny_ < grid_size:
                nr = xy_to_region(nx_, ny_, grid_size)
                if nr in accessible_set:
                    neighbors[r].append(nr)
    return neighbors, accessible

def build_true_distribution(accessible, grid_size=GRID_SIZE):
    """
    True target distribution rho* (Eq. 1).
    Gaussian importance bumps centered on ROI_CENTERS.
    Returns: rho_star (normalized, shape n), phi (full grid, shape grid^2)
    """
    phi = np.ones(grid_size * grid_size) * BASE_WEIGHT
    for (rx, ry) in ROI_CENTERS:
        for r in accessible:
            x, y = region_to_xy(r, grid_size)
            dist = np.sqrt((x - rx)**2 + (y - ry)**2)
            phi[r] += ROI_WEIGHT * np.exp(-0.5 * dist**2)
    for r in range(grid_size * grid_size):
        x, y = region_to_xy(r, grid_size)
        if (x, y) in NO_FLY:
            phi[r] = 0.0
    phi_acc = np.array([phi[r] for r in accessible])
    rho_star = phi_acc / phi_acc.sum()
    return rho_star, phi

def get_roi_region_indices(accessible, grid_size=GRID_SIZE):
    """
    Return indices (into accessible[]) of the regions closest to each ROI center.
    Used for ROI discovery time tracking.
    """
    roi_indices = []
    for (rx, ry) in ROI_CENTERS:
        best_i, best_dist = 0, float('inf')
        for i, r in enumerate(accessible):
            x, y = region_to_xy(r, grid_size)
            d = np.sqrt((x - rx)**2 + (y - ry)**2)
            if d < best_dist:
                best_dist = d
                best_i = i
        roi_indices.append(best_i)
    return roi_indices


# ─────────────────────────────────────────────────────────
# REMC POLICY  (Section 5.3)
# ─────────────────────────────────────────────────────────

def compute_remc(rho_target, accessible, neighbors):
    """
    Metropolis-Hastings construction of ergodic Markov chain.
    Closed-form approximation to REMC (Wong et al., 2025).

    P(j|i) = min(1, rho_j * deg_i / (rho_i * deg_j))  for neighbors i,j
    P(i|i) = 1 - sum_{j!=i} P(j|i)

    Guarantees: stationary distribution = rho_target, valid transition matrix.
    """
    n = len(accessible)
    idx = {r: i for i, r in enumerate(accessible)}
    P = np.zeros((n, n))

    for r in accessible:
        i = idx[r]
        rho_i = max(rho_target[i], 1e-10)
        deg_i = len(neighbors[r])
        for r2 in neighbors[r]:
            j = idx[r2]
            rho_j = max(rho_target[j], 1e-10)
            deg_j = len(neighbors[r2])
            P[j, i] = min(1.0, (rho_j * deg_i) / (rho_i * deg_j))
        P[i, i] = max(0.0, 1.0 - P[:, i].sum())
    return P


# ─────────────────────────────────────────────────────────
# GP BELIEF UPDATE  (Algorithm 2)
# ─────────────────────────────────────────────────────────

def gp_ucb_update(dataset, accessible, beta=BETA, grid_size=GRID_SIZE):
    """
    Fit GP to observation dataset. Return UCB belief.

    phi_bar(r) = mu(r) + beta * sigma(r)   [Eq. 6]
    rho_bar    = normalize(phi_bar)

    Returns: phi_ucb (unnormalized), rho_bar (normalized)
    """
    n = len(accessible)
    if len(dataset) < 3:
        return np.ones(n), np.ones(n) / n

    coords  = np.array([list(region_to_xy(accessible[i], grid_size)) for i in range(n)])
    X_train = np.array([coords[d[0]] for d in dataset])
    y_train = np.array([d[1] for d in dataset])

    kernel = RBF(length_scale=2.0) + WhiteKernel(noise_level=SENSOR_NOISE_STD**2)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0)
    gp.fit(X_train, y_train)

    mu, sigma = gp.predict(coords, return_std=True)
    phi_ucb = mu + beta * sigma
    phi_ucb = np.maximum(phi_ucb, 1e-8)
    rho_bar = phi_ucb / phi_ucb.sum()
    return phi_ucb, rho_bar


# ─────────────────────────────────────────────────────────
# COORDINATION FUNCTIONS  (your contribution)
# ─────────────────────────────────────────────────────────

def coordinated_target_v1(phi_bar_m, neighbors_empirical, alpha):
    """
    V1 — Original subtraction (uniform across all regions).

    phi_coord(r) = phi_bar_m(r) - alpha * avg_j[rho_hat_j(r)] * ||phi_bar_m||_1
    rho_bar_m    = normalize(clip(phi_coord, 1e-8))
    """
    subtract = np.zeros_like(phi_bar_m)
    if len(neighbors_empirical) > 0:
        subtract = np.mean(neighbors_empirical, axis=0)

    phi_coord = phi_bar_m - alpha * subtract * phi_bar_m.sum()
    phi_coord = np.maximum(phi_coord, 1e-8)
    return phi_coord / phi_coord.sum()


def coordinated_target_v2(phi_bar_m, rho_bar_m, neighbors_empirical, alpha):
    """
    V2 — Importance-weighted subtraction (protects high-importance regions).

    phi_coord(r) = phi_bar_m(r) - alpha * (1 - rho_bar_m(r)) * avg_j[rho_hat_j(r)] * ||phi_bar_m||_1
    rho_bar_m    = normalize(clip(phi_coord, 1e-8))

    The (1 - rho_bar_m(r)) weight means:
      - High-importance regions (large rho_bar_m) are barely subtracted
      - Low-importance regions are subtracted more aggressively
    This removes the coverage artifact from V1.
    """
    subtract = np.zeros_like(phi_bar_m)
    if len(neighbors_empirical) > 0:
        subtract = np.mean(neighbors_empirical, axis=0)

    importance_weight = 1.0 - rho_bar_m   # protects high-importance regions
    phi_coord = phi_bar_m - alpha * importance_weight * subtract * phi_bar_m.sum()
    phi_coord = np.maximum(phi_coord, 1e-8)
    return phi_coord / phi_coord.sum()


# ─────────────────────────────────────────────────────────
# MAIN SIMULATION LOOP  (Algorithm 1)
# ─────────────────────────────────────────────────────────

def run_simulation(
    mode="baseline",     # "baseline" | "v1" | "v2"
    alpha=0.2,
    seed=42,
    t_final=T_FINAL,
    tau_gp=TAU_GP,
    tau_p=TAU_P,
    n_uavs=N_UAVS,
    grid_size=GRID_SIZE,
):
    """
    Run Algorithm 1 for one episode.

    mode:
      "baseline" — Mendoza Algorithm 1 unmodified (alpha ignored)
      "v1"       — Original coordination subtraction
      "v2"       — Importance-weighted coordination subtraction

    Returns dict with:
      regret        — time-averaged regret curve, shape (t_final,)
      belief_err    — team belief error curve, shape (t_final,)
      overlap       — spatial overlap curve, shape (t_final,)
      rho_hat       — list of M final empirical distributions
      rho_star      — true target distribution
      accessible    — list of accessible region indices
      roi_discovery — dict {roi_idx: first_timestep_discovered}, -1 if not found
    """
    rng = np.random.default_rng(seed)
    neighbors, accessible = build_graph(grid_size)
    n = len(accessible)
    idx = {r: i for i, r in enumerate(accessible)}
    rho_star, _ = build_true_distribution(accessible, grid_size)
    roi_indices = get_roi_region_indices(accessible, grid_size)

    # Initialize
    positions  = list(rng.choice(n, size=n_uavs, replace=False))
    rho_hat    = [np.zeros(n) for _ in range(n_uavs)]
    datasets   = [[] for _ in range(n_uavs)]
    rho_bar    = [np.ones(n) / n for _ in range(n_uavs)]
    phi_bar    = [np.ones(n) for _ in range(n_uavs)]
    P          = [compute_remc(np.ones(n)/n, accessible, neighbors) for _ in range(n_uavs)]

    # ROI discovery tracking
    roi_discovery = {roi_i: -1 for roi_i in roi_indices}

    regret_history = []
    belief_err_history = []
    overlap_history = []
    cumulative_regret = 0.0

    for k in range(t_final):

        # ── Step 1: Observe ──
        for m in range(n_uavs):
            r_m = accessible[positions[m]]
            x_m, y_m = region_to_xy(r_m, grid_size)
            for i, r in enumerate(accessible):
                x, y = region_to_xy(r, grid_size)
                if np.sqrt((x-x_m)**2 + (y-y_m)**2) <= R_SENSE:
                    obs = rho_star[i] + rng.normal(0, SENSOR_NOISE_STD)
                    datasets[m].append((i, float(obs)))

            # ROI discovery: check if this UAV is at an ROI
            for roi_i in roi_indices:
                if roi_discovery[roi_i] == -1 and positions[m] == roi_i:
                    roi_discovery[roi_i] = k

        # ── Step 2: Communicate ──
        comm_neighbors = [[] for _ in range(n_uavs)]
        for m in range(n_uavs):
            r_m = accessible[positions[m]]
            x_m, y_m = region_to_xy(r_m, grid_size)
            for l in range(n_uavs):
                if l == m:
                    continue
                r_l = accessible[positions[l]]
                x_l, y_l = region_to_xy(r_l, grid_size)
                if np.sqrt((x_m-x_l)**2 + (y_m-y_l)**2) <= R_COMM:
                    comm_neighbors[m].append(l)
                    for obs in datasets[l][-5:]:
                        datasets[m].append(obs)

        # ── Step 3: Belief update ──
        if k % tau_gp == 0:
            for m in range(n_uavs):
                D_m = datasets[m][-200:]
                phi_bar[m], rho_bar_base = gp_ucb_update(D_m, accessible, grid_size=grid_size)

                if mode == "baseline" or len(comm_neighbors[m]) == 0:
                    rho_bar[m] = rho_bar_base

                elif mode == "v1":
                    neighbor_rhos = [rho_hat[l] for l in comm_neighbors[m]]
                    rho_bar[m] = coordinated_target_v1(phi_bar[m], neighbor_rhos, alpha)

                elif mode == "v2":
                    neighbor_rhos = [rho_hat[l] for l in comm_neighbors[m]]
                    rho_bar[m] = coordinated_target_v2(
                        phi_bar[m], rho_bar_base, neighbor_rhos, alpha
                    )

        # ── Step 4: Policy update ──
        if k % tau_p == 0:
            for m in range(n_uavs):
                P[m] = compute_remc(rho_bar[m], accessible, neighbors)

        # ── Step 5: Move ──
        for m in range(n_uavs):
            col   = positions[m]
            probs = P[m][:, col]
            probs = np.maximum(probs, 0)
            s     = probs.sum()
            probs = probs / s if s > 1e-10 else np.ones(n) / n
            positions[m] = rng.choice(n, p=probs)

        # ── Update empirical distributions ──
        for m in range(n_uavs):
            indicator = np.zeros(n)
            indicator[positions[m]] = 1.0
            rho_hat[m] = (k * rho_hat[m] + indicator) / (k + 1)

        # ── Metrics ──
        rho_hat_team = np.mean(rho_hat, axis=0)
        l1_err = np.sum(np.abs(rho_hat_team - rho_star))
        cumulative_regret += l1_err
        regret_history.append(cumulative_regret / (k + 1))

        rho_bar_team = np.mean(rho_bar, axis=0)
        belief_err_history.append(np.sum(np.abs(rho_bar_team - rho_star)))

        n_unique = len(set(positions))
        overlap_history.append(1.0 - (n_unique / n_uavs))

    return {
        "regret":        np.array(regret_history),
        "belief_err":    np.array(belief_err_history),
        "overlap":       np.array(overlap_history),
        "rho_hat":       rho_hat,
        "rho_star":      rho_star,
        "accessible":    accessible,
        "roi_discovery": roi_discovery,
    }


def run_multiple(mode, alpha=0.2, n_runs=N_RUNS, tau_gp=TAU_GP, tau_p=TAU_P, n_uavs=N_UAVS):
    """Run N_RUNS seeds and return mean/std for regret, belief_err, overlap, roi_discovery."""
    all_regret, all_belief, all_overlap = [], [], []
    all_roi = []

    for seed in range(n_runs):
        res = run_simulation(mode=mode, alpha=alpha, seed=seed,
                             tau_gp=tau_gp, tau_p=tau_p, n_uavs=n_uavs)
        all_regret.append(res["regret"])
        all_belief.append(res["belief_err"])
        all_overlap.append(res["overlap"])
        all_roi.append(res["roi_discovery"])

    # ROI discovery: mean and std across seeds per ROI
    roi_keys = list(all_roi[0].keys())
    roi_times = {}
    for k in roi_keys:
        times = [r[k] for r in all_roi]
        # Replace -1 (not found) with T_FINAL as worst case
        times = [t if t >= 0 else T_FINAL for t in times]
        roi_times[k] = (np.mean(times), np.std(times))

    return {
        "regret":     (np.mean(all_regret, axis=0), np.std(all_regret, axis=0)),
        "belief_err": (np.mean(all_belief, axis=0), np.std(all_belief, axis=0)),
        "overlap":    (np.mean(all_overlap, axis=0), np.std(all_overlap, axis=0)),
        "roi_times":  roi_times,
        "last_res":   res,   # last run, used for heatmaps
    }
