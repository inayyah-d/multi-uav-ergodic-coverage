# Coordination-Aware Ergodic Coverage for Multi-UAV Search and Rescue

In multi-UAV search and rescue, drones that plan from similar beliefs end up
covering the same places, while other parts of the map go unobserved. This
project extends the decentralized ergodic coverage framework of
**Mendoza et al. (2026)**, which is the baseline for every comparison here. In
that framework, each UAV fits a Gaussian Process belief over an unknown
importance map and follows a Markov-chain policy whose stationary distribution
matches that belief. I add a **coordination term** that subtracts a weighted
version of the neighbors' empirical visitation from each UAV's target
distribution before it computes its policy. This needs no extra communication,
because neighbors already share this information. I compare two formulations,
uniform subtraction (V1) and importance-weighted subtraction (V2), against the
unmodified baseline in simulation. I measure regret, spatial overlap and ROI
discovery time, and run extra studies on update frequency and team size.

*EE290 final project, UC Berkeley, Spring 2026.*

## Headline result

With α = 0.2, **V2 cuts spatial overlap by 73%** (0.026 → 0.007) and
**improves last-ROI discovery time by 32%** (60.8 → 41.0 steps). Its final
regret stays within ~3% of the baseline (0.604 vs. 0.596).

![Belief error and spatial overlap over time for baseline, V1 and V2](outputs/fig_belief_overlap.png)

*Belief error (left) and spatial overlap (right) over time. M = 3 UAVs,
α = 0.2, averaged over 5 seeds, with ±1 std shaded.*

| Metric (M = 3, α = 0.2, τ = 20, 5 seeds) | Baseline | V1    | V2    |
|------------------------------------------|---------:|------:|------:|
| Final regret                             | 0.596    | 0.610 | 0.604 |
| Mean ROI discovery (steps)               | 30.0     | 29.5  | 30.9  |
| Last ROI discovery (steps)               | 60.8     | 41.2  | 41.0  |
| Mean spatial overlap                     | 0.026    | 0.009 | 0.007 |

*Table I of the report.*

## Report

The full write-up is in **[`report.pdf`](report.pdf)**. It covers the
problem formulation, related work, all experiments (α selection, update
frequency, scalability) and limitations.

## Method

### Baseline: Mendoza et al., Algorithm 1

> M. G. Mendoza, V. M. Tuck, C. Maheshwari, and S. Sastry, "Decentralized
> ergodic coverage control in unknown time-varying environments,"
> arXiv:2604.04280, 2026.

The environment is a graph of $N$ regions with an unknown importance map
$\phi^\star$. Each UAV $m$ runs the following steps:

1. It observes regions within $R_\text{sense}$ and shares data with neighbors
   within $R_\text{comm}$.
2. Every $\tau_\text{GP}$ steps, it fits a GP to its dataset and builds a UCB
   belief target:
   $\bar\phi^m_k(r) = \mu^m_k(r) + \beta\,\sigma^m_k(r)$.
   Normalizing this gives $\bar\rho^m_k$.
3. Every $\tau_P$ steps, it computes a transition matrix $P^m_k$ with
   stationary distribution $\bar\rho^m_k$ using REMC.
4. It samples its next region from $P^m_k$.

The team minimizes time-averaged regret,
$\text{Regret}(K) = \frac{1}{K}\sum_{k=1}^{K}\mathbb{E}\big[\lVert\hat\rho_k - \rho^\star_k\rVert_1\big]$,
where $\hat\rho_k$ is the team's empirical visitation distribution.

**The coordination gap.** Shared observations produce similar GP posteriors.
Similar posteriors produce similar belief targets and similar policies, so the
UAVs cluster on the same high-weight regions.

### Coordination-aware modification

I change Line 11 of Algorithm 1 (the belief-target step). Let $\hat\rho^j_k$
be the empirical visitation distribution of neighbor
$j \in \mathcal{N}^m_k$, which the communication step already provides.

**V1: uniform subtraction**

$$
\bar\phi^m_{k,\text{coord}}(r) = \bar\phi^m_k(r) - \alpha \cdot \frac{1}{|\mathcal{N}^m_k|}\sum_{j\in\mathcal{N}^m_k}\hat\rho^j_k(r)\cdot\lVert\bar\phi^m_k\rVert_1
$$

**V2: importance-weighted subtraction.** The $(1 - \bar\rho^m_k(r))$ weight
protects high-importance regions from being down-weighted:

$$
\bar\phi^m_{k,\text{coord}}(r) = \bar\phi^m_k(r) - \alpha\,\big(1-\bar\rho^m_k(r)\big)\cdot\frac{1}{|\mathcal{N}^m_k|}\sum_{j\in\mathcal{N}^m_k}\hat\rho^j_k(r)\cdot\lVert\bar\phi^m_k\rVert_1
$$

Both variants then produce a valid REMC input:

$$
\bar\rho^m_k \leftarrow \text{normalize}\big(\text{clip}(\bar\phi^m_{k,\text{coord}},\,10^{-8})\big)
$$

With α = 0, both reduce to the baseline exactly. REMC is implemented with a
closed-form Metropolis–Hastings construction. This preserves ergodicity but
may mix more slowly than the optimized REMC.

## How to run

```bash
pip install -r requirements.txt

python simulation_main.py         # Baseline vs V1 vs V2  (Table I, Figs. 1–4)
python experiment_tau_sweep.py    # update-period sweep    (Fig. 6)
python experiment_scalability.py  # team-size sweep        (Fig. 7)
```

Figures are written to `./outputs/`:

| Script | Output files |
|---|---|
| `simulation_main.py` | `fig_regret.png`, `fig_belief_overlap.png`, `fig_roi_discovery.png`, `fig_heatmaps.png` |
| `experiment_tau_sweep.py` | `fig_tau_sweep.png`, `fig_tau_curves.png` |
| `experiment_scalability.py` | `fig_scalability.png` |

`core.py` holds the shared environment, GP belief update, REMC policy, V1/V2
coordination functions and simulation loop. It is imported by the scripts and
is not run on its own.

The preliminary α sweep (Fig. 5 in the report) has no script in this repo.

## Parameters

Shared defaults from `core.py`:

| Parameter | Symbol | Value |
|---|---|---|
| Grid size | — | 8 × 8 (N = 64 regions) |
| No-fly zone | — | 2 × 2 block at center: (3,3), (3,4), (4,3), (4,4) |
| ROI centers | — | (2,2), (2,5), (5,2), (5,5) |
| ROI / base weight | — | 5.0 / 1.0 |
| Number of UAVs | M | 3 |
| Episode length | T | 600 |
| GP update period | τ_GP | 20 |
| Policy update period | τ_P | 20 |
| Sensing radius | R_sense | 1.5 |
| Communication radius | R_comm | 2.5 |
| UCB exploration weight | β | 1.0 |
| Sensor noise std | — | 0.05 |
| Default α (`run_simulation`, `run_multiple`) | α | 0.2 |

Per-experiment settings, as set in each script:

| Experiment | Script | α | Swept values | Seeds |
|---|---|---|---|---|
| Main comparison | `simulation_main.py` | 0.2 (V1, V2); 0 (baseline) | — | 5 |
| Update period | `experiment_tau_sweep.py` | 0.2 and 0.4 (V2); 0 (baseline) | τ ∈ {10, 20, 50, 100} | 3 |
| Regret curves by τ | `experiment_tau_sweep.py` (`fig_tau_curves.png`) | 0.4 (V2) | τ ∈ {10, 20, 50, 100} | 3 |
| Team size | `experiment_scalability.py` | **0.4** (V2); 0 (baseline) | M ∈ {2, 3, 4, 6} | 3 |

## Notes on the report

- **α in the scalability study.** The report selects α = 0.2 for all primary
  comparisons. The team-size experiment (Fig. 7, `fig_scalability.png`) was
  run with **α = 0.4**, as its legend shows. The script keeps α = 0.4 so that
  it reproduces that figure.
- **Seed counts.** The report says all results are averaged over 5 seeds. That
  holds for the main comparison. The update-period and team-size scripts use
  3 seeds.
- **Fig. 7 caption vs. figure.** The caption says V2 "consistently achieves
  lower regret and faster last ROI discovery than the baseline." In
  `fig_scalability.png`, however, the baseline has lower final regret and
  faster last-ROI discovery than V2 at M = 6. The body text (Section IV-E)
  makes the same "consistently lower regret" claim. Its statement that V2
  discovers the last ROI faster at small team sizes does match the figure.

## Acknowledgment

Thanks to Maria G. Mendoza for guidance, feedback, and for making her
ergodic coverage framework available for this project.
