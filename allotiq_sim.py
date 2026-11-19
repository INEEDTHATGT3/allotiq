"""
AllotIQ — proof-of-concept simulation for TBO VoyageHack 4.0 (Inventory Optimization).

Synthetic world: 8 destinations x 6 hotels x 8 stay-weeks = 384 inventory SKUs
(hotel x stay-week, unit = room-night). Everything here is SYNTHETIC and illustrative.

Policies compared under the same capital budget:
  P0  No pre-buy           — sell only via dynamic supplier APIs (thin margin, can sell out)
  P1  Last-year rule       — hard blocks sized in proportion to last season's bookings
                             (bookings are censored by last season's sell-outs)
  P2  + Demand unconstrain — recover true demand of sold-out SKUs from search (look-to-book)
                             signal, then size hard blocks with the newsvendor rule
  P3  AllotIQ              — unconstrained forecast + portfolio optimiser that chooses
                             hard-block vs soft-allotment mix per SKU under the budget
                             (Lagrangian decomposition) with a downside-risk control

Run:  python allotiq_sim.py            (one synthetic market, seed 7)
      SEED=3 python allotiq_sim.py     (another market)
      python run_all.py                (markets 1-8 + the summary quoted in the deck)
"""
import json
import numpy as np

import os
SEED_WORLD = int(os.environ.get("SEED", 7))   # which synthetic market
SEED_PLAN, SEED_EVAL = 11, 23
OUT_DIR = os.environ.get("OUT_DIR", ".")
N_PLAN, N_EVAL = 2000, 6000

# ----------------------------------------------------------------------------------------
# 1. Synthetic world
# ----------------------------------------------------------------------------------------
rng = np.random.default_rng(SEED_WORLD)
DEST = ["Dubai", "Bangkok", "Singapore", "Bali", "Goa", "Maldives", "London", "Phuket"]
N_D, N_H, N_W = len(DEST), 6, 8
dest_sigma = rng.uniform(0.12, 0.35, N_D)          # destination-level demand shock volatility
GLOBAL_SIGMA, IDIO_SIGMA = 0.10, 0.30
EVENT_P, EVENT_HIT = 0.05, 0.45                    # 5% chance a destination is disrupted (-55%)
ltb_true = rng.uniform(18, 35, N_D)                # look-to-book ratio (searches per booking)
growth = rng.uniform(-0.05, 0.35, N_D)             # YoY demand growth per destination

rows = []
for d in range(N_D):
    for h in range(N_H):
        base = rng.uniform(25, 110)                # mean room-nights per stay-week
        rate = rng.uniform(4000, 20000)            # dynamic market rate, INR/night
        for w in range(N_W):
            season = 1.0 + 0.45 * np.sin((w + d) / N_W * 2 * np.pi)   # peak / lean weeks
            rows.append(dict(d=d, h=h, w=w, mu=base * season, rate=rate, peak=season > 1.2))
N = len(rows)
d_idx = np.array([r["d"] for r in rows])
mu_true = np.array([r["mu"] for r in rows])
rate = np.array([r["rate"] for r in rows])
peak = np.array([r["peak"] for r in rows])

# Commercial terms (fractions of the dynamic rate)
P_SELL = rate                       # sell at market rate
C_HARD = 0.72 * rate                # non-refundable hard block: deepest discount
C_SOFT = 0.85 * rate                # soft allotment: release-back allowed before cut-off
SALVAGE = 0.35 * C_HARD             # distressed resale of unsold hard-block rooms
REL_PEN = 0.10 * C_SOFT             # penalty per released (unsold) soft-allotment room
DEPOSIT = 0.30                      # soft allotment locks 30% of cost as deposit (budget)
M_DYN = 0.09 * rate                 # margin on dynamic API sales
AVAIL = np.where(peak, 0.45, 0.90)  # share of excess demand dynamic supply can still serve


def demand_scenarios(mu, n, seed, return_events=False):
    """Poisson-lognormal demand with global + destination + idiosyncratic shocks (correlated)
    plus a rare destination-level disruption (advisory, strike, weather...)."""
    r = np.random.default_rng(seed)
    g = r.normal(0, GLOBAL_SIGMA, (n, 1))
    z = r.normal(0, 1, (n, N_D)) * dest_sigma
    e = r.normal(0, IDIO_SIGMA, (n, N))
    var = GLOBAL_SIGMA**2 + dest_sigma[d_idx] ** 2 + IDIO_SIGMA**2
    hit = r.random((n, N_D)) < EVENT_P
    event = np.where(hit, EVENT_HIT, 1.0)
    D = r.poisson(mu * np.exp(g + z[:, d_idx] + e - var / 2) * event[:, d_idx])
    return (D, hit) if return_events else D


# ----------------------------------------------------------------------------------------
# 2. Three past seasons: bookings are CENSORED by sell-outs (allotments stayed flat while
#    demand grew); searches are not censored
# ----------------------------------------------------------------------------------------
N_HIST = 3
cap = np.round(mu_true / (1 + growth[d_idx]) ** N_HIST * rng.uniform(0.8, 1.3, N))  # flat allotment
bookings, searches, soldout, disrupted = [], [], [], []
for k in range(N_HIST):                                   # k=0 oldest ... k=2 last season
    mu_k = mu_true / (1 + growth[d_idx]) ** (N_HIST - k)
    D_k, hit_k = demand_scenarios(mu_k, 1, SEED_WORLD + 100 + k, return_events=True)
    D_k = D_k[0]
    bookings.append(np.minimum(D_k, cap))
    soldout.append(D_k >= cap)
    searches.append(rng.poisson(D_k * ltb_true[d_idx]))
    disrupted.append(hit_k[0])                            # past disruptions are known events
bookings, searches, soldout, disrupted = map(np.array, (bookings, searches, soldout, disrupted))
sold_out = soldout[-1]                                     # sold out last season

# Forecast A (status quo): "same as last season" — last season's bookings
mu_naive = np.maximum(bookings[-1], 1.0)

# Forecast B (unconstrained):
#  (1) look-to-book ratio from SKU-seasons that did NOT sell out
#  (2) search-implied demand where bookings were capped by a sell-out (demand unconstraining)
#  (3) destination growth trend from undisrupted seasons, shrunk toward the network median
#  (4) project each undisrupted season to next season; weight recent seasons more
ltb_hat = np.array([searches[:, d_idx == d][~soldout[:, d_idx == d]].sum() /
                    max(bookings[:, d_idx == d][~soldout[:, d_idx == d]].sum(), 1)
                    for d in range(N_D)])
dem_hat = np.where(soldout, np.maximum(searches / ltb_hat[d_idx], bookings), bookings)
g_raw = np.zeros(N_D)
for d in range(N_D):
    ok = ~disrupted[:, d]
    tot = dem_hat[:, d_idx == d].sum(1)
    t = np.arange(N_HIST)[ok]
    g_raw[d] = np.exp(np.polyfit(t, np.log(tot[ok]), 1)[0]) - 1 if ok.sum() >= 2 else np.nan
g_med = np.nanmedian(g_raw)
g_hat = np.clip(0.5 * np.nan_to_num(g_raw, nan=g_med) + 0.5 * g_med, -0.10, 0.40)
proj = dem_hat * (1 + g_hat[d_idx]) ** (N_HIST - np.arange(N_HIST))[:, None]
wts = np.array([0.2, 0.3, 0.5])[:, None] * ~disrupted[:, d_idx]
mu_unc = np.maximum((proj * wts).sum(0) / np.maximum(wts.sum(0), 1e-9), 1.0)

# ----------------------------------------------------------------------------------------
# 3. P&L of a pre-buy plan under demand scenarios
# ----------------------------------------------------------------------------------------
def pnl(qh, qs, D):
    """Total profit per scenario (pre-bought + dynamic). qh/qs: (N,), D: (S, N)."""
    hard_sold = np.minimum(D, qh)
    soft_sold = np.minimum(np.maximum(D - qh, 0), qs)
    excess = np.maximum(D - qh - qs, 0)
    prebuy = (hard_sold * (P_SELL - C_HARD) - (qh - hard_sold) * (C_HARD - SALVAGE)
              + soft_sold * (P_SELL - C_SOFT) - (qs - soft_sold) * REL_PEN)
    dyn = AVAIL * excess * M_DYN
    served = hard_sold + soft_sold + AVAIL * excess
    return prebuy, dyn, served


def budget_use(qh, qs):
    return (qh * C_HARD + qs * DEPOSIT * C_SOFT).sum()


def evaluate(qh, qs, D):
    pre, dyn, served = pnl(qh, qs, D)
    total = (pre + dyn).sum(1)
    pre_book = pre.sum(1)
    k = int(0.05 * len(total))
    worst = np.sort(total)[:k]
    return dict(
        exp_profit=float(total.mean()),
        cvar5=float(worst.mean()),
        p_prebuy_loss=float((pre_book < 0).mean()),
        availability=float(served.sum(1).mean() / D.sum(1).mean()),
        unsold_hard=float(np.maximum(qh - D, 0).sum(1).mean()),
        capital=float(budget_use(qh, qs)),
    )


# Max budget TBO provides: enough to hard-block ~75% of expected demand (a ceiling, not a target)
BUDGET = float(0.75 * (mu_true * C_HARD).sum())

D_plan = demand_scenarios(mu_unc, N_PLAN, SEED_PLAN)      # planning uses the FORECAST
D_eval = demand_scenarios(mu_true, N_EVAL, SEED_EVAL)     # judging uses the TRUTH

# ----------------------------------------------------------------------------------------
# 4. Policies
# ----------------------------------------------------------------------------------------
zero = np.zeros(N)


def scale_to_budget(qh, qs=None):
    qs = zero if qs is None else qs
    used = budget_use(qh, qs)
    f = min(1.0, BUDGET / used) if used > 0 else 0
    return np.floor(qh * f), np.floor(qs * f)


# P1: last-year rule — hard blocks proportional to last season's bookings, spend full budget
w = mu_naive * C_HARD
qh1 = np.floor(BUDGET * (mu_naive / w.sum()))
# P2: newsvendor on unconstrained forecast, hard blocks only
cu = P_SELL - C_HARD - AVAIL * M_DYN          # gain of a sold hard room vs serving dynamically
co = C_HARD - SALVAGE                         # loss of an unsold hard room
crit = cu / (cu + co)
qh2 = np.array([np.quantile(D_plan[:, i], crit[i]) for i in range(N)])
qh2, _ = scale_to_budget(qh2)

# P3: AllotIQ — per-SKU option grid over (hard, soft), Lagrangian on budget, downside penalty
GRID = np.linspace(0, 1.6, 9)                 # quantities as multiples of forecast mean


def allotiq(kappa):
    """kappa = weight on expected pre-buy downside (risk aversion)."""
    opts_val, opts_cost, opts_q = [], [], []
    for i in range(N):
        qa = np.round(GRID * mu_unc[i])
        QH, QS = np.meshgrid(qa, qa, indexing="ij")
        QH, QS = QH.ravel(), QS.ravel()
        keep = QH + QS <= 1.6 * mu_unc[i] + 1
        QH, QS = QH[keep], QS[keep]
        Di = D_plan[:, i][:, None]
        hs = np.minimum(Di, QH)
        ss = np.minimum(np.maximum(Di - QH, 0), QS)
        ex = np.maximum(Di - QH - QS, 0)
        pre = (hs * (P_SELL[i] - C_HARD[i]) - (QH - hs) * (C_HARD[i] - SALVAGE[i])
               + ss * (P_SELL[i] - C_SOFT[i]) - (QS - ss) * REL_PEN[i])
        tot = pre + AVAIL[i] * ex * M_DYN[i]
        val = tot.mean(0) - kappa * np.maximum(-pre, 0).mean(0)
        opts_val.append(val)
        opts_cost.append(QH * C_HARD[i] + QS * DEPOSIT * C_SOFT[i])
        opts_q.append((QH, QS))

    def choose(lam):
        qh, qs, cost = np.zeros(N), np.zeros(N), 0.0
        for i in range(N):
            j = np.argmax(opts_val[i] - lam * opts_cost[i])
            qh[i], qs[i] = opts_q[i][0][j], opts_q[i][1][j]
            cost += opts_cost[i][j]
        return qh, qs, cost

    lo, hi = 0.0, 2.0
    if choose(0.0)[2] <= BUDGET:
        return choose(0.0)[:2]
    for _ in range(40):                        # bisection on the budget shadow price
        mid = (lo + hi) / 2
        if choose(mid)[2] > BUDGET:
            lo = mid
        else:
            hi = mid
    return choose(hi)[:2]


policies = {
    "P0 No pre-buy (dynamic only)": (zero, zero),
    "P1 Last-year rule": (qh1, zero),
    "P2 + Demand unconstraining": (qh2, zero),
}
# Risk dial: pick risk aversion that maximises a risk-adjusted score on PLANNING scenarios
# (expected profit + worst-5% average, equal weight). Never peeks at the evaluation truth.
best, dial = None, []
for kappa in [0.0, 1.0, 2.0, 4.0]:
    qh, qs = allotiq(kappa)
    plan = evaluate(qh, qs, D_plan)
    score = plan["exp_profit"] + plan["cvar5"]
    dial.append(dict(kappa=kappa, plan_exp=plan["exp_profit"], plan_cvar5=plan["cvar5"],
                     capital=plan["capital"]))
    if best is None or score > best[0]:
        best = (score, kappa, qh, qs)
policies["P3 AllotIQ (portfolio + risk)"] = (best[2], best[3])

# ----------------------------------------------------------------------------------------
# 5. Evaluate on true demand and report
# ----------------------------------------------------------------------------------------
res = {k: evaluate(qh, qs, D_eval) for k, (qh, qs) in policies.items()}
base = res["P0 No pre-buy (dynamic only)"]["exp_profit"]
p1 = res["P1 Last-year rule"]["exp_profit"]
cr = 1e7
print(f"SKUs={N}  sold-out last season={sold_out.sum()}  budget=INR {BUDGET/cr:.2f} cr  "
      f"kappa*={best[1]}")
print(f"{'Policy':34s} {'E[profit] cr':>12s} {'vs P1':>7s} {'CVaR5% cr':>10s} {'vs P1':>7s} "
      f"{'avail':>6s} {'unsold hard RN':>14s} {'capital cr':>10s}")
c1 = res["P1 Last-year rule"]["cvar5"]
for k, r in res.items():
    print(f"{k:34s} {r['exp_profit']/cr:12.3f} {100*(r['exp_profit']/p1-1):+6.1f}% "
          f"{r['cvar5']/cr:10.3f} {100*(r['cvar5']/c1-1):+6.1f}% "
          f"{100*r['availability']:5.1f}% {r['unsold_hard']:14.0f} {r['capital']/cr:10.2f}")
print("Risk dial (planning view):", [(x["kappa"], round(x["plan_exp"]/cr, 3),
                                     round(x["plan_cvar5"]/cr, 3)) for x in dial])

qh3, qs3 = policies["P3 AllotIQ (portfolio + risk)"]
mix = dict(hard_room_nights=float(qh3.sum()), soft_room_nights=float(qs3.sum()),
           skus_with_prebuy=int(((qh3 + qs3) > 0).sum()))
print("AllotIQ mix:", mix)
fc_err = dict(
    mape_naive_all=float(np.mean(np.abs(mu_naive - mu_true) / mu_true)),
    mape_unc_all=float(np.mean(np.abs(mu_unc - mu_true) / mu_true)),
    mape_naive_soldout=float(np.mean(np.abs(mu_naive - mu_true)[sold_out] / mu_true[sold_out])),
    mape_unc_soldout=float(np.mean(np.abs(mu_unc - mu_true)[sold_out] / mu_true[sold_out])),
    bias_naive_soldout=float(np.mean((mu_naive - mu_true)[sold_out] / mu_true[sold_out])),
    bias_unc_soldout=float(np.mean((mu_unc - mu_true)[sold_out] / mu_true[sold_out])),
)
print("Forecast error (MAPE / bias):", {k: round(v, 3) for k, v in fc_err.items()})

json.dump(dict(results=res, budget=BUDGET, kappa=best[1], mix=mix, forecast=fc_err,
               n_skus=N, n_soldout=int(sold_out.sum())),
          open(os.path.join(OUT_DIR, "allotiq_results.json"), "w"), indent=2)

# Chart: expected profit vs worst-5% outcome
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
labels = ["No pre-buy", "Last-year rule", "+ Unconstrained\nforecast", "AllotIQ"]
vals = [r["exp_profit"] / cr for r in res.values()]
cvar = [r["cvar5"] / cr for r in res.values()]
x = np.arange(4)
fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
ax.bar(x - 0.18, vals, 0.36, label="Expected profit", color=["#c9ced6", "#9aa4b2", "#5b8def", "#0b5fff"])
ax.bar(x + 0.18, cvar, 0.36, label="Worst-5% average", color="#f2a65a")
ax.set_xticks(x, labels, fontsize=8)
ax.set_ylabel("INR crore", fontsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.legend(fontsize=7, frameon=False)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "allotiq_results.png"), transparent=True)
print("saved chart")
