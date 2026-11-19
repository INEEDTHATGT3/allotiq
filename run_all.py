"""
Run AllotIQ's simulation on synthetic markets 1-8 and summarise AllotIQ (P3) against the
"buy like last season" rule (P1). This produces every simulated number quoted in the
VoyageHack 4.0 Round 1 deck.

    python run_all.py

Writes results/market_<n>.json (full output per market), results/summary.json and
results/summary.md.
"""
import json
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SEEDS = range(1, 9)
P1, P3 = "P1 Last-year rule", "P3 AllotIQ (portfolio + risk)"

os.makedirs(RESULTS, exist_ok=True)
rows = []
for s in SEEDS:
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, SEED=str(s), OUT_DIR=tmp, MPLBACKEND="Agg")
        subprocess.run([sys.executable, os.path.join(HERE, "allotiq_sim.py")], env=env,
                       check=True, stdout=subprocess.DEVNULL)
        out = json.load(open(os.path.join(tmp, "allotiq_results.json")))
    json.dump(out, open(os.path.join(RESULTS, f"market_{s}.json"), "w"), indent=2)
    a, b, fc = out["results"][P1], out["results"][P3], out["forecast"]
    rows.append({
        "market": s,
        "budget_inr_cr": out["budget"] / 1e7,
        "worst5_profit_change_pct": 100 * (b["cvar5"] / a["cvar5"] - 1),
        "unsold_hard_change_pct": 100 * (b["unsold_hard"] / a["unsold_hard"] - 1),
        "demand_served_change_pts": 100 * (b["availability"] - a["availability"]),
        "expected_profit_change_pct": 100 * (b["exp_profit"] / a["exp_profit"] - 1),
        "budget_unspent_pct": 100 * (1 - b["capital"] / out["budget"]),
        "lastyear_bias_on_soldout_pct": 100 * fc["bias_naive_soldout"],
    })
    print(f"market {s} done")

labels = {
    "worst5_profit_change_pct": "Profit in the worst 5% of outcomes (%)",
    "unsold_hard_change_pct": "Unsold hard-block room-nights (%)",
    "demand_served_change_pts": "Share of demand served (points)",
    "expected_profit_change_pct": "Expected profit (%)",
    "budget_unspent_pct": "Budget left unspent by AllotIQ (%)",
    "lastyear_bias_on_soldout_pct": "Last season's bookings vs true demand, sold-out SKUs (%)",
    "budget_inr_cr": "Budget ceiling (INR crore)",
}
summary = {}
for k in labels:
    v = [r[k] for r in rows]
    summary[k] = {"median": statistics.median(v), "min": min(v), "max": max(v)}

json.dump({"per_market": rows, "summary": summary}, open(os.path.join(RESULTS, "summary.json"), "w"),
          indent=2)

lines = ["# AllotIQ vs \"buy like last season\" — 8 synthetic markets", "",
         "Change from the last-season rule (P1) to AllotIQ (P3), same budget ceiling.", "",
         "| Metric | Median | Min | Max |", "|---|---:|---:|---:|"]
for k, lab in labels.items():
    m = summary[k]
    lines.append(f"| {lab} | {m['median']:+.1f} | {m['min']:+.1f} | {m['max']:+.1f} |"
                 if k != "budget_inr_cr" else
                 f"| {lab} | {m['median']:.1f} | {m['min']:.1f} | {m['max']:.1f} |")
lines += ["", "## Per market", "",
          "| Market | Worst-5% profit | Unsold hard-block | Demand served | Expected profit | Budget unspent |",
          "|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    lines.append(f"| {r['market']} | {r['worst5_profit_change_pct']:+.1f}% | "
                 f"{r['unsold_hard_change_pct']:+.1f}% | {r['demand_served_change_pts']:+.1f} pts | "
                 f"{r['expected_profit_change_pct']:+.1f}% | {r['budget_unspent_pct']:.1f}% |")
lines += ["", "All data is synthetic. See README.md for every assumption."]
open(os.path.join(RESULTS, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
