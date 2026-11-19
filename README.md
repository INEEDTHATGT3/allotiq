# AllotIQ — risk-aware inventory pre-buy for TBO.com

**VoyageHack 4.0 · Problem statement: Automate Inventory Optimization**

Team ELIXIR, HBTU Kanpur: Sambhav Jaiswal, Badal Khevadiya, Himanshu Raj, Nishant Poddar, Abhishek Yadav

> **Everything in this repository runs on synthetic data.** No TBO data was used. The numbers in our
> Round 1 deck come from this code and depend on the assumptions listed below. They show that the
> method works as designed. They do not predict TBO's results.

## The idea

TBO buys hotel inventory in advance (pre-buy) to guarantee availability and earn better margins.
Every pre-bought room is a bet. AllotIQ decides **what to buy, how many, on which contract, by
when and at what price**, inside a budget ceiling, so that profit stays high even in bad outcomes.

It rests on three observations:

1. **Sell-outs hide demand.** When a hotel sells out, bookings record what was available, not
   how many people wanted it. Planning from last season's bookings under-buys the hottest hotels.
   Searches keep counting after the rooms run out, so the search-to-booking ratio (look-to-book)
   can recover the hidden demand.
2. **Pre-buys are correlated bets.** Events, advisories and weather move a whole destination at
   once, so risk is managed for the portfolio, not hotel by hotel.
3. **Contracts differ.** Hard blocks (deepest rate, no returns), soft allotments (can be
   released before cut-off) and live supplier APIs (no risk, thin margin, sell out at peak)
   suit different levels of demand certainty. The best plan mixes them.

## What is in this repo

| File | What it does |
|---|---|
| `allotiq_sim.py` | One synthetic market: builds the market, three seasons of history, two forecasts, four buying policies, and scores them on 6,000 fresh demand scenarios |
| `run_all.py` | Runs markets 1 to 8 and writes `results/summary.md`, the source of every simulated number in the deck |
| `results/` | Output of `run_all.py`: one JSON per market plus the summary |

## Results (median of 8 synthetic markets)

AllotIQ compared with "buy like last season", under the same budget ceiling:

| Metric | Median | Range across markets |
|---|---:|---:|
| Profit in the worst 5% of outcomes | **+23.9%** | +3.4% to +28.8% |
| Unsold hard-block room-nights | **−48.4%** | −70.1% to −15.9% |
| Share of total demand served | **+3.7 points** | +1.2 to +4.0 |
| Expected profit | +2.2% | −0.1% to +9.0% |
| Budget left unspent by AllotIQ | 4.2% | 0% to 16.7% |

The main gain is **lower downside risk and higher availability**, not a big jump in average
profit. In one market (6) expected profit was flat, because the demand forecast overshot. Full
per-market table: [`results/summary.md`](results/summary.md).

## How the simulation works

1. **Market.** 8 destinations × 6 hotels × 8 stay-weeks = 384 inventory units ("SKUs"; one unit
   is a room-night). Each SKU has a mean demand, a seasonal pattern and a live nightly rate.
2. **History with sell-outs.** Three past seasons are generated. Demand grows each year while
   allotments stay flat, so many SKUs sell out and their bookings are capped. Searches are not
   capped.
3. **Two forecasts.**
   - *Last season:* last season's bookings, as they are.
   - *AllotIQ:* recover sold-out demand from searches ÷ look-to-book ratio; estimate each
     destination's growth trend from undisrupted seasons, shrunk toward the network median;
     project to next season, weighting recent seasons more.
4. **Four buying policies**, all under the same budget ceiling:
   - P0: no pre-buy, live APIs only
   - P1: hard blocks in proportion to last season's bookings (the "buy like last season" rule)
   - P2: P1's contract type, sized with the newsvendor rule on the AllotIQ forecast
   - P3 **AllotIQ:** for every SKU, choose hard-block and soft-allotment quantities from a grid.
     The budget is enforced through its shadow price (Lagrangian decomposition, solved by
     bisection). A risk penalty is set by maximising expected profit + worst-5% average on
     planning scenarios only.
5. **Scoring.** Plans are made on 2,000 scenarios drawn from the *forecast*, then judged on
   6,000 scenarios drawn from the *true* demand, with a different random seed. The planner never
   sees the truth.

## Every assumption (all chosen by us)

| Parameter | Value |
|---|---|
| Mean demand per hotel-week | 25–110 room-nights, × seasonal factor 1 ± 0.45 |
| Live (dynamic) rate | INR 4,000–20,000 per night; also the selling price |
| Hard block cost | 72% of the live rate, non-refundable |
| Unsold hard-block room | resold for 35% of its cost |
| Soft allotment cost | 85% of the live rate; 30% deposit counts against the budget |
| Released (unsold) soft room | penalty of 10% of its cost |
| Live API margin | 9% of the rate |
| Live API can serve excess demand | 45% of it in peak weeks, 90% otherwise |
| Demand noise | Poisson-lognormal: global σ 0.10, destination σ 0.12–0.35, hotel-week σ 0.30 |
| Disruption risk | 5% chance per destination per season; demand × 0.45 |
| Look-to-book ratio | 18–35 searches per booking, by destination |
| Demand growth per year | −5% to +35%, by destination |
| Past allotments | flat at about three-seasons-ago demand × 0.8–1.3 (this is what causes sell-outs) |
| Budget ceiling | enough to hard-block 75% of expected demand: INR 15.5–18.3 crore per market |

A consequence of these assumptions: on sold-out SKUs, last season's bookings understate true
demand by a median of 32%. That figure follows from the flat-allotment assumption above; it is
not a measurement of TBO data.

## Prototype vs the full design

| | This prototype | Planned for TBO |
|---|---|---|
| Forecast | look-to-book + trend + disruption flags | LightGBM quantile models on bookings, searches, events, flight searches, FX, advisories |
| Optimiser | Lagrangian over a quantity grid | MILP (OR-Tools) with a CVaR constraint and supplier/destination caps |
| Not modelled yet | cancellations, price response, multi-week release timing | all three |
| Interface | console output | ranked buy list with reason codes, risk dashboard, release alerts, API |

## Run it

```bash
pip install -r requirements.txt
python allotiq_sim.py            # market 7
SEED=3 python allotiq_sim.py     # another market
python run_all.py                # markets 1-8 and results/summary.md (about 30 seconds)
```

## Limitations

- Synthetic data with a demand structure the model partly knows; real data will be noisier.
- Contract terms (discounts, salvage, penalties) are assumptions; change them at the top of
  `allotiq_sim.py` to test other terms.
- One season ahead, one decision point; no in-season re-optimisation yet.
