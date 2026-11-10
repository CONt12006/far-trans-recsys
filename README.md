<p align="center">
  <a href="README.md"><img alt="English" src="https://img.shields.io/badge/English-1f6feb?style=for-the-badge"></a>
  <a href="README.ru.md"><img alt="Русский" src="https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-555555?style=for-the-badge"></a>
</p>

# FAR-Trans: investment product recommendations

For a customer at date `t`, we rank the available assets (stocks, bonds, funds) so that the ones they will buy
in the next 60 days come first, without violating their risk profile. Quality is measured with three groups
of metrics: behavioural accuracy, financial outcome, and suitability.

Data: [FAR-Trans](https://doi.org/10.5525/gla.researchdata.1658) (Sanz-Cruzado, Droukas, McCreadie, 2024),
real customer trades of a European bank, 2018–2022, CC BY 4.0. Short business report (in Russian): [docs/report.md](docs/report.md).

## Highlights

* **Two-stage model** (CF candidates → LightGBM lambdarank) beats every baseline on each of the 6 test
  cutoffs: NDCG@10 **0.604** vs 0.580, on assets new to the customer **0.287** vs 0.269.
* **Under the dataset authors' protocol** Item-kNN reaches nDCG@10 0.353 vs 0.340 for the paper's best method (LightGCN);
  the ranker on a subset of dates gets 0.345 vs 0.335 for kNN and 0.280 for popularity.
* **Suitability is expensive:** customers mostly buy assets that do not fit their profile, so fully banning
  unsuitable recommendations cuts NDCG by ~77%. A trade-off curve shows the cost of each policy.
* **"Recommend what will go up" does not work:** popularity does not predict returns, and momentum
  reversed in 2022. An honest negative result.
* **Found and fixed:** backfilled customer profiles (leakage), a portfolio snapshot disguised as purchases,
  suspended stocks that looked like the safest assets.

## Structure

```
configs/base.yaml           cutoff dates, windows, risk scale and policies
src/farrec/data.py          loading, cleaning, as-of selections
src/farrec/portfolio.py     portfolio reconstruction from trades
src/farrec/splits.py        time cutoffs, target, train/valid/test
src/farrec/risk.py          asset risk levels 1..7 (by volatility, SRRI-style)
src/farrec/metrics.py       ranking / financial / suitability
src/farrec/baselines.py     simple baselines
src/farrec/cf.py            Item-kNN and ALS
src/farrec/candidates.py    stage 1: candidate generation
src/farrec/features.py      customer, asset and pair features
src/farrec/ranker.py        stage 2: LightGBM / CatBoost ranker
src/farrec/postprocess.py   re-ranking by risk and return
src/farrec/evaluate.py      unified evaluation protocol
src/farrec/paper_protocol.py  evaluation protocol from the dataset paper
scripts/                    experiment runners (see reproduce.sh)
service/app.py              FastAPI: top-k recommendations for a customer at a date
tests/                      leakage tests and unit tests
```

## Running

```
pip install -r requirements.txt
# unpack the dataset archive into data/raw/FAR-Trans/
pytest -q
bash scripts/reproduce.sh          # all tables and figures, ~1 h
uvicorn service.app:app --port 8000
```

Example request:
`GET /recommend?customer_id=DD1E7FA6C74787F13EC7&date=2022-09-15&k=5&policy=strict` returns the top 5 with each
asset's risk level, a "bought before" flag, the recommendation source, and the number of candidates dropped by the risk filter.
The first request for a month trains the model (~2 min); subsequent ones take ~9 s.

## Experimental setup

* Cutoffs: the first day of each month. Features and candidates come only from `(-inf, t)`; target: purchases in `[t, t+60d)`.
* For an evaluation cutoff `T` the model is trained only on the past: valid = the last cutoff whose target window ends
  before `T` (early stopping), train = 12 cutoffs whose windows end before valid. Without this gap, train labels would
  overlap the evaluation period.
* Only customers who bought something in the target window are evaluated. Baselines are evaluated on 12 cutoffs
  (2021-10 … 2022-09), the ranker on 6 (2022-04 … 2022-09) with retraining at each one.
* `_new` metrics: everything the customer bought before is hidden; the target is only first purchases of an asset. This matters
  because **78.5% of purchases are repeats**.

## Data quirks

| Finding | Handling |
|---|---|
| 2018-01-02: ~35.7k "purchases" in one day (normally 50–300): an export of initial portfolios | `isSnapshot` flag: they go into portfolios but are not counted as purchases |
| `transactionID` is not unique (~29k duplicates): joint accounts | deduplicate only on the full row |
| Risk profile is sometimes `Predicted_*` | prefix stripped, separate flag |
| Customer profile never changes over time and is backfilled | see "Leakage" |
| Suspended stocks with a frozen price: volatility 0 → risk level 1, "safer than bonds" (FOLLIE FOLLIE, AEGEK, DIONIC…) | a stock with >50% of days without price change gets level 7 |
| Sales exceed visible purchases (history before 2018 is not visible) | positions are clipped at zero |
| 89% of purchases are stocks | taken into account when interpreting results |

## Baselines

Mean over 12 cutoffs, std in brackets. kNN and ALS parameters are tuned only on cutoffs before the test (`scripts/tune_cf.py`).

| method | NDCG@10 | HR@10 | NDCG@10_new | HR@10_new | Ret@5 (60d) | Unsuit@5 |
|---|---|---|---|---|---|---|
| random | 0.002 | 0.01 | 0.003 | 0.01 | −0.4% | 0.38 |
| pop_30d | 0.308 (0.070) | 0.64 | 0.243 | 0.48 | −1.0% (8.7%) | 0.62 |
| pop_segment_90d | 0.261 | 0.61 | 0.188 | 0.43 | −1.9% | 0.56 |
| momentum_90d | 0.017 | 0.06 | 0.016 | 0.05 | −5.6% (19%) | 0.72 |
| repeat_then_pop | **0.546** (0.045) | **0.81** | 0.239 | 0.47 | −1.3% | 0.66 |
| item_knn | 0.539 (0.050) | 0.81 | **0.281** (0.037) | **0.53** | −1.3% | 0.68 |
| als | 0.540 (0.041) | 0.80 | 0.202 | 0.38 | −1.4% | 0.67 |

* On overall NDCG, "own purchases + popularity" is as good as kNN and ALS: the difference between models only shows on new assets.
* **A single test cutoff lies.** On the 2022-09 cutoff popularity yields +8.1% over 60 days, while the mean over 12 cutoffs is −1.0%
  with std 8.7%. Financial metrics only make sense across many cutoffs.
* No baseline beats the market. Momentum is the worst: it picks illiquid securities with sharp price jumps.

## Two-stage model

**Stage 1, candidates** ([candidates.py](src/farrec/candidates.py)): union of the top 50 from four sources
(own past purchases, Item-kNN, ALS, popularity), ~85 assets per customer on average.

| top-N per source | candidates per customer | recall (all purchases) | recall (new assets) |
|---|---|---|---|
| 20 | 35 | 88–89% | 54–67% |
| **50** | **85** | **97%** | **84–89%** |
| 100 | 155 | 99% | 95–97% |

**Stage 2, ranking** ([ranker.py](src/farrec/ranker.py)): LightGBM lambdarank, group = (cutoff, customer),
68 features ([features.py](src/farrec/features.py)): customer (activity, portfolio, profile), asset (returns,
volatility, drawdown, popularity and its trend), pair (currently held, number of past purchases, sector share in the portfolio,
risk excess, popularity within the risk segment, source scores and ranks).

Mean over 6 cutoffs (2022-04 … 2022-09); the model beats the baselines **on every** cutoff:

| method | NDCG@10 | HR@10 | NDCG@10_new | HR@10_new |
|---|---|---|---|---|
| pop_30d | 0.335 | 0.66 | 0.239 | 0.45 |
| repeat_then_pop | 0.580 | 0.83 | 0.235 | 0.44 |
| item_knn | 0.574 | 0.83 | 0.269 | 0.49 |
| als | 0.570 | 0.82 | 0.188 | 0.34 |
| **ranker_lgbm** | **0.604** | **0.85** | **0.287** | **0.52** |

**CatBoost YetiRank** on the last 3 cutoffs: NDCG@10 0.590 vs 0.588 for LightGBM, on new assets 0.269 vs
0.271, i.e. within noise, while training about 30× longer (~1 h vs ~2 min per cutoff). The main model is LightGBM.

## Comparison with the dataset paper

The paper's protocol ([paper_protocol.py](src/farrec/paper_protocol.py)) differs from mine: 61 dates every two weeks
(2019-08 … 2022-01), a 6-month window, only assets new to the customer, ROI@10 = mean monthly return of an
equally weighted top 10. To make sure the protocol is reproduced, I ran random first: ROI@10 **0.0071**,
exactly as in the paper; nDCG@10 0.0098 vs 0.0106.

| method | nDCG@10 | ROI@10 | source |
|---|---|---|---|
| LightGCN (best in the paper) | 0.340 | 0.0004 | paper, 61 dates |
| Popularity | 0.271 | 0.0006 | paper, 61 dates |
| UB kNN | 0.160 | 0.0119 | paper, 61 dates |
| **Item-kNN** | **0.353** | −0.0086 | this project, same 61 dates |
| pop_30d | 0.304 | −0.0140 | this project, same 61 dates |

The ranker is retrained at every date, so I ran it on 18 monthly dates inside the paper's period (2020-06 … 2021-11):
**0.345** vs 0.335 for Item-kNN (better on 15 of 18 dates) and 0.280 for popularity. ROI@10 is 1.42% per month vs 1.25% for the market.
The gap over kNN is smaller here than in the main protocol: the model learns on a 60-day target with repeat purchases,
while the paper's protocol measures only new assets over 6 months.

## What the model learned (SHAP)

![shap](reports/figures/shap_summary.png)

* 47% of total |SHAP| comes from source scores and ranks (kNN, ALS, repeats), 28% from asset features, 12–13% each
  from customer and pair features.
* **The model learned "people buy what went up":** the contribution of the 90-day return grows with its value, plus a weak
  buy-the-dip effect.
* **The model barely cares about risk:** the mean contribution of risk-level excess is ~0.008 vs ~0.35 for the kNN rank.
  Customer behaviour does not teach the model to respect the profile, so constraints are applied separately.

![shap dependence](reports/figures/shap_dependence.png)

## Risk profile: the quality–suitability trade-off

Asset risk level 1..7 by annualised volatility (SRRI-style buckets: 0.5/2/5/10/15/25%). Allowed levels cannot
be calibrated from behaviour: **the profile barely affects purchases**.

| profile | share of level-7 purchases | share of stocks |
|---|---|---|
| Conservative | 70% | 75% |
| Income | 79% | 83% |
| Balanced | 82% | 87% |
| Aggressive | 82% | 87% |

The trades in the dataset are customer-initiated (execution-only), whereas a recommendation is advice subject to
suitability requirements. So limits are set by policy ([configs/base.yaml](configs/base.yaml)): `strict` (Conservative ≤3,
Income ≤4, Balanced ≤5, Aggressive ≤7) and `moderate` (+1 level for the first three profiles).

Re-ranking ([postprocess.py](src/farrec/postprocess.py)): final score = model score percentile − λ × level
excess. The hard filter drops unsuitable assets and fills the list with the most popular suitable ones.

![risk](reports/figures/risk_tradeoff.png)

| strict | NDCG@10 | NDCG@10_new | unsuitable share in top 5 |
|---|---|---|---|
| λ = 0 (no constraints) | 0.604 | 0.287 | 73% |
| λ = 0.1 | 0.574 | 0.280 | 69% |
| λ = 0.2 | 0.504 | 0.239 | 61% |
| λ = 0.5 | 0.200 | 0.127 | 17% |
| hard filter | 0.140 | 0.105 | 0% |

* **There is no free suitability:** every unsuitable recommendation removed is a lost hit.
  Full compliance with the strict policy costs ~77% of NDCG.
* The curve is non-linear: at λ ≤ 0.1 the loss is ~5%, then it drops off between 0.2 and 0.5.
* The moderate policy gives noticeably more quality at the same violation rate. The choice of policy is a business decision;
  the curve shows its price.

## Financial score: does "recommend what will go up" work?

Final score = α × percentile of P(purchase) + (1 − α) × percentile of a return signal. First I checked whether
the signals predict future returns (rank IC with the 60-day return, 28 cutoffs 2020–2022):

| signal | mean IC | share of months with IC > 0 |
|---|---|---|
| momentum 90d | +0.081 | 71% |
| momentum 365d | +0.041 | 61% |
| low volatility | −0.090 | 39% |
| 30-day popularity | +0.008 | 43% |

![return](reports/figures/return_tradeoff.png)

* **Popularity does not predict returns:** people buy what is in the news, and that is unrelated to future growth.
* **90-day momentum hurts returns in 2022:** at α = 0 recommendations lag the market by 6.4 pp vs 2.0 pp
  for the pure model. A signal that worked on average over 2020–2022 reversed.
* 365-day momentum gives +0.5 pp at α = 0.8, but NDCG falls from 0.60 to 0.43, and return noise across cutoffs is ±7 pp.
* **Conclusion:** on this data the financial score does not work. That is a result, not a failed experiment.

## Quality by segment

![segments](reports/figures/segments.png)

| segment | n | repeat_then_pop | item_knn | ranker | ranker, new | item_knn, new |
|---|---|---|---|---|---|---|
| Newcomers (≤2 purchases) | 268 | 0.334 | 0.348 | **0.390** | **0.247** | 0.196 |
| Active (≥10 in 90d) | 247 | 0.695 | 0.645 | **0.707** | 0.271 | **0.290** |
| Conservative and Income | 588 | 0.606 | 0.604 | **0.627** | **0.319** | 0.305 |
| All | 1753 | 0.580 | 0.574 | **0.604** | **0.289** | 0.269 |

* The ranker gains the most on newcomers: +12% over the best baseline, +26% on new assets.
* For active customers kNN beats the ranker on new assets: the model leans on repeat purchases.

## Leakage found and closed

* **Customer profiles are backfilled.** In 38.5% of rows `lastQuestionnaireDate` is later than the row's own date,
  and no customer's risk profile ever changes. The profile reflects the final state, not the state at the date.
  The "days since last questionnaire" feature produced negative values and was removed; the risk profile is used as a
  static attribute (a data limitation).
* **The 2018-01-02 portfolio snapshot disguised as purchases** would have inflated popularity and the target.
* Prices are forward-filled only, never backward-filled (otherwise future prices leak into the past).
* Candidates are limited to assets that had a price before the cutoff.
* `forward_return` uses future prices and is called only in metrics.
* Ranker training cutoffs are only those whose target window ended before the evaluation cutoff.
* Hyperparameter tuning only on cutoffs before the test.
* Test [test_no_leakage.py](tests/test_no_leakage.py): second-stage recommendations and features on full data and on data
  truncated at the cutoff date must match.

## Limitations

* The risk profile is static and probably reflects questionnaires filled after the cutoff date.
* The risk level is computed per security, not per portfolio: a real product needs an estimate of the contribution to portfolio risk.
* The target is the fact of purchase. We do not know what the customer was shown, so this measures "guessing", not the effect of recommendations.
* The 2022 test period is a falling market; financial conclusions are tied to the market regime.
