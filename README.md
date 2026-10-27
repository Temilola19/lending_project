# Loan Approval Strategy: Growth vs. Risk

An end-to-end consumer-lending analytics project: data cleaning, segmentation, risk modeling, policy simulation, and a business recommendation.

**Read first:** [`MEMO.md`](MEMO.md) (one-page recommendation). Code is in `analysis.py`; outputs are in `tables/` and `charts/`.

## Business question
Where can a lender approve more customers without a meaningful rise in defaults, and which customer segments deserve a different strategy?

## Data
German Credit dataset (UCI Machine Learning Repository), 1,000 loans, 20 applicant/loan attributes, outcome = good/bad. Loaded from the `scorecardpy` package. No missing values or duplicates. 30% of loans are "bad".

## Method
1. **Clean and profile** — type fixes, bucket duration and amount, default rate by segment, chi-square tests of segment vs. default.
2. **Segment economics** — default rate, share of defaults, and expected profit per loan by segment.
3. **Model** — logistic regression vs. gradient boosting (5-fold CV AUC). Sensitive attributes (age, sex/marital status, foreign-worker) excluded from the decision model. Every loan is scored out-of-fold so no loan is scored by a model that trained on it.
4. **Policy simulation** — approve if predicted default probability < threshold; compare approval rate, default rate, and expected profit across thresholds; compare to a simple three-rule policy.
5. **Validation** — split-sample threshold check (200 repeats) and a sensitivity grid on profit/loss assumptions.
6. **Pilot design and fairness spot check** — sample sizes for an A/B pilot; approval rate by age band and foreign-worker status.

## Key results
- Default rate 30%; approve-all is unprofitable under stated assumptions (–$173 per loan).
- Cutoff policy: approve 54%, default rate 13%, profit +$143K. Moderate version: approve 61%, default 15%, profit +$131K.
- Logistic regression AUC 0.78; adding sensitive attributes adds ~0.002.
- Simple three-rule policy captures ~80% of the model's gain.

## Assumptions (not in the data)
Good loan earns 20% of principal over its life; bad loan loses 50% of principal net of recovery. Dollar values are illustrative; the sensitivity grid in `tables/sensitivity.csv` shows other values.

## Limitations
Small, older, non-U.S. dataset; only approved loans observed; correlational findings; economics are assumed. See the memo.

## Run it
```
pip install -r requirements.txt
python analysis.py
```

## Dashboard
Tableau/Power BI were not used in this repo. To build the dashboard, import `tables/all_segments.csv`, `tables/policy_curve.csv`, and `tables/scored_loans.csv`. Suggested pages: (1) default rate and profit by segment, (2) approval-rate vs. default-rate vs. profit slider, (3) fairness spot check.

## Resume bullets (use only after you can explain every number yourself)
- Analyzed 1,000 consumer loans to identify segments driving defaults; found customers with negative checking balances were 27% of loans but 45% of defaults.
- Built an explainable logistic regression (AUC 0.78) excluding sensitive attributes with no meaningful accuracy loss; simulated approval-cutoff policies and recommended a tiered tighten/hold/grow strategy, validated with a split-sample test and sensitivity analysis.
- Designed an A/B pilot (sample-size calculation) and a fair-lending guardrail; presented findings in a one-page business memo.
