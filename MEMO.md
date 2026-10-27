# Memo: Loan Approval Strategy — Where to Tighten, Hold, and Grow

**To:** Head of Lending Analytics  **From:** Temilola Afolabi  **Data:** German Credit (UCI), 1,000 loans, 30% defaulted
**Status:** Portfolio-style analysis on a small public dataset. Dollar figures rely on stated assumptions and are illustrative.

## The question
Where can we approve more customers without a meaningful rise in defaults, and where should we change strategy?

## Bottom line
Approving every applicant does not work in this portfolio: **30% of loans defaulted, and under my assumptions that turns a –$173K result (–$173 per loan) into a loss.** A risk-based cutoff that approves about **54%** of applicants lowers the default rate of approved loans to **13%** and moves expected profit to **+$143K**. Because the profit curve is flat between roughly 45% and 60% approval, a more moderate cut at **~61% approval (15% default, +$131K)** is nearly as profitable while turning away fewer customers. The question for the business is where on that curve to sit.

## What the data shows
1. **Risk is concentrated.** Customers with a negative checking balance are 27% of loans but **45% of all defaults** (49% default rate vs. 30% overall). Customers with under 100 DM in savings are 60% of loans and 72% of defaults.
2. **Long, large loans lose money.** Terms over 36 months default at 52% and lose about $1,190 per loan; the largest-quartile loans default at 42%.
3. **There are safe pockets to grow.** Terms of 12 months or less default at 21% and are profitable. Customers with no checking account at the bank (39% of loans) default at only 12% and earn about $276 per loan. Used-car loans (17% default) beat new-car loans (38%).
4. **A simple, explainable model works.** Logistic regression reached AUC 0.78 (5-fold CV, ±0.02), as good as gradient boosting (0.77). Adding age, sex/marital status, and foreign-worker status moved AUC by only 0.002, so I **excluded them** at no real accuracy cost.

## Recommendation: a three-tier strategy
| Tier | Who | Action |
|---|---|---|
| **Tighten** | Negative checking balance combined with long term (>36 mo) or very large amount | Decline or require added support (guarantor, smaller amount, shorter term) |
| **Hold** | Mid-risk majority | Keep current terms; monitor |
| **Grow** | Short-term loans (≤12 mo), high-savings and no-checking segments, used-car purposes | Target marketing and faster approval |

If the business needs transparency, a **three-rule policy** (decline negative checking balance, terms >36 months, or top-quartile amounts) approves 53% of applicants at a 18% default rate and +$86K, capturing about 80% of the model's improvement.

## How sure are we?
- **Out-of-sample check:** choosing the cutoff on half the loans and testing it on the other half, 200 times, beat approve-all in every repeat (average gain ≈ $298 per applicant, 5th–95th percentile $233–$367). Approve-all is a low bar here, so the real test is a pilot.
- **Sensitivity:** across nine combinations of profit (15–25%) and loss (40–60%) assumptions, the cutoff policy always beat approve-all. The best approval rate ranged from 35% to 58%, so **the shape of the recommendation is stable but the exact cutoff depends on loss economics we must confirm.**

## Risks and limits
- Small, dated, non-U.S. sample; defaults are far higher than a typical lender's portfolio.
- Only approved loans are observed, so we cannot see how declined applicants would have performed (selection bias).
- Findings are correlational. Some labels look counterintuitive (e.g., "no checking account" is the safest segment), which probably reflects how the data was collected and should be checked with data owners.
- **Fair-lending flag:** the policy approves 42% of applicants aged 25 or under vs. 62% of those 51+. Under-25s also had a higher actual default rate (42%), but any real deployment needs compliance and disparate-impact review before use.

## Proposed pilot
Run the new policy on a randomly assigned share of applications against the current policy. To detect a drop in default rate from 30% to 20% with 80% power at 5% significance requires about **292 loans per arm**; detecting 30% → 25% requires about **1,250 per arm**. Track default rate, approval rate, profit per loan, and approval rates by age band as a fairness guardrail, and read results only after enough loans have matured.

## Next steps
1. Confirm true revenue and loss-given-default per product.
2. Add rejected-applicant data to correct for selection bias (reject inference).
3. Repeat on a larger lending dataset and re-test stability over time.
4. Brief Compliance on the age-band gap before any pilot.
