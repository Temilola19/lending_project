"""
Loan Approval Strategy: Growth vs. Risk
Real data: German Credit (UCI), 1,000 loans, bundled in the `scorecardpy` package.
Run: python analysis.py   (writes tables/ and charts/ and results.json)
"""
import json, os
import numpy as np
import pandas as pd
import scorecardpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)
os.makedirs("tables", exist_ok=True)
os.makedirs("charts", exist_ok=True)
R = {}  # results collected for the memo

# ------------------------------------------------------------------ 1. LOAD + CLEAN
df = sc.germancredit().copy()
df = df.rename(columns={
    "status_of_existing_checking_account": "checking",
    "duration_in_month": "duration",
    "credit_history": "history",
    "credit_amount": "amount",
    "savings_account_and_bonds": "savings",
    "present_employment_since": "employment",
    "installment_rate_in_percentage_of_disposable_income": "install_rate",
    "personal_status_and_sex": "status_sex",
    "other_debtors_or_guarantors": "debtors",
    "present_residence_since": "residence",
    "age_in_years": "age",
    "other_installment_plans": "other_plans",
    "number_of_existing_credits_at_this_bank": "n_credits",
    "number_of_people_being_liable_to_provide_maintenance_for": "n_liable",
})
for c in df.select_dtypes(include=["category", "object", "string"]).columns:
    df[c] = df[c].astype(str).str.strip()
df["bad"] = (df["creditability"] == "bad").astype(int)

R["n_loans"] = len(df)
R["missing_cells"] = int(df.isna().sum().sum())
R["duplicates"] = int(df.duplicated().sum())
R["bad_rate"] = round(df["bad"].mean(), 4)
R["avg_amount"] = round(df["amount"].mean(), 0)
R["avg_amount_good"] = round(df.loc[df.bad == 0, "amount"].mean(), 0)
R["avg_amount_bad"] = round(df.loc[df.bad == 1, "amount"].mean(), 0)

# Economics: ASSUMPTIONS (not in the data). Stated up front and stress-tested later.
GAIN, LOSS = 0.20, 0.50  # good loan earns 20% of principal over its life; bad loan loses 50% (net of recovery)
df["profit"] = np.where(df["bad"] == 1, -LOSS * df["amount"], GAIN * df["amount"])
R["assumptions"] = {"gain_on_good": GAIN, "loss_on_bad": LOSS}
R["approve_all_profit"] = round(df["profit"].sum(), 0)
R["approve_all_profit_per_loan"] = round(df["profit"].mean(), 1)

# Friendly buckets
df["duration_band"] = pd.cut(df["duration"], [0, 12, 24, 36, 100],
                             labels=["<=12m", "13-24m", "25-36m", ">36m"])
df["amount_band"] = pd.qcut(df["amount"], 4, labels=["Q1 small", "Q2", "Q3", "Q4 large"])

# ------------------------------------------------------------------ 2. SEGMENT VIEW
def seg_table(col):
    g = df.groupby(col, observed=True).agg(
        loans=("bad", "size"), bad_loans=("bad", "sum"),
        bad_rate=("bad", "mean"), total_amount=("amount", "sum"),
        profit=("profit", "sum"))
    g["share_of_loans"] = g["loans"] / len(df)
    g["share_of_bad_loans"] = g["bad_loans"] / df["bad"].sum()
    g["profit_per_loan"] = g["profit"] / g["loans"]
    g["bad_rate_vs_overall"] = g["bad_rate"] / df["bad"].mean()
    # chi-square for association between segment and default
    chi2, p, _, _ = chi2_contingency(pd.crosstab(df[col], df["bad"]))
    g.attrs["p"] = p
    return g.sort_values("bad_rate"), p

seg_cols = ["checking", "history", "purpose", "savings", "duration_band", "amount_band", "employment"]
all_seg = []
R["segments"] = {}
for c in seg_cols:
    t, p = seg_table(c)
    t.round(4).to_csv(f"tables/segment_{c}.csv")
    R["segments"][c] = {"p_value": float(p),
                        "table": t.round(4).reset_index().to_dict(orient="records")}
    tt = t.reset_index().rename(columns={c: "segment"})
    tt.insert(0, "dimension", c)
    all_seg.append(tt)
pd.concat(all_seg).round(4).to_csv("tables/all_segments.csv", index=False)

# Chart 1: default rate by key segments
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
for ax, c, title in zip(axes, ["checking", "duration_band", "history"],
                        ["Checking account status", "Loan duration", "Credit history"]):
    t = R["segments"][c]["table"]
    d = pd.DataFrame(t).sort_values("bad_rate")
    labs = [str(x)[:34] for x in d[c]]
    bars = ax.barh(labs, d["bad_rate"] * 100, color="#2f6f9f")
    ax.axvline(df["bad"].mean() * 100, color="#c0392b", ls="--", lw=1)
    for b, n in zip(bars, d["loans"]):
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2, f"n={n}", va="center", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Default (bad) rate, %")
fig.suptitle(f"Default rate by segment (red dashed = overall {df['bad'].mean()*100:.0f}%)", fontsize=13)
plt.tight_layout()
plt.savefig("charts/01_default_rate_by_segment.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 3. MODELS
PROTECTED = ["age", "status_sex", "foreign_worker"]  # sensitive / proxy attributes: excluded from the decision model
num_cols = ["duration", "amount", "install_rate", "residence", "n_credits", "n_liable"]
cat_cols = ["checking", "history", "purpose", "savings", "employment", "debtors",
            "property", "other_plans", "housing", "job", "telephone"]
y = df["bad"].values

def make_lr(num, cat):
    pre = ColumnTransformer([("n", StandardScaler(), num),
                             ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre), ("m", LogisticRegression(C=0.5, max_iter=3000))])

def make_gb(num, cat):
    pre = ColumnTransformer([("n", "passthrough", num),
                             ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    return Pipeline([("pre", pre), ("m", HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED))])

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
X_fair = df[num_cols + cat_cols]
X_full = df[num_cols + cat_cols + PROTECTED + []]
models = {
    "Logistic regression (policy-safe features)": (make_lr(num_cols, cat_cols), X_fair),
    "Gradient boosting (policy-safe features)": (make_gb(num_cols, cat_cols), X_fair),
    "Logistic regression (+ age, sex/status, foreign worker)":
        (make_lr(num_cols + ["age"], cat_cols + ["status_sex", "foreign_worker"]),
         df[num_cols + ["age"] + cat_cols + ["status_sex", "foreign_worker"]]),
}
R["model_auc"] = {}
for name, (m, X) in models.items():
    s = cross_val_score(m, X, y, cv=cv, scoring="roc_auc")
    R["model_auc"][name] = {"mean": round(s.mean(), 3), "sd": round(s.std(), 3)}

# Out-of-fold predicted default probability from the chosen model (every loan scored by a model that never saw it)
chosen = make_lr(num_cols, cat_cols)
df["pd"] = cross_val_predict(chosen, X_fair, y, cv=cv, method="predict_proba")[:, 1]

# Drivers: odds ratios from the final fit
chosen.fit(X_fair, y)
names = chosen.named_steps["pre"].get_feature_names_out()
coefs = chosen.named_steps["m"].coef_[0]
drv = pd.DataFrame({"feature": names, "coef": coefs, "odds_ratio": np.exp(coefs)})
drv["feature"] = drv["feature"].str.replace("n__", "", regex=False).str.replace("c__", "", regex=False)
drv = drv.sort_values("coef")
drv.round(3).to_csv("tables/model_drivers.csv", index=False)
R["drivers_up"] = drv.tail(6).round(3).to_dict(orient="records")[::-1]
R["drivers_down"] = drv.head(6).round(3).to_dict(orient="records")

# Chart 2: drivers
top = pd.concat([drv.head(7), drv.tail(7)])
fig, ax = plt.subplots(figsize=(9, 5.5))
cols = ["#2e8b57" if v < 0 else "#c0392b" for v in top["coef"]]
ax.barh([f[:55] for f in top["feature"]], top["coef"], color=cols)
ax.axvline(0, color="k", lw=0.8)
ax.set_title("What moves default risk (logistic regression coefficients)\nred = more risk, green = less risk", fontsize=11)
ax.set_xlabel("Coefficient (log-odds)")
plt.tight_layout()
plt.savefig("charts/02_risk_drivers.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 4. POLICY SIMULATION
def evaluate(mask, gain=GAIN, loss=LOSS):
    sub = df[mask]
    profit = np.where(sub["bad"] == 1, -loss * sub["amount"], gain * sub["amount"]).sum()
    return {"approved": int(mask.sum()), "approval_rate": float(mask.mean()),
            "bad_rate_approved": float(sub["bad"].mean()) if len(sub) else np.nan,
            "profit": float(profit),
            "profit_per_approved": float(profit / max(len(sub), 1))}

ths = np.round(np.arange(0.05, 0.81, 0.01), 2)
rows = []
for t in ths:
    e = evaluate(df["pd"] < t)
    e["threshold"] = t
    rows.append(e)
curve = pd.DataFrame(rows)
curve.round(4).to_csv("tables/policy_curve.csv", index=False)

best = curve.loc[curve["profit"].idxmax()]
base = evaluate(pd.Series(True, index=df.index))
R["policy_baseline"] = {k: round(v, 4) for k, v in base.items()}
R["policy_best"] = {k: round(float(v), 4) for k, v in best.items()}

# Named, easy-to-explain policies
def at_rate(target):  # threshold giving approval rate closest to target
    r = curve.iloc[(curve["approval_rate"] - target).abs().argsort().iloc[0]]
    return {k: round(float(v), 4) for k, v in r.items()}
R["policy_at_approval"] = {f"{int(x*100)}%": at_rate(x) for x in [0.9, 0.8, 0.7, 0.6, 0.5]}

# Honest check: pick the threshold on half the loans, score it on the other half (200 repeats)
uplifts, rates_chosen = [], []
for _ in range(200):
    idx = rng.permutation(len(df))
    a, b = df.iloc[idx[: len(df)//2]], df.iloc[idx[len(df)//2:]]
    best_t, best_p = None, -1e18
    for t in ths:
        m = a["pd"] < t
        p = np.where(a.loc[m, "bad"] == 1, -LOSS, GAIN) @ a.loc[m, "amount"].values
        if p > best_p:
            best_p, best_t = p, t
    mb = b["pd"] < best_t
    p_pol = (np.where(b.loc[mb, "bad"] == 1, -LOSS, GAIN) * b.loc[mb, "amount"]).sum()
    p_all = (np.where(b["bad"] == 1, -LOSS, GAIN) * b["amount"]).sum()
    uplifts.append((p_pol - p_all) / len(b))   # profit uplift per applicant
    rates_chosen.append(mb.mean())
R["holdout_check"] = {
    "uplift_per_applicant_mean": round(float(np.mean(uplifts)), 1),
    "uplift_p5": round(float(np.percentile(uplifts, 5)), 1),
    "uplift_p95": round(float(np.percentile(uplifts, 95)), 1),
    "share_of_repeats_better_than_approve_all": round(float(np.mean(np.array(uplifts) > 0)), 3),
    "approval_rate_mean": round(float(np.mean(rates_chosen)), 3),
}

# Sensitivity of the best threshold to the economic assumptions
sens = []
for g in [0.15, 0.20, 0.25]:
    for l in [0.40, 0.50, 0.60]:
        best_t, best_p = None, -1e18
        for t in ths:
            e = evaluate(df["pd"] < t, g, l)
            if e["profit"] > best_p:
                best_p, best_t, ar = e["profit"], t, e["approval_rate"]
        ba = evaluate(pd.Series(True, index=df.index), g, l)["profit"]
        sens.append({"gain": g, "loss": l, "best_threshold": float(best_t),
                     "approval_rate": round(ar, 3), "profit_best": round(best_p),
                     "profit_approve_all": round(ba)})
pd.DataFrame(sens).to_csv("tables/sensitivity.csv", index=False)
R["sensitivity"] = sens

# Chart 3: trade-off curve
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(curve["approval_rate"] * 100, curve["bad_rate_approved"] * 100, color="#c0392b", lw=2, label="Default rate of approved loans")
ax1.set_xlabel("Approval rate, %")
ax1.set_ylabel("Default rate among approved, %", color="#c0392b")
ax2 = ax1.twinx()
ax2.plot(curve["approval_rate"] * 100, curve["profit"] / 1000, color="#2f6f9f", lw=2, label="Total profit ($K)")
ax2.axhline(0, color="gray", lw=0.7)
ax2.set_ylabel("Total expected profit, $K (assumption-driven)", color="#2f6f9f")
ax1.axvline(best["approval_rate"] * 100, color="k", ls="--", lw=1)
ax1.text(best["approval_rate"] * 100 + 1, ax1.get_ylim()[0] + 1,
         f"profit-max: approve {best['approval_rate']*100:.0f}%", fontsize=9)
ax1.invert_xaxis()
plt.title("Growth vs. risk: approving more raises both volume and defaults", fontsize=11)
plt.tight_layout()
plt.savefig("charts/03_policy_tradeoff.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 5. SEGMENT ACTION MAP
# Which segments make money under approve-all vs lose money?
act = pd.concat(all_seg)
act = act[act["loans"] >= 30].copy()
act["verdict"] = np.where(act["profit_per_loan"] > 0, "profitable", "loses money")
act.round(4).sort_values("profit_per_loan").to_csv("tables/segment_action_map.csv", index=False)
R["worst_segments"] = act.sort_values("profit_per_loan").head(6)[
    ["dimension", "segment", "loans", "bad_rate", "profit_per_loan"]].round(3).to_dict(orient="records")
R["best_segments"] = act.sort_values("profit_per_loan", ascending=False).head(6)[
    ["dimension", "segment", "loans", "bad_rate", "profit_per_loan"]].round(3).to_dict(orient="records")

# Chart 4: profit per loan by segment (selected)
sel = act[act["dimension"].isin(["checking", "purpose", "duration_band"])].sort_values("profit_per_loan")
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.barh([f"{r.dimension}: {str(r.segment)[:32]}" for r in sel.itertuples()],
        sel["profit_per_loan"], color=["#c0392b" if v < 0 else "#2e8b57" for v in sel["profit_per_loan"]])
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Expected profit per loan ($) under approve-all, using stated assumptions")
ax.set_title("Which segments make or lose money", fontsize=11)
plt.tight_layout()
plt.savefig("charts/04_segment_profit.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 6. PILOT DESIGN
# Sample size per arm to detect a drop in default rate among approved loans
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
def n_per_arm(p1, p2, power=0.8, alpha=0.05):
    es = proportion_effectsize(p1, p2)
    return int(np.ceil(NormalIndPower().solve_power(es, power=power, alpha=alpha, ratio=1)))
R["pilot_sample_sizes"] = {
    f"{int(p1*100)}% -> {int(p2*100)}%": n_per_arm(p1, p2)
    for p1, p2 in [(0.30, 0.25), (0.30, 0.20), (0.25, 0.20)]
}

# Fairness spot check: approval rate of the chosen policy by (excluded) sensitive attributes
pol = df["pd"] < best["threshold"]
fair = []
df["age_band"] = pd.cut(df["age"], [0, 25, 35, 50, 100], labels=["<=25", "26-35", "36-50", "51+"])
for c in ["age_band", "foreign_worker"]:
    for k, g in df.groupby(c, observed=True):
        fair.append({"attribute": c, "group": str(k), "n": len(g),
                     "approval_rate_under_policy": round(float(pol[g.index].mean()), 3),
                     "actual_bad_rate": round(float(g["bad"].mean()), 3)})
pd.DataFrame(fair).to_csv("tables/fairness_spot_check.csv", index=False)
R["fairness"] = fair

df.drop(columns=["creditability"]).to_csv("tables/scored_loans.csv", index=False)
with open("results.json", "w") as f:
    json.dump(R, f, indent=2, default=float)
print(json.dumps(R, indent=1, default=float)[:12000])
