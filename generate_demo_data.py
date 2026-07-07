# ═══════════════════════════════════════════════════════════════════════════════
# generate_demo_data.py
# Builds a synthetic analysis_results.pkl with fabricated numbers, matching the
# exact schema app.py expects. Safe to publish publicly — no real client data.
# Run once: python generate_demo_data.py  →  produces analysis_results.pkl
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import pickle

rng = np.random.default_rng(42)

CARDS = ["Platinum Elite", "Business Gold", "Corporate Starter"]
QUARTERS = ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"]
MCC = ["Travel", "Dining", "Office Supplies", "Software/SaaS",
       "Fuel", "Lodging", "Entertainment", "Shipping"]
REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
CLUSTERS = ["High-Value Diversified", "Heavy Traveler", "Procurement-Led",
            "Occasional Spender", "Dormant / At-Risk"]

# ── kpi_summary ───────────────────────────────────────────────────────────────
by_card_latest = {"Platinum Elite": 2_450_000, "Business Gold": 1_680_000, "Corporate Starter": 512_000}
by_card_prev   = {"Platinum Elite": 2_310_000, "Business Gold": 1_540_000, "Corporate Starter": 471_000}
total_rev_latest = sum(by_card_latest.values())
total_rev_prev    = sum(by_card_prev.values())

kpi_summary = {
    "latest_quarter": "Q4 2024",
    "total_rev_latest": total_rev_latest,
    "total_rev_prev": total_rev_prev,
    "total_qoq_pct": (total_rev_latest - total_rev_prev) / total_rev_prev,
    "by_card_latest": by_card_latest,
    "by_card_prev": by_card_prev,
    "targets": {
        "Platinum Elite":    {"base_rev": 2_450_000, "target_rev": 2_695_000, "gap_usd": 245_000, "pct": 0.10},
        "Business Gold":     {"base_rev": 1_680_000, "target_rev": 2_100_000, "gap_usd": 420_000, "pct": 0.25},
        "Corporate Starter": {"base_rev": 512_000,   "target_rev": 742_400,   "gap_usd": 230_400, "pct": 0.45},
    },
}

# ── rev_by_quarter ────────────────────────────────────────────────────────────
rows = []
for ct in CARDS:
    base = {"Platinum Elite": 2_100_000, "Business Gold": 1_350_000, "Corporate Starter": 400_000}[ct]
    for i, q in enumerate(QUARTERS):
        growth = 1 + 0.05 * i + rng.normal(0, 0.01)
        total = base * growth
        fee_share = {"Platinum Elite": 0.35, "Business Gold": 0.28, "Corporate Starter": 0.0}[ct]
        fee = total * fee_share
        rows.append({
            "card_type": ct, "quarter": q,
            "total_rev_usd": total,
            "fee_rev_usd": fee,
            "interchange_rev_usd": total - fee,
        })
rev_by_quarter = pd.DataFrame(rows)

# ── spend_by_mcc ──────────────────────────────────────────────────────────────
rows = []
for ct in CARDS:
    for mcc in MCC:
        amt = rng.uniform(200_000, 4_000_000)
        yld = rng.uniform(0.008, 0.028)
        rows.append({"card_type": ct, "merchant_category": mcc,
                     "amount_usd": amt, "interchange_yield": yld})
spend_by_mcc = pd.DataFrame(rows)

# ── spend_trend_monthly ───────────────────────────────────────────────────────
months = pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%b %Y")
rows = []
for ct in CARDS:
    base = {"Platinum Elite": 700_000, "Business Gold": 450_000, "Corporate Starter": 130_000}[ct]
    for i, m in enumerate(months):
        rows.append({"card_type": ct, "month": m,
                     "total_spend": base * (1 + 0.03 * i) * rng.uniform(0.92, 1.08)})
spend_trend_monthly = pd.DataFrame(rows)

# ── card_features (per-card client-level features) ───────────────────────────
n_cards = 3000
size_order = ["50-200", "201-500", "501-2000", "2000+"]
card_type_choice = rng.choice(CARDS, size=n_cards, p=[0.25, 0.4, 0.35])
cluster_choice = rng.choice(CLUSTERS, size=n_cards, p=[0.2, 0.2, 0.2, 0.25, 0.15])
card_features = pd.DataFrame({
    "card_id": [f"CARD-{i:05d}" for i in range(n_cards)],
    "card_type": card_type_choice,
    "cluster_name": cluster_choice,
    "company_size": rng.choice(size_order, size=n_cards),
    "avg_monthly_spend": rng.gamma(3, 2500, size=n_cards),
    "avg_utilization": rng.uniform(0.05, 0.95, size=n_cards),
})

# ── cluster_summary / cluster_rev ─────────────────────────────────────────────
cluster_summary = card_features.groupby("cluster_name").size().reset_index(name="count")
cluster_rev = pd.DataFrame({
    "cluster_name": CLUSTERS,
    "total_interchange": rng.uniform(200_000, 1_800_000, size=len(CLUSTERS)),
})

# ── geo_rev ────────────────────────────────────────────────────────────────────
geo_rev = pd.DataFrame({
    "region": REGIONS,
    "total_spend": rng.uniform(3_000_000, 12_000_000, size=len(REGIONS)),
    "interchange": rng.uniform(60_000, 240_000, size=len(REGIONS)),
    "txn_count": rng.integers(20_000, 90_000, size=len(REGIONS)),
})

# ── ltv_by_segment ─────────────────────────────────────────────────────────────
rows = []
for cl in CLUSTERS:
    for ct in CARDS:
        rows.append({"cluster_name": cl, "card_type": ct,
                     "avg_ltv": rng.uniform(3_000, 45_000)})
ltv_by_segment = pd.DataFrame(rows)

# ── active_features (churn scoring) ───────────────────────────────────────────
n_active = 2500
active_features = pd.DataFrame({
    "card_id": [f"ACT-{i:05d}" for i in range(n_active)],
    "card_type": rng.choice(CARDS, size=n_active, p=[0.25, 0.4, 0.35]),
    "churn_score": np.clip(rng.normal(35, 22, size=n_active), 0, 100),
})

# ── churn_by_band ──────────────────────────────────────────────────────────────
def band(score):
    if score >= 66: return "High"
    if score >= 33: return "Medium"
    return "Low"

active_features["churn_risk_band"] = active_features["churn_score"].apply(band)
churn_by_band = (active_features.groupby("churn_risk_band")
                 .agg(count=("card_id", "count"))
                 .reset_index())
churn_by_band["rev_at_risk"] = churn_by_band["count"] * rng.uniform(800, 2200)

# ── warning_summary ────────────────────────────────────────────────────────────
warning_summary = {
    "spend_collapse": {"count": 142, "rev": 380_000},
    "recency":        {"count": 96,  "rev": 245_000},
    "low_util":       {"count": 210, "rev": 410_000},
}

# ── dormant_summary ────────────────────────────────────────────────────────────
dormant_summary = {
    "total_accounts": 318,
    "avg_prior_spend": 4_250,
    "recoverable_rev": 675_000,
    "by_card": {"Platinum Elite": 40, "Business Gold": 118, "Corporate Starter": 160},
}

# ── platinum_watch ─────────────────────────────────────────────────────────────
n_watch = 18
platinum_watch = pd.DataFrame({
    "Client": [f"Client {chr(65+i)}" for i in range(n_watch)],
    "Qtr Revenue ($)": rng.uniform(8_000, 60_000, size=n_watch),
    "Churn Score": rng.uniform(60, 95, size=n_watch),
    "Days Inactive": rng.integers(10, 90, size=n_watch),
})

# ── recommendations ────────────────────────────────────────────────────────────
recommendations = [
    {"id": "R1", "card": "Platinum Elite", "title": "Retention offer for at-risk high-LTV Platinum clients",
     "action": "Targeted concierge outreach + fee waiver for top 18 at-risk accounts.",
     "evidence": "Watch list represents $410K/qtr at risk with churn scores above 60.",
     "color": "#818cf8", "rev_q": [95_000, 110_000, 120_000, 125_000],
     "total_annual_rev": 450_000, "cost_usd": 35_000, "roi": 12.9},
    {"id": "R2", "card": "Business Gold", "title": "Category-based interchange yield improvement",
     "action": "Steer Gold cardholders toward higher-yield spend categories via targeted rewards.",
     "evidence": "Gold's lowest-yield category is 2.3x below its highest-yield category.",
     "color": "#f59e0b", "rev_q": [40_000, 65_000, 90_000, 110_000],
     "total_annual_rev": 305_000, "cost_usd": 28_000, "roi": 10.9},
    {"id": "R3", "card": "Corporate Starter", "title": "Upgrade mis-matched Starter clients to Gold",
     "action": "Proactively upgrade high-spend Starter clients whose profile matches Gold usage patterns.",
     "evidence": "214 Starter clients show spend behaviour consistent with Gold-tier clients.",
     "color": "#10b981", "rev_q": [20_000, 55_000, 95_000, 140_000],
     "total_annual_rev": 310_000, "cost_usd": 18_000, "roi": 17.2},
    {"id": "R4", "card": "Business Gold", "title": "Upgrade high-value Gold clients to Platinum",
     "action": "Invite top-quartile Gold spenders to Platinum with a limited-time fee waiver.",
     "evidence": "68 Gold clients show Platinum-level spend but haven't upgraded.",
     "color": "#818cf8", "rev_q": [15_000, 40_000, 70_000, 95_000],
     "total_annual_rev": 220_000, "cost_usd": 22_000, "roi": 10.0},
    {"id": "R5", "card": "Corporate Starter", "title": "Dormant account reactivation campaign",
     "action": "Targeted cashback offer to reactivate the 160 dormant Starter accounts.",
     "evidence": "Dormant accounts averaged $4,250/mo in spend before going inactive.",
     "color": "#22c55e", "rev_q": [60_000, 80_000, 90_000, 100_000],
     "total_annual_rev": 330_000, "cost_usd": 20_000, "roi": 16.5},
    {"id": "R6", "card": "Platinum Elite", "title": "Expand T&E category partnerships",
     "action": "Negotiate enhanced interchange terms with top 3 travel merchant partners.",
     "evidence": "Platinum's T&E spend is 3x the portfolio average but yield hasn't scaled with it.",
     "color": "#f59e0b", "rev_q": [10_000, 30_000, 55_000, 80_000],
     "total_annual_rev": 175_000, "cost_usd": 15_000, "roi": 11.7},
]

# ── rev_impact_table ───────────────────────────────────────────────────────────
impact_rows = []
cumulative = 0
for r in recommendations:
    cumulative += r["total_annual_rev"]
    impact_rows.append({
        "Action": r["id"], "Card": r["card"],
        "Q3 2024": r["rev_q"][0], "Q4 2024": r["rev_q"][1],
        "Q1 2025": r["rev_q"][2], "Q2 2025": r["rev_q"][3],
        "Annual": r["total_annual_rev"], "ROI": r["roi"],
    })
impact_rows.append({
    "Action": "Total", "Card": "All",
    "Q3 2024": sum(r["rev_q"][0] for r in recommendations),
    "Q4 2024": sum(r["rev_q"][1] for r in recommendations),
    "Q1 2025": sum(r["rev_q"][2] for r in recommendations),
    "Q2 2025": sum(r["rev_q"][3] for r in recommendations),
    "Annual": cumulative,
    "ROI": cumulative / sum(r["cost_usd"] for r in recommendations),
})
rev_impact_table = pd.DataFrame(impact_rows)

# ── forecast ───────────────────────────────────────────────────────────────────
quarters_fc = ["Q3 2024", "Q4 2024", "Q1 2025", "Q2 2025"]
baseline_rev = [total_rev_latest * (1 + 0.02 * i) for i in range(4)]
incremental_rev = [sum(r["rev_q"][i] for r in recommendations) for i in range(4)]
total_rev_fc = [b + i for b, i in zip(baseline_rev, incremental_rev)]
cumulative_uplift = np.cumsum(incremental_rev).tolist()

forecast = pd.DataFrame({
    "quarter": quarters_fc,
    "baseline_rev": baseline_rev,
    "incremental_rev": incremental_rev,
    "total_rev": total_rev_fc,
    "cumulative_uplift": cumulative_uplift,
})

# ── target_check ───────────────────────────────────────────────────────────────
target_check = {
    "Platinum Elite":    {"base": 2_450_000, "target": 2_695_000, "forecast": 2_710_000, "achieved": True},
    "Business Gold":      {"base": 1_680_000, "target": 2_100_000, "forecast": 2_050_000, "achieved": False},
    "Corporate Starter":  {"base": 512_000,   "target": 742_400,   "forecast": 760_000,   "achieved": True},
}

# ── upgrade tables + summary ───────────────────────────────────────────────────
def make_upgrade_df(n, label):
    return pd.DataFrame({
        "Client": [f"{label} Client {i+1}" for i in range(n)],
        "Avg Monthly Spend ($)": rng.uniform(3_000, 18_000, size=n),
        "Qtr Uplift ($)": rng.uniform(500, 4_000, size=n),
        "Annual Uplift ($)": rng.uniform(2_000, 16_000, size=n),
    })

upgrade_gold_display = make_upgrade_df(214, "Starter→Gold")
upgrade_plat_display = make_upgrade_df(68, "Gold→Plat")

upgrade_summary = {
    "starter_to_gold": {
        "count": 214,
        "total_annual": upgrade_gold_display["Annual Uplift ($)"].sum(),
        "avg_per_client": upgrade_gold_display["Annual Uplift ($)"].mean(),
    },
    "gold_to_plat": {
        "count": 68,
        "total_annual": upgrade_plat_display["Annual Uplift ($)"].sum(),
        "avg_per_client": upgrade_plat_display["Annual Uplift ($)"].mean(),
    },
}

# ── assemble & save ────────────────────────────────────────────────────────────
R = {
    "kpi_summary": kpi_summary,
    "rev_by_quarter": rev_by_quarter,
    "spend_by_mcc": spend_by_mcc,
    "spend_trend_monthly": spend_trend_monthly,
    "card_features": card_features,
    "cluster_summary": cluster_summary,
    "cluster_rev": cluster_rev,
    "geo_rev": geo_rev,
    "ltv_by_segment": ltv_by_segment,
    "active_features": active_features,
    "churn_by_band": churn_by_band,
    "warning_summary": warning_summary,
    "dormant_summary": dormant_summary,
    "platinum_watch": platinum_watch,
    "recommendations": recommendations,
    "rev_impact_table": rev_impact_table,
    "forecast": forecast,
    "target_check": target_check,
    "upgrade_gold_display": upgrade_gold_display,
    "upgrade_plat_display": upgrade_plat_display,
    "upgrade_summary": upgrade_summary,
}

with open("analysis_results.pkl", "wb") as f:
    pickle.dump(R, f)

print("✅ analysis_results.pkl created with synthetic demo data.")
