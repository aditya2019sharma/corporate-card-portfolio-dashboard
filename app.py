# ═══════════════════════════════════════════════════════════════════════════════
# CORPORATE CARD PORTFOLIO DASHBOARD — app.py
# Run: streamlit run app.py
# Requires: analysis_results.pkl in the same folder
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import pickle

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Corporate Card Portfolio",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME COLORS ──────────────────────────────────────────────────────────────
C_PLAT   = "#818cf8"
C_GOLD   = "#f59e0b"
C_START  = "#10b981"
C_BG     = "#0f172a"
C_CARD   = "#1e293b"
C_TEXT   = "#f1f5f9"
C_MUTED  = "#94a3b8"
C_GREEN  = "#22c55e"
C_RED    = "#ef4444"
C_AMBER  = "#f59e0b"

CARD_COLORS = {
    "Platinum Elite":    C_PLAT,
    "Business Gold":     C_GOLD,
    "Corporate Starter": C_START,
}

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark theme base */
.stApp { background-color: #0f172a; color: #f1f5f9; }
section[data-testid="stSidebar"] { background-color: #1e293b; }

/* KPI metric cards */
.kpi-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    height: 100%;
}
.kpi-label  { font-size: 12px; color: #94a3b8; text-transform: uppercase;
               letter-spacing: 0.08em; margin-bottom: 6px; }
.kpi-value  { font-size: 28px; font-weight: 700; color: #f1f5f9; line-height: 1.1; }
.kpi-delta  { font-size: 13px; font-weight: 500; margin-top: 4px; }
.kpi-pos    { color: #22c55e; }
.kpi-neg    { color: #ef4444; }
.kpi-sub    { font-size: 11px; color: #64748b; margin-top: 3px; }

/* Section headers */
.section-header {
    font-size: 11px; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 24px 0 12px; border-bottom: 1px solid #1e293b; padding-bottom: 6px;
}

/* Insight box */
.insight-box {
    background: #1e293b; border-left: 3px solid;
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0;
}
.insight-title { font-size: 13px; font-weight: 600; color: #f1f5f9; margin-bottom: 4px; }
.insight-body  { font-size: 12px; color: #94a3b8; line-height: 1.5; }
.insight-money { font-size: 13px; font-weight: 700; color: #22c55e; margin-top: 6px; }

/* RAG status */
.rag-green  { color: #22c55e; font-weight: 600; }
.rag-amber  { color: #f59e0b; font-weight: 600; }
.rag-red    { color: #ef4444; font-weight: 600; }

/* Table styling */
.styled-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.styled-table th { background: #1e293b; color: #94a3b8; padding: 8px 12px;
                    text-align: left; font-size: 11px; text-transform: uppercase; }
.styled-table td { padding: 8px 12px; border-bottom: 1px solid #1e293b; color: #f1f5f9; }
.styled-table tr:hover td { background: #1e293b; }

/* Chapter nav pills */
.nav-pill {
    display: inline-block; padding: 6px 14px; border-radius: 999px;
    font-size: 12px; font-weight: 500; margin: 2px;
    background: #1e293b; color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_results():
    with open("analysis_results.pkl", "rb") as f:
        return pickle.load(f)

try:
    R = load_results()
except FileNotFoundError:
    st.error("❌ analysis_results.pkl not found. Run analysis.py first.")
    st.stop()

# Unpack commonly used objects
kpi            = R["kpi_summary"]
targets        = kpi["targets"]
rev_by_q       = R["rev_by_quarter"]
spend_mcc      = R["spend_by_mcc"]
spend_trend    = R["spend_trend_monthly"]
card_feat      = R["card_features"]
cluster_sum    = R["cluster_summary"]
cluster_rev    = R["cluster_rev"]
geo_rev        = R["geo_rev"]
ltv_seg        = R["ltv_by_segment"]
active_feat    = R["active_features"]
churn_band     = R["churn_by_band"]
warn_sum       = R["warning_summary"]
dorm_sum       = R["dormant_summary"]
plat_watch     = R["platinum_watch"]
recs           = R["recommendations"]
impact_tbl     = R["rev_impact_table"]
forecast       = R["forecast"]
target_check   = R["target_check"]
upg_gold       = R["upgrade_gold_display"]
upg_plat       = R["upgrade_plat_display"]
upg_sum        = R["upgrade_summary"]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Card Portfolio")
    st.markdown("---")
    chapter = st.radio(
        "Navigate",
        ["1 · Executive Briefing",
         "2 · Portfolio Health",
         "3 · Customer Intelligence",
         "4 · Risk & Churn",
         "5 · Recommendations",
         "6 · Revenue Roadmap"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f"<div style='font-size:11px;color:#64748b'>Latest quarter<br>"
                f"<b style='color:#f1f5f9'>{kpi['latest_quarter']}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:#64748b;margin-top:8px'>Portfolio revenue<br>"
                f"<b style='color:#22c55e'>${kpi['total_rev_latest']/1e6:.2f}M / qtr</b></div>",
                unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:#64748b;margin-top:8px'>Incremental opportunity<br>"
                f"<b style='color:#818cf8'>${impact_tbl.iloc[-1]['Annual']/1e6:.2f}M / yr</b></div>",
                unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Corporate Card Portfolio Analytics · 2024")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Plotly dark figure defaults
# ══════════════════════════════════════════════════════════════════════════════
def dark_fig(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_TEXT, size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
        yaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    )
    return fig

def fmt_usd(v, decimals=0):
    if v >= 1e6:  return f"${v/1e6:.{decimals}f}M"
    if v >= 1e3:  return f"${v/1e3:.{decimals}f}K"
    return f"${v:.0f}"

def kpi_card(label, value, delta=None, delta_label="", sub=None, color=None):
    delta_html = ""
    if delta is not None:
        cls  = "kpi-pos" if delta >= 0 else "kpi-neg"
        sign = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{sign} {abs(delta)*100:.1f}% {delta_label}</div>'
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    val_color = color or C_TEXT
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{val_color}">{value}</div>
        {delta_html}{sub_html}
    </div>"""

def insight(title, body, money=None, color=C_PLAT):
    money_html = f'<div class="insight-money">{money}</div>' if money else ""
    return f"""
    <div class="insight-box" style="border-color:{color}">
        <div class="insight-title">{title}</div>
        <div class="insight-body">{body}</div>
        {money_html}
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# CH 1 — EXECUTIVE BRIEFING
# ══════════════════════════════════════════════════════════════════════════════
if chapter == "1 · Executive Briefing":
    st.markdown("## Where We Stand")
    st.markdown(f"<div style='color:{C_MUTED};font-size:13px;margin-bottom:24px'>"
                f"Portfolio performance as of {kpi['latest_quarter']} · "
                f"Target: Platinum +10% · Gold +25% · Starter +45%</div>",
                unsafe_allow_html=True)

    # ── Row 1: Top-line KPIs ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    qoq = kpi["total_qoq_pct"]
    with c1:
        st.markdown(kpi_card(
            "Total Portfolio Revenue",
            fmt_usd(kpi["total_rev_latest"]),
            delta=qoq, delta_label="QoQ",
            sub=f"vs {fmt_usd(kpi['total_rev_prev'])} prior quarter"
        ), unsafe_allow_html=True)
    with c2:
        pt_rev  = kpi["by_card_latest"].get("Platinum Elite", 0)
        pt_prev = kpi["by_card_prev"].get("Platinum Elite", 0)
        st.markdown(kpi_card(
            "Platinum Elite Revenue",
            fmt_usd(pt_rev),
            delta=(pt_rev-pt_prev)/pt_prev if pt_prev else 0, delta_label="QoQ",
            color=C_PLAT
        ), unsafe_allow_html=True)
    with c3:
        gd_rev  = kpi["by_card_latest"].get("Business Gold", 0)
        gd_prev = kpi["by_card_prev"].get("Business Gold", 0)
        st.markdown(kpi_card(
            "Business Gold Revenue",
            fmt_usd(gd_rev),
            delta=(gd_rev-gd_prev)/gd_prev if gd_prev else 0, delta_label="QoQ",
            color=C_GOLD
        ), unsafe_allow_html=True)
    with c4:
        st_rev  = kpi["by_card_latest"].get("Corporate Starter", 0)
        st_prev = kpi["by_card_prev"].get("Corporate Starter", 0)
        st.markdown(kpi_card(
            "Corporate Starter Revenue",
            fmt_usd(st_rev),
            delta=(st_rev-st_prev)/st_prev if st_prev else 0, delta_label="QoQ",
            color=C_START
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Gap to target + Revenue trend ─────────────────────────────────
    col_left, col_right = st.columns([1, 1.6])

    with col_left:
        st.markdown('<div class="section-header">Gap to Target — This Quarter</div>',
                    unsafe_allow_html=True)
        for ct, t in targets.items():
            pct_done = min(1.0, t["base_rev"] / t["target_rev"]) if t["target_rev"] else 1
            color    = CARD_COLORS[ct]
            gap      = t["gap_usd"]
            st.markdown(f"""
            <div style="margin-bottom:16px">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="font-size:13px;font-weight:500;color:{color}">{ct}</span>
                    <span style="font-size:12px;color:{C_MUTED}">
                        {fmt_usd(t['base_rev'])} → {fmt_usd(t['target_rev'])}
                        &nbsp;<b style="color:{C_AMBER}">gap: {fmt_usd(gap)}</b>
                    </span>
                </div>
                <div style="background:#1e293b;border-radius:999px;height:8px;overflow:hidden">
                    <div style="background:{color};width:{pct_done*100:.0f}%;height:100%;
                                border-radius:999px;transition:width 0.3s"></div>
                </div>
                <div style="font-size:11px;color:{C_MUTED};margin-top:2px">
                    Target: +{t['pct']*100:.0f}% · Currently at {pct_done*100:.0f}% of target
                </div>
            </div>
            """, unsafe_allow_html=True)

        total_gap = sum(t["gap_usd"] for t in targets.values())
        st.markdown(f"""
        <div style="background:#1e293b;border-radius:8px;padding:12px 16px;margin-top:8px">
            <div style="font-size:11px;color:{C_MUTED}">Total quarterly gap to hit all targets</div>
            <div style="font-size:22px;font-weight:700;color:{C_AMBER}">{fmt_usd(total_gap)}</div>
            <div style="font-size:11px;color:{C_MUTED};margin-top:2px">
                = {fmt_usd(total_gap*4)} annualised · Addressed by 6 recommendations in Ch.5
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">Quarterly Revenue Trend by Card</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        for ct, color in CARD_COLORS.items():
            d = rev_by_q[rev_by_q.card_type == ct].sort_values("quarter")
            fig.add_trace(go.Bar(
                name=ct, x=d["quarter"], y=d["total_rev_usd"],
                marker_color=color, opacity=0.85,
                hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>"
            ))
        fig.update_layout(barmode="stack", xaxis_title="", yaxis_title="Revenue ($)")
        st.plotly_chart(dark_fig(fig), use_container_width=True)

    # ── Row 3: What this dashboard answers ───────────────────────────────────
    st.markdown('<div class="section-header">What The Next 5 Chapters Answer</div>',
                unsafe_allow_html=True)
    cols = st.columns(5)
    chapters_desc = [
        ("📊", "Portfolio Health", "Which parts of our portfolio are healthy, stalling, or declining?"),
        ("👥", "Customer Intel",   "Which segments drive revenue — and who are we under-serving?"),
        ("⚠️", "Risk & Churn",    "What revenue are we about to lose, and which clients need action now?"),
        ("💡", "Recommendations",  "6 evidence-backed actions with dollar impact per card."),
        ("📈", "Revenue Roadmap",  "Quarter-by-quarter forecast showing when the money shows up."),
    ]
    for col, (icon, title, desc) in zip(cols, chapters_desc):
        with col:
            st.markdown(f"""
            <div style="background:#1e293b;border-radius:10px;padding:14px;height:100%;text-align:center">
                <div style="font-size:24px;margin-bottom:6px">{icon}</div>
                <div style="font-size:12px;font-weight:600;color:{C_TEXT};margin-bottom:4px">{title}</div>
                <div style="font-size:11px;color:{C_MUTED};line-height:1.4">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CH 2 — PORTFOLIO HEALTH
# ══════════════════════════════════════════════════════════════════════════════
elif chapter == "2 · Portfolio Health":
    st.markdown("## What Do We See")
    st.markdown(f"<div style='color:{C_MUTED};font-size:13px;margin-bottom:24px'>"
                f"Revenue composition, spend behaviour, and category yield analysis</div>",
                unsafe_allow_html=True)

    # ── 2.1: Revenue streams ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">2.1 · Revenue Composition — Fees vs Interchange by Quarter</div>',
                unsafe_allow_html=True)

    col_chart, col_insight = st.columns([2, 1])
    with col_chart:
        fig = make_subplots(rows=1, cols=3,
            subplot_titles=["Platinum Elite","Business Gold","Corporate Starter"])
        for i, (ct, color) in enumerate(CARD_COLORS.items(), 1):
            d = rev_by_q[rev_by_q.card_type == ct].sort_values("quarter")
            fig.add_trace(go.Bar(name="Interchange", x=d["quarter"],
                y=d["interchange_rev_usd"], marker_color=color,
                opacity=0.9, showlegend=(i==1),
                hovertemplate="Interchange: $%{y:,.0f}<extra></extra>"), row=1, col=i)
            fig.add_trace(go.Bar(name="Annual Fees", x=d["quarter"],
                y=d["fee_rev_usd"], marker_color="#334155",
                showlegend=(i==1),
                hovertemplate="Fees: $%{y:,.0f}<extra></extra>"), row=1, col=i)
        fig.update_layout(barmode="stack")
        st.plotly_chart(dark_fig(fig, height=320), use_container_width=True)

    with col_insight:
        st.markdown("<br><br>", unsafe_allow_html=True)
        total_fees = rev_by_q.fee_rev_usd.sum()
        total_ir   = rev_by_q.interchange_rev_usd.sum()
        total_all  = total_fees + total_ir
        st.markdown(insight(
            "Fee dependency",
            f"Annual fees account for {total_fees/total_all*100:.0f}% of total revenue. "
            f"Starter earns zero fee revenue in year 1 — its entire value is interchange + future upgrade.",
            money=f"Fee revenue: {fmt_usd(total_fees)} | Interchange: {fmt_usd(total_ir)}",
            color=C_PLAT
        ), unsafe_allow_html=True)
        st.markdown(insight(
            "Platinum punches above its weight",
            f"Platinum generates the highest interchange despite having fewer active cards — "
            f"driven by higher spend volume and better interchange rates on T&E categories.",
            color=C_GOLD
        ), unsafe_allow_html=True)

    st.markdown("---")

    # ── 2.2: Spend by category ───────────────────────────────────────────────
    st.markdown('<div class="section-header">2.2 · Spend by Merchant Category — Interchange Yield Heatmap</div>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns([1.8, 1])
    with col_left:
        pivot = spend_mcc.pivot(index="card_type", columns="merchant_category",
                                values="interchange_yield").fillna(0)
        fig = px.imshow(
            pivot,
            color_continuous_scale=[[0,"#1e293b"],[0.5,C_GOLD],[1,C_GREEN]],
            text_auto=".3f",
            aspect="auto",
        )
        fig.update_coloraxes(showscale=False)
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x}<br>Yield: %{z:.3f}<extra></extra>")
        st.plotly_chart(dark_fig(fig, height=220), use_container_width=True)
        st.caption("Numbers show interchange yield per dollar spent. Higher = more revenue per transaction.")

    with col_right:
        st.markdown("<br>", unsafe_allow_html=True)
        # Find lowest yield category per card
        for ct in CARD_COLORS:
            d = spend_mcc[spend_mcc.card_type == ct]
            if len(d):
                top_spend  = d.nlargest(1, "amount_usd").iloc[0]
                low_yield  = d.nsmallest(1, "interchange_yield").iloc[0]
                st.markdown(f"""
                <div style="background:#1e293b;border-radius:8px;padding:10px 12px;margin-bottom:8px">
                    <div style="font-size:11px;font-weight:600;color:{CARD_COLORS[ct]}">{ct}</div>
                    <div style="font-size:11px;color:{C_MUTED};margin-top:3px">
                        Top spend: <b style="color:{C_TEXT}">{top_spend['merchant_category']}</b>
                        (${top_spend['amount_usd']/1e6:.1f}M)
                    </div>
                    <div style="font-size:11px;color:{C_MUTED}">
                        Lowest yield: <b style="color:{C_RED}">{low_yield['merchant_category']}</b>
                        ({low_yield['interchange_yield']:.3f})
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 2.3: Spend trend ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">2.3 · Monthly Spend Trend — Last 12 Months</div>',
                unsafe_allow_html=True)

    fig = go.Figure()
    for ct, color in CARD_COLORS.items():
        d = spend_trend[spend_trend.card_type == ct].sort_values("month")
        fig.add_trace(go.Scatter(
            name=ct, x=d["month"], y=d["total_spend"],
            mode="lines+markers", line=dict(color=color, width=2),
            marker=dict(size=5),
            hovertemplate="<b>%{x}</b><br>Spend: $%{y:,.0f}<extra></extra>"
        ))
    fig.update_layout(xaxis_title="", yaxis_title="Total Spend ($)")
    st.plotly_chart(dark_fig(fig, height=300), use_container_width=True)

    # ── 2.4: Card distribution bubble ────────────────────────────────────────
    st.markdown('<div class="section-header">2.4 · Client Size vs Average Spend — Card Type Distribution</div>',
                unsafe_allow_html=True)

    size_order = ["50-200","201-500","501-2000","2000+"]
    size_map   = {s: i for i, s in enumerate(size_order)}
    plot_df    = card_feat.copy()
    plot_df    = plot_df[plot_df["company_size"].isin(size_order)]
    plot_df["size_num"] = plot_df["company_size"].map(size_map)

    fig = px.scatter(
        plot_df.sample(min(800, len(plot_df)), random_state=42),
        x="size_num", y="avg_monthly_spend",
        color="card_type",
        color_discrete_map=CARD_COLORS,
        size="avg_monthly_spend",
        size_max=20,
        hover_data={"size_num": False, "card_type": True,
                    "avg_monthly_spend": ":,.0f", "company_size": True},
        labels={"avg_monthly_spend": "Avg Monthly Spend ($)", "size_num": "Company Size"},
    )
    fig.update_xaxes(tickvals=list(range(4)), ticktext=size_order)
    fig.update_layout(yaxis_title="Avg Monthly Spend ($)")
    st.plotly_chart(dark_fig(fig, height=360), use_container_width=True)
    st.caption("Large companies on Corporate Starter (green, top-right) represent the highest-value upgrade targets.")


# ══════════════════════════════════════════════════════════════════════════════
# CH 3 — CUSTOMER INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif chapter == "3 · Customer Intelligence":
    st.markdown("## Who Are Our Customers")
    st.markdown(f"<div style='color:{C_MUTED};font-size:13px;margin-bottom:24px'>"
                f"Segmentation, LTV, geographic distribution, and upgrade pipeline</div>",
                unsafe_allow_html=True)

    # ── 3.1: Cluster profiles ────────────────────────────────────────────────
    st.markdown('<div class="section-header">3.1 · Five Customer Segments — Spend Profile & Revenue Share</div>',
                unsafe_allow_html=True)

    seg_colors = {
        "High-Value Diversified": "#818cf8",
        "Heavy Traveler":         "#f59e0b",
        "Procurement-Led":        "#10b981",
        "Occasional Spender":     "#64748b",
        "Dormant / At-Risk":      "#ef4444",
    }

    seg_agg = (card_feat
        .groupby("cluster_name")
        .agg(count=("card_id","count"),
             avg_spend=("avg_monthly_spend","mean"),
             avg_util=("avg_utilization","mean"))
        .reset_index()
    )
    seg_agg = seg_agg.merge(cluster_rev, on="cluster_name", how="left")
    seg_agg["rev_share"] = seg_agg["total_interchange"] / seg_agg["total_interchange"].sum()

    cols = st.columns(len(seg_agg))
    for col, (_, row) in zip(cols, seg_agg.iterrows()):
        color = seg_colors.get(row["cluster_name"], C_PLAT)
        with col:
            st.markdown(f"""
            <div style="background:#1e293b;border-top:3px solid {color};border-radius:0 0 10px 10px;
                        padding:14px;text-align:center;height:100%">
                <div style="font-size:12px;font-weight:600;color:{color};margin-bottom:8px;
                            min-height:32px">{row['cluster_name']}</div>
                <div style="font-size:22px;font-weight:700;color:{C_TEXT}">{row['count']}</div>
                <div style="font-size:11px;color:{C_MUTED}">cards</div>
                <div style="margin:8px 0;height:1px;background:#334155"></div>
                <div style="font-size:13px;font-weight:600;color:{C_GREEN}">
                    ${row['avg_spend']:,.0f}/mo</div>
                <div style="font-size:11px;color:{C_MUTED}">avg spend</div>
                <div style="font-size:12px;font-weight:600;color:{C_AMBER};margin-top:6px">
                    {row['rev_share']*100:.0f}% rev share</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── 3.2: Upgrade candidates ──────────────────────────────────────────────
    st.markdown('<div class="section-header">3.2 · Mis-Matched Clients — Right Spend Profile, Wrong Card</div>',
                unsafe_allow_html=True)

    tab1, tab2 = st.tabs([
        f"🟢  Starter → Gold  ({upg_sum['starter_to_gold']['count']} candidates)",
        f"🟣  Gold → Platinum  ({upg_sum['gold_to_plat']['count']} candidates)"
    ])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.markdown(kpi_card("Upgrade Candidates", str(upg_sum['starter_to_gold']['count']),
                             color=C_START), unsafe_allow_html=True)
        c2.markdown(kpi_card("Annual Revenue Uplift",
                             fmt_usd(upg_sum['starter_to_gold']['total_annual']),
                             color=C_GREEN), unsafe_allow_html=True)
        c3.markdown(kpi_card("Avg Uplift per Client",
                             fmt_usd(upg_sum['starter_to_gold']['avg_per_client']),
                             color=C_GOLD), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if isinstance(upg_gold, pd.DataFrame) and len(upg_gold):
            st.dataframe(
                upg_gold.style.format({
                    "Avg Monthly Spend ($)": "${:,.0f}",
                    "Qtr Uplift ($)":        "${:,.0f}",
                    "Annual Uplift ($)":     "${:,.0f}",
                }).background_gradient(subset=["Annual Uplift ($)"], cmap="Greens"),
                use_container_width=True, height=300
            )

    with tab2:
        c1, c2, c3 = st.columns(3)
        c1.markdown(kpi_card("Upgrade Candidates", str(upg_sum['gold_to_plat']['count']),
                             color=C_PLAT), unsafe_allow_html=True)
        c2.markdown(kpi_card("Annual Revenue Uplift",
                             fmt_usd(upg_sum['gold_to_plat']['total_annual']),
                             color=C_GREEN), unsafe_allow_html=True)
        c3.markdown(kpi_card("Avg Uplift per Client",
                             fmt_usd(upg_sum['gold_to_plat']['avg_per_client']),
                             color=C_GOLD), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if isinstance(upg_plat, pd.DataFrame) and len(upg_plat):
            st.dataframe(
                upg_plat.style.format({
                    "Avg Monthly Spend ($)": "${:,.0f}",
                    "Qtr Uplift ($)":        "${:,.0f}",
                    "Annual Uplift ($)":     "${:,.0f}",
                }).background_gradient(subset=["Annual Uplift ($)"], cmap="Purples"),
                use_container_width=True, height=300
            )

    st.markdown("---")

    # ── 3.3: Geography ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">3.3 · Revenue by Region</div>',
                unsafe_allow_html=True)

    col_chart, col_table = st.columns([1.5, 1])
    with col_chart:
        fig = px.bar(
            geo_rev.sort_values("total_spend", ascending=True),
            x="total_spend", y="region", orientation="h",
            color="interchange", color_continuous_scale=["#1e293b", C_PLAT],
            labels={"total_spend": "Total Spend ($)", "interchange": "Interchange ($)"},
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(dark_fig(fig, height=280), use_container_width=True)

    with col_table:
        geo_display = geo_rev.copy()
        geo_display["Interchange Yield"] = (geo_display["interchange"] /
                                             geo_display["total_spend"]).round(4)
        geo_display = geo_display.sort_values("interchange", ascending=False)
        geo_display.columns = ["Region","Total Spend","Interchange","Txn Count","Yield"]
        st.dataframe(
            geo_display[["Region","Interchange","Yield"]].style.format({
                "Interchange": "${:,.0f}", "Yield": "{:.4f}"
            }),
            use_container_width=True, height=260
        )

    st.markdown("---")

    # ── 3.4: LTV ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">3.4 · 3-Year LTV by Segment — Who Is Actually Worth The Most</div>',
                unsafe_allow_html=True)

    if isinstance(ltv_seg, pd.DataFrame) and len(ltv_seg):
        fig = px.bar(
            ltv_seg.sort_values("avg_ltv", ascending=True),
            x="avg_ltv", y="cluster_name", color="card_type",
            orientation="h", color_discrete_map=CARD_COLORS,
            labels={"avg_ltv": "Avg 3-Year LTV ($)", "cluster_name": ""},
            barmode="group",
        )
        st.plotly_chart(dark_fig(fig, height=300), use_container_width=True)
    st.caption("LTV = 3 years of interchange + fees + expected upgrade value. "
               "High-spend Starter clients often exceed low-spend Platinum clients in 3-year value.")


# ══════════════════════════════════════════════════════════════════════════════
# CH 4 — RISK & CHURN
# ══════════════════════════════════════════════════════════════════════════════
elif chapter == "4 · Risk & Churn":
    st.markdown("## What Are We at Risk of Losing")
    st.markdown(f"<div style='color:{C_MUTED};font-size:13px;margin-bottom:24px'>"
                f"Churn risk scores, early warning signals, dormancy, and the Platinum watch list</div>",
                unsafe_allow_html=True)

    # ── 4.1: Churn risk distribution ─────────────────────────────────────────
    st.markdown('<div class="section-header">4.1 · Churn Risk Score Distribution by Card Type</div>',
                unsafe_allow_html=True)

    col_chart, col_kpis = st.columns([1.8, 1])
    with col_chart:
        fig = px.histogram(
            active_feat, x="churn_score", color="card_type",
            color_discrete_map=CARD_COLORS, nbins=30, barmode="overlay",
            opacity=0.75,
            labels={"churn_score": "Churn Risk Score (0=Safe, 100=High Risk)",
                    "card_type": "Card"},
        )
        fig.add_vline(x=66, line_dash="dash", line_color=C_RED,
                      annotation_text="High Risk threshold",
                      annotation_font_color=C_RED)
        fig.add_vline(x=33, line_dash="dash", line_color=C_AMBER,
                      annotation_text="Medium threshold",
                      annotation_font_color=C_AMBER)
        st.plotly_chart(dark_fig(fig, height=320), use_container_width=True)

    with col_kpis:
        for band, color in [("High", C_RED), ("Medium", C_AMBER), ("Low", C_GREEN)]:
            band_data = churn_band[churn_band.churn_risk_band == band]
            total_count = band_data["count"].sum() if len(band_data) else 0
            total_rev   = band_data["rev_at_risk"].sum() if len(band_data) else 0
            st.markdown(f"""
            <div style="background:#1e293b;border-left:3px solid {color};
                        border-radius:0 8px 8px 0;padding:12px 14px;margin-bottom:8px">
                <div style="font-size:12px;font-weight:700;color:{color}">{band} Risk</div>
                <div style="font-size:20px;font-weight:700;color:{C_TEXT}">{total_count} cards</div>
                <div style="font-size:12px;color:{C_MUTED}">
                    Rev at risk: <b style="color:{color}">{fmt_usd(total_rev)}/qtr</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 4.2: Early warning signals ───────────────────────────────────────────
    st.markdown('<div class="section-header">4.2 · Early Warning Signals — Patterns That Precede Churn</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    signals = [
        ("📉 Spend Collapse",  "spend_collapse", "Spend dropped 30%+ over 3 months",   C_RED),
        ("🕐 Recency Gap",     "recency",        "No transaction in 45+ days",          C_AMBER),
        ("📊 Low Utilization", "low_util",       "Card utilization below 15%",          C_PLAT),
    ]
    for col, (title, key, desc, color) in zip([c1, c2, c3], signals):
        w = warn_sum.get(key, {})
        cnt = w.get("count", 0)
        rev = w.get("rev", 0)
        with col:
            st.markdown(f"""
            <div style="background:#1e293b;border-radius:10px;padding:16px;text-align:center">
                <div style="font-size:14px;font-weight:600;color:{color};margin-bottom:8px">{title}</div>
                <div style="font-size:28px;font-weight:700;color:{C_TEXT}">{cnt}</div>
                <div style="font-size:11px;color:{C_MUTED};margin-bottom:8px">active cards flagged</div>
                <div style="font-size:12px;font-weight:600;color:{C_AMBER}">{fmt_usd(rev)}/qtr at risk</div>
                <div style="font-size:11px;color:{C_MUTED};margin-top:4px">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 4.3: Dormancy ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">4.3 · Dormant Account Opportunity — Found Money</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Dormant Accounts",
                          str(dorm_sum["total_accounts"]),
                          sub="60+ days inactive, not cancelled"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Avg Prior Monthly Spend",
                          fmt_usd(dorm_sum["avg_prior_spend"]),
                          sub="before going dormant"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Recoverable Revenue",
                          fmt_usd(dorm_sum["recoverable_rev"]),
                          color=C_GREEN,
                          sub="at 50% spend recovery"), unsafe_allow_html=True)
    recovery_roi = dorm_sum["recoverable_rev"] / 20000 if dorm_sum["recoverable_rev"] else 0
    c4.markdown(kpi_card("Reactivation ROI",
                          f"{recovery_roi:.0f}x",
                          color=C_GOLD,
                          sub="vs $20K cashback offer cost"), unsafe_allow_html=True)

    # Dormant by card type
    if dorm_sum["by_card"]:
        st.markdown("<br>", unsafe_allow_html=True)
        dorm_df = pd.DataFrame(list(dorm_sum["by_card"].items()),
                               columns=["Card Type","Dormant Accounts"])
        fig = px.bar(dorm_df, x="Card Type", y="Dormant Accounts",
                     color="Card Type", color_discrete_map=CARD_COLORS)
        fig.update_layout(showlegend=False)
        st.plotly_chart(dark_fig(fig, height=240), use_container_width=True)

    st.markdown("---")

    # ── 4.4: Platinum watch list ──────────────────────────────────────────────
    st.markdown('<div class="section-header">4.4 · Platinum Watch List — Top At-Risk Accounts by Revenue</div>',
                unsafe_allow_html=True)

    if isinstance(plat_watch, pd.DataFrame) and len(plat_watch):
        total_at_risk = plat_watch["Qtr Revenue ($)"].sum()
        st.markdown(f"""
        <div style="background:#1e293b;border:1px solid {C_RED}33;border-radius:8px;
                    padding:12px 16px;margin-bottom:12px;display:flex;
                    justify-content:space-between;align-items:center">
            <div>
                <div style="font-size:12px;color:{C_MUTED}">Combined quarterly revenue at risk</div>
                <div style="font-size:22px;font-weight:700;color:{C_RED}">{fmt_usd(total_at_risk)}</div>
            </div>
            <div style="font-size:12px;color:{C_MUTED};text-align:right">
                {len(plat_watch)} Platinum accounts<br>with churn score ≥ 60
            </div>
        </div>
        """, unsafe_allow_html=True)

        styled = plat_watch.copy()
        styled["Qtr Revenue ($)"] = styled["Qtr Revenue ($)"].apply(lambda x: f"${x:,.0f}")
        styled["Churn Score"]     = styled["Churn Score"].apply(
            lambda x: f"🔴 {x:.0f}" if x >= 75 else f"🟡 {x:.0f}")
        styled["Days Inactive"]   = styled["Days Inactive"].apply(
            lambda x: f"⚠️ {x:.0f}" if x > 45 else f"{x:.0f}")
        st.dataframe(styled, use_container_width=True, height=380)
    else:
        st.info("No Platinum accounts currently meet the high-risk threshold. Portfolio is healthy.")


# ══════════════════════════════════════════════════════════════════════════════
# CH 5 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif chapter == "5 · Recommendations":
    st.markdown("## What We Should Do — And Why")
    st.markdown(f"<div style='color:{C_MUTED};font-size:13px;margin-bottom:24px'>"
                f"6 evidence-backed recommendations with full dollar impact, cost, and ROI</div>",
                unsafe_allow_html=True)

    # Summary bar
    total_rev  = sum(r["total_annual_rev"] for r in recs)
    total_cost = sum(r["cost_usd"] for r in recs)
    total_roi  = total_rev / total_cost if total_cost else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Total Annual Uplift", fmt_usd(total_rev), color=C_GREEN),
                unsafe_allow_html=True)
    c2.markdown(kpi_card("Total Investment", fmt_usd(total_cost), color=C_AMBER),
                unsafe_allow_html=True)
    c3.markdown(kpi_card("Portfolio ROI", f"{total_roi:.0f}x", color=C_PLAT),
                unsafe_allow_html=True)
    c4.markdown(kpi_card("Actions to Execute", "6", sub="across 3 card types"),
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Recommendation cards
    for r in recs:
        color = r["color"]
        q_vals = r["rev_q"]
        with st.expander(f"  {r['id']} · {r['title']}", expanded=False):
            col_detail, col_numbers = st.columns([1.6, 1])
            with col_detail:
                st.markdown(f"""
                <div style="background:#1e293b;border-radius:8px;padding:14px">
                    <div style="font-size:11px;font-weight:700;color:{color};
                                text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px">
                        {r['card']} · {r['id']}
                    </div>
                    <div style="font-size:14px;font-weight:600;color:{C_TEXT};margin-bottom:10px">
                        {r['title']}
                    </div>
                    <div style="font-size:12px;color:{C_MUTED};margin-bottom:6px">
                        <b style="color:{C_TEXT}">Action:</b> {r['action']}
                    </div>
                    <div style="font-size:12px;color:{C_MUTED}">
                        <b style="color:{C_TEXT}">Evidence:</b> {r['evidence']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_numbers:
                quarters = ["Q3 2024","Q4 2024","Q1 2025","Q2 2025"]
                fig = go.Figure(go.Bar(
                    x=quarters, y=q_vals,
                    marker_color=color, opacity=0.85,
                    text=[fmt_usd(v) for v in q_vals],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>"
                ))
                fig.update_layout(yaxis_title="Revenue ($)", showlegend=False)
                st.plotly_chart(dark_fig(fig, height=220), use_container_width=True)

                st.markdown(f"""
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px">
                    <div style="background:#1e293b;border-radius:6px;padding:8px;text-align:center">
                        <div style="font-size:10px;color:{C_MUTED}">Annual Rev</div>
                        <div style="font-size:15px;font-weight:700;color:{C_GREEN}">
                            {fmt_usd(r['total_annual_rev'])}</div>
                    </div>
                    <div style="background:#1e293b;border-radius:6px;padding:8px;text-align:center">
                        <div style="font-size:10px;color:{C_MUTED}">Investment</div>
                        <div style="font-size:15px;font-weight:700;color:{C_AMBER}">
                            {fmt_usd(r['cost_usd'])}</div>
                    </div>
                    <div style="background:#1e293b;border-radius:6px;padding:8px;text-align:center">
                        <div style="font-size:10px;color:{C_MUTED}">ROI</div>
                        <div style="font-size:15px;font-weight:700;color:{C_PLAT}">
                            {r['roi']:.0f}x</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    # Full impact table
    st.markdown('<div class="section-header">Full Revenue Impact Summary</div>',
                unsafe_allow_html=True)

    display_tbl = impact_tbl.copy()
    money_cols  = [c for c in display_tbl.columns
                   if c not in ["Action","Card","ROI"]]
    st.dataframe(
        display_tbl.style
            .format({c: "${:,.0f}" for c in money_cols if c != "ROI"})
            .format({"ROI": "{:.1f}x"})
            .apply(lambda x: ["background-color:#1e293b;font-weight:700"
                               if x.name == len(display_tbl)-1 else "" for _ in x], axis=1),
        use_container_width=True, height=300
    )


# ══════════════════════════════════════════════════════════════════════════════
# CH 6 — REVENUE ROADMAP
# ══════════════════════════════════════════════════════════════════════════════
elif chapter == "6 · Revenue Roadmap":
    st.markdown("## What This Is Worth — Quarter by Quarter")
    st.markdown(f"<div style='color:{C_MUTED};font-size:13px;margin-bottom:24px'>"
                f"12-month revenue forecast, target achievement status, and scenario model</div>",
                unsafe_allow_html=True)

    # ── Target achievement ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Target Achievement Forecast — End of Q2 2025</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, (ct, tc) in zip([c1, c2, c3], target_check.items()):
        color    = CARD_COLORS[ct]
        achieved = tc["achieved"]
        status   = "✅ On Track" if achieved else "⚠️ Gap Remains"
        s_color  = C_GREEN if achieved else C_AMBER
        pct      = tc["forecast"] / tc["target"] * 100 if tc["target"] else 100
        with col:
            st.markdown(f"""
            <div style="background:#1e293b;border-top:3px solid {color};
                        border-radius:0 0 10px 10px;padding:16px;text-align:center">
                <div style="font-size:12px;font-weight:600;color:{color};margin-bottom:10px">{ct}</div>
                <div style="font-size:11px;color:{C_MUTED}">Baseline</div>
                <div style="font-size:16px;font-weight:600;color:{C_TEXT}">{fmt_usd(tc['base'])}/qtr</div>
                <div style="margin:8px 0;font-size:20px">→</div>
                <div style="font-size:11px;color:{C_MUTED}">Target</div>
                <div style="font-size:16px;font-weight:600;color:{C_AMBER}">{fmt_usd(tc['target'])}/qtr</div>
                <div style="margin:8px 0;font-size:20px">→</div>
                <div style="font-size:11px;color:{C_MUTED}">Forecast</div>
                <div style="font-size:20px;font-weight:700;color:{C_GREEN}">{fmt_usd(tc['forecast'])}/qtr</div>
                <div style="margin-top:10px;font-size:12px;font-weight:700;color:{s_color}">{status}</div>
                <div style="font-size:11px;color:{C_MUTED}">{pct:.0f}% of target achieved</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Forecast chart ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Quarterly Revenue Forecast — Baseline vs With Recommendations</div>',
                unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Baseline (current run-rate)",
        x=forecast["quarter"], y=forecast["baseline_rev"],
        marker_color="#334155", opacity=0.9,
        hovertemplate="<b>%{x}</b><br>Baseline: $%{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        name="Incremental from recommendations",
        x=forecast["quarter"], y=forecast["incremental_rev"],
        marker_color=C_GREEN, opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Incremental: $%{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        name="Total Revenue",
        x=forecast["quarter"], y=forecast["total_rev"],
        mode="lines+markers+text",
        line=dict(color=C_PLAT, width=2),
        marker=dict(size=8),
        text=[fmt_usd(v) for v in forecast["total_rev"]],
        textposition="top center",
        hovertemplate="<b>%{x}</b><br>Total: $%{y:,.0f}<extra></extra>"
    ))
    fig.update_layout(barmode="stack", xaxis_title="", yaxis_title="Revenue ($)")
    st.plotly_chart(dark_fig(fig, height=380), use_container_width=True)

    st.markdown("---")

    # ── Cumulative uplift ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Cumulative Incremental Revenue — 12 Months</div>',
                unsafe_allow_html=True)

    col_chart, col_summary = st.columns([1.5, 1])
    with col_chart:
        fig2 = go.Figure(go.Scatter(
            x=forecast["quarter"],
            y=forecast["cumulative_uplift"],
            mode="lines+markers+text",
            fill="tozeroy",
            fillcolor=f"{C_GREEN}22",
            line=dict(color=C_GREEN, width=3),
            marker=dict(size=10, color=C_GREEN),
            text=[fmt_usd(v) for v in forecast["cumulative_uplift"]],
            textposition="top center",
            hovertemplate="<b>%{x}</b><br>Cumulative uplift: $%{y:,.0f}<extra></extra>"
        ))
        fig2.update_layout(yaxis_title="Cumulative Incremental Revenue ($)")
        st.plotly_chart(dark_fig(fig2, height=300), use_container_width=True)

    with col_summary:
        st.markdown("<br><br>", unsafe_allow_html=True)
        annual_uplift = forecast["cumulative_uplift"].iloc[-1]
        total_invest  = sum(r["cost_usd"] for r in recs)
        st.markdown(f"""
        <div style="background:#1e293b;border-radius:10px;padding:20px">
            <div style="font-size:11px;color:{C_MUTED};margin-bottom:4px">
                12-month incremental revenue</div>
            <div style="font-size:32px;font-weight:700;color:{C_GREEN}">
                {fmt_usd(annual_uplift)}</div>

            <div style="height:1px;background:#334155;margin:14px 0"></div>

            <div style="font-size:11px;color:{C_MUTED};margin-bottom:4px">
                Total cost of all 6 actions</div>
            <div style="font-size:20px;font-weight:600;color:{C_AMBER}">
                {fmt_usd(total_invest)}</div>

            <div style="height:1px;background:#334155;margin:14px 0"></div>

            <div style="font-size:11px;color:{C_MUTED};margin-bottom:4px">Portfolio ROI</div>
            <div style="font-size:28px;font-weight:700;color:{C_PLAT}">
                {annual_uplift/total_invest:.0f}x</div>

            <div style="height:1px;background:#334155;margin:14px 0"></div>

            <div style="font-size:11px;color:{C_MUTED};margin-bottom:4px">
                First revenue appears</div>
            <div style="font-size:16px;font-weight:600;color:{C_TEXT}">
                Q3 2024 (R1 + R5 immediate)</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Closing CFO statement ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1e293b,#0f172a);
                border:1px solid #334155;border-radius:12px;padding:24px;margin-top:8px">
        <div style="font-size:11px;color:{C_MUTED};text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:8px">Executive Summary</div>
        <div style="font-size:15px;color:{C_TEXT};line-height:1.7">
            Executing all 6 recommendations requires a total investment of
            <b style="color:{C_AMBER}">{fmt_usd(total_invest)}</b> and delivers
            <b style="color:{C_GREEN}">{fmt_usd(annual_uplift)}</b> in incremental annual revenue —
            a <b style="color:{C_PLAT}">{annual_uplift/total_invest:.0f}x ROI</b>.
            Revenue begins in Q3 2024 with R1 (Platinum retention) and R5 (Starter reactivation),
            scaling to full run-rate by Q2 2025. All three card targets —
            <b style="color:{C_PLAT}">Platinum +10%</b>,
            <b style="color:{C_GOLD}">Gold +25%</b>, and
            <b style="color:{C_START}">Starter +45%</b> — are achievable within 12 months.
        </div>
    </div>
    """, unsafe_allow_html=True)
