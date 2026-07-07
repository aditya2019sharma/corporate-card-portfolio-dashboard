# 💳 Corporate Card Portfolio Dashboard

An interactive analytics dashboard for a corporate card portfolio spanning three card
tiers (Platinum Elite, Business Gold, Corporate Starter). Built with **Streamlit** and
**Plotly**, it walks through six chapters of analysis: portfolio health, customer
segmentation, churn/risk, evidence-backed revenue recommendations, and a 12-month
revenue forecast.

**[🔗 Live demo](#)** — replace this with your Streamlit Cloud URL once deployed

> **Note:** This repo ships with a synthetic, fabricated dataset
> (`generate_demo_data.py` → `analysis_results.pkl`) so the dashboard can be shared
> publicly. No real client or transaction data is included.

## What it does

- **1 · Executive Briefing** — top-line KPIs, revenue trend by card, gap-to-target tracking
- **2 · Portfolio Health** — fee vs. interchange composition, spend-by-category yield heatmap, monthly spend trend
- **3 · Customer Intelligence** — 5-segment clustering, upgrade candidate pipeline, revenue by region, 3-year LTV by segment
- **4 · Risk & Churn** — churn score distribution, early warning signals, dormant account recovery opportunity, Platinum watch list
- **5 · Recommendations** — 6 evidence-backed actions with dollar impact, cost, and ROI per action
- **6 · Revenue Roadmap** — quarter-by-quarter forecast, target achievement tracking, cumulative uplift model

## Tech stack

- [Streamlit](https://streamlit.io/) — app framework
- [Plotly](https://plotly.com/python/) — interactive charts
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling

## Running locally

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
python generate_demo_data.py   # builds analysis_results.pkl from synthetic data
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deploying

This app is deployed for free on [Streamlit Community Cloud](https://share.streamlit.io/).
To deploy your own copy:

1. Fork or push this repo to your own GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io/), sign in with GitHub
3. Click **New app**, point it at this repo, and set the main file to `app.py`
4. Make sure `analysis_results.pkl` is committed to the repo (or add a build step that
   runs `generate_demo_data.py` first)

## Project structure

```
.
├── app.py                  # Streamlit dashboard
├── generate_demo_data.py   # Builds the synthetic analysis_results.pkl
├── analysis_results.pkl    # Pre-computed synthetic data (generated)
├── requirements.txt
└── README.md
```
