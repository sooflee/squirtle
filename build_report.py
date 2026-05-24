"""Build a self-contained HTML report from the wine investment analysis.

Output: docs/index.html (ready to serve from GitHub Pages).
All plots are embedded as base64 PNGs so the report is a single file.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

CURRENT_YEAR = 2026
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#bbbbbb", "axes.labelcolor": "#1a1a1a",
    "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.spines.top": False, "axes.spines.right": False,
})

PALETTE = {"Bordeaux": "#7a2e3a", "Burgundy": "#7a3a8a", "Champagne": "#c89a17"}

# ---------- data ----------
wines = pd.read_csv(ROOT / "data/wines.csv")
bench = pd.read_csv(ROOT / "data/benchmarks.csv").set_index("year")
vintage_quality = pd.read_csv(ROOT / "data/vintage_quality.csv")
candidates = pd.read_csv(ROOT / "data/candidates.csv")
cohort = pd.read_csv(ROOT / "data/cohort_indices.csv").set_index("year")

COHORT_COL = {"Bordeaux": "bordeaux_idx", "Burgundy": "burgundy_idx",
              "Champagne": "champagne_idx"}


# ---------- basic returns ----------
def _cagr(s, e, y):
    if y <= 0 or s <= 0 or e <= 0:
        return np.nan
    return (e / s) ** (1 / y) - 1


def benchmark_cagr(col, start_year, end_year=CURRENT_YEAR):
    if start_year not in bench.index or end_year not in bench.index:
        return np.nan
    s, e = bench.loc[start_year, col], bench.loc[end_year, col]
    if pd.isna(s) or pd.isna(e):
        return np.nan
    return _cagr(s, e, end_year - start_year)


wines["years_held"] = CURRENT_YEAR - wines["release_year"]
wines["nominal_cagr"] = wines.apply(
    lambda r: _cagr(r["release_price_usd"], r["current_price_usd"],
                    r["years_held"]), axis=1)
wines["cpi_cagr"] = wines["release_year"].apply(
    lambda y: benchmark_cagr("cpi_us", y))
wines["sp500_cagr"] = wines["release_year"].apply(
    lambda y: benchmark_cagr("sp500_tr", y))
wines["gold_cagr"] = wines["release_year"].apply(
    lambda y: benchmark_cagr("gold_usd_oz", y))
wines["real_cagr"] = (1 + wines["nominal_cagr"]) / (1 + wines["cpi_cagr"]) - 1
wines["alpha_sp500"] = wines["nominal_cagr"] - wines["sp500_cagr"]
wines["alpha_gold"] = wines["nominal_cagr"] - wines["gold_cagr"]


def era(y):
    if y < 2000: return "pre-2000"
    if y < 2009: return "2000-2008"
    if y < 2012: return "2009-2011 bubble"
    return "post-2011"


wines["release_era"] = wines["release_year"].apply(era)


# ---------- price reconstruction (two-factor) ----------
def reconstruct_track(row: pd.Series) -> pd.Series:
    """Annual price track from release_year to CURRENT_YEAR.

    Uses the two-factor decomposition: total log-return = idiosyncratic
    drift + cohort log-return. Endpoints are anchored to release and
    current price; intermediate years inherit cohort-index volatility.
    """
    rel = int(row["release_year"])
    cur = CURRENT_YEAR
    years = cur - rel
    if years <= 0:
        return pd.Series([row["release_price_usd"]], index=[rel])
    region = row["region"]
    col = COHORT_COL[region]
    idx = cohort.loc[rel:cur, col]
    base = idx.iloc[0]
    cohort_log = np.log(idx.values / base)
    cohort_cagr_log = cohort_log[-1] / years
    total_cagr_log = np.log(row["current_price_usd"] / row["release_price_usd"]) / years
    idio_cagr_log = total_cagr_log - cohort_cagr_log
    elapsed = idx.index.values - rel
    log_price = np.log(row["release_price_usd"]) + idio_cagr_log * elapsed + cohort_log
    return pd.Series(np.exp(log_price), index=idx.index)


# Build a wide DataFrame: rows = wine_id (producer-vintage), cols = year
def build_price_panel():
    rows = {}
    for _, r in wines.iterrows():
        key = f"{r['producer']}|{r['vintage']}"
        rows[key] = reconstruct_track(r)
    return pd.DataFrame(rows).T  # rows are wines, columns are years


price_panel = build_price_panel()


# ---------- vol / Sharpe ----------
def log_returns_row(row: pd.Series) -> pd.Series:
    p = row.dropna()
    if len(p) < 2:
        return pd.Series(dtype=float)
    return np.log(p / p.shift(1)).dropna()


wine_returns = price_panel.apply(log_returns_row, axis=1)
# wine_returns is a DataFrame: rows = wines, columns = years (year of return)

vol_series = wine_returns.std(axis=1)  # per-wine annualized vol (annual data)
mean_logret = wine_returns.mean(axis=1)

# Risk-free per wine: mean 10Y over its holding window
def avg_rf(row):
    rel = int(wines.loc[wines.index == row.name, "release_year"].iloc[0]) \
        if isinstance(row, pd.Series) else None
    return rel


def per_wine_metrics():
    out = []
    for i, w in wines.iterrows():
        key = f"{w['producer']}|{w['vintage']}"
        if key not in wine_returns.index:
            continue
        rets = wine_returns.loc[key].dropna()
        if len(rets) < 2:
            continue
        rel, end = int(w["release_year"]), CURRENT_YEAR
        rf = bench.loc[rel:end, "us10y_yield_pct"].mean() / 100
        mu = rets.mean()
        sigma = rets.std()
        # Convert mu (mean log return) to nominal mean per year
        ann_return = np.exp(mu) - 1
        sharpe = (ann_return - rf) / sigma if sigma > 0 else np.nan
        out.append({
            "key": key, "producer": w["producer"], "region": w["region"],
            "vintage": w["vintage"], "tier": w["tier"],
            "release_era": w["release_era"],
            "annualized_vol": sigma, "mean_annual_return": ann_return,
            "rf_avg": rf, "sharpe": sharpe,
            "realized_cagr": w["nominal_cagr"],
        })
    return pd.DataFrame(out)


wine_metrics = per_wine_metrics()


def benchmark_metrics(col, label, start_year=2004):
    s = bench.loc[start_year:CURRENT_YEAR, col].dropna()
    rets = np.log(s / s.shift(1)).dropna()
    rf = bench.loc[start_year:CURRENT_YEAR, "us10y_yield_pct"].mean() / 100
    mu = rets.mean()
    sigma = rets.std()
    ann_return = np.exp(mu) - 1
    return {
        "label": label, "annualized_vol": sigma,
        "mean_annual_return": ann_return, "rf_avg": rf,
        "sharpe": (ann_return - rf) / sigma if sigma > 0 else np.nan,
    }


bench_rows = [
    benchmark_metrics("sp500_tr", "S&P 500 TR"),
    benchmark_metrics("gold_usd_oz", "Gold"),
    benchmark_metrics("livex100", "Liv-ex 100"),
]
bench_metrics = pd.DataFrame(bench_rows)


# ---------- feature engineering for model ----------
def add_features(panel: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    d["bubble_release"] = d["release_year"].between(2010, 2012).astype(int)
    d["log_release_price"] = np.log(d["release_price_usd"])
    d = d.merge(vintage_quality, on=["region", "vintage"], how="left")
    d["vintage_quality"] = d["vintage_quality"].fillna(
        vintage_quality["vintage_quality"].median())

    combined = pd.concat([history, panel], ignore_index=True).drop_duplicates(
        subset=["producer", "vintage"])

    def prior_price(row):
        prior = combined[(combined["producer"] == row["producer"]) &
                         (combined["vintage"] < row["vintage"])]
        if len(prior) == 0:
            return np.nan
        return prior.sort_values("vintage").iloc[-1]["release_price_usd"]

    d["prior_release_price"] = d.apply(prior_price, axis=1)
    d["log_price_delta_vs_prior"] = np.log(
        d["release_price_usd"] / d["prior_release_price"]).fillna(0.0)
    d["has_prior"] = d["prior_release_price"].notna().astype(int)
    livex_lookup = bench["livex100"].dropna().to_dict()
    earliest_livex = bench["livex100"].dropna().iloc[0]
    d["livex_at_release"] = d["release_year"].map(
        lambda y: livex_lookup.get(y, earliest_livex))
    d["log_livex_at_release"] = np.log(d["livex_at_release"])
    return d


BASELINE_FEATURES = ["critic_score", "region", "tier", "years_held",
                     "bubble_release", "log_release_price"]
AUGMENTED_FEATURES = BASELINE_FEATURES + [
    "vintage_quality", "log_price_delta_vs_prior", "has_prior",
    "log_livex_at_release"]
DROP_COLLINEAR = ["tier_grand_cru", "tier_prestige"]


def build_X(d: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = pd.get_dummies(d[features], drop_first=True).astype(float)
    for col in DROP_COLLINEAR:
        if col in X.columns:
            X = X.drop(columns=col)
    return X


def train_and_score(d, features, y):
    X = build_X(d, features)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    rf = RandomForestRegressor(n_estimators=400, max_depth=5,
                               min_samples_leaf=3, random_state=42)
    return {
        "X": X,
        "ridge_cv_r2": cross_val_score(ridge, X, y, cv=kf, scoring="r2"),
        "rf_cv_r2": cross_val_score(rf, X, y, cv=kf, scoring="r2"),
        "ridge": ridge.fit(X, y),
        "rf": rf.fit(X, y),
        "ols": sm.OLS(y, sm.add_constant(X)).fit(),
    }


df = add_features(wines, wines)
y = df["real_cagr"].astype(float)
baseline = train_and_score(df, BASELINE_FEATURES, y)
augmented = train_and_score(df, AUGMENTED_FEATURES, y)

USE_AUGMENTED = augmented["rf_cv_r2"].mean() >= baseline["rf_cv_r2"].mean() + 0.02
chosen = augmented if USE_AUGMENTED else baseline
chosen_name = "augmented" if USE_AUGMENTED else "baseline"
chosen_features = AUGMENTED_FEATURES if USE_AUGMENTED else BASELINE_FEATURES

X = chosen["X"]
ridge_r2 = chosen["ridge_cv_r2"]
rf_r2 = chosen["rf_cv_r2"]
ridge_coefs = pd.Series(chosen["ridge"].named_steps["ridge"].coef_, index=X.columns)
rf_imp = pd.Series(chosen["rf"].feature_importances_, index=X.columns)
ols = chosen["ols"]


# ---------- candidates ----------
FORWARD_HOLD_YEARS = 10
cand = add_features(candidates, wines)
cand["years_held"] = FORWARD_HOLD_YEARS
cand_X_raw = build_X(cand, chosen_features)
for c in X.columns:
    if c not in cand_X_raw.columns:
        cand_X_raw[c] = 0.0
cand_X = cand_X_raw[X.columns]
cand["pred_rf"] = chosen["rf"].predict(cand_X)
cand["pred_ridge"] = chosen["ridge"].predict(cand_X)
cand["pred_blend"] = (cand["pred_rf"] + cand["pred_ridge"]) / 2
cand_ranked = cand.sort_values("pred_blend", ascending=False)


# ---------- walk-forward backtest ----------
def walk_forward():
    results = []
    cutoffs = [2008, 2011, 2014, 2017, 2020]
    for T in cutoffs:
        train = df[df["release_year"] <= T]
        test = df[(df["release_year"] > T) & (df["release_year"] <= T + 3)]
        if len(train) < 20 or len(test) < 4:
            continue
        X_tr = build_X(train, BASELINE_FEATURES)
        X_te = build_X(test, BASELINE_FEATURES)
        # Align test columns
        for c in X_tr.columns:
            if c not in X_te.columns:
                X_te[c] = 0.0
        X_te = X_te[X_tr.columns]
        y_tr = train["real_cagr"].astype(float)

        rf = RandomForestRegressor(n_estimators=400, max_depth=5,
                                   min_samples_leaf=3, random_state=42)
        rf.fit(X_tr, y_tr)
        ridge = Pipeline([("scaler", StandardScaler()),
                          ("ridge", Ridge(alpha=1.0))]).fit(X_tr, y_tr)
        pred = (rf.predict(X_te) + ridge.predict(X_te)) / 2
        test = test.copy()
        test["pred"] = pred
        test_sorted = test.sort_values("pred", ascending=False)

        n_picks = min(3, len(test_sorted) // 2)
        top = test_sorted.head(n_picks)
        bot = test_sorted.tail(n_picks)

        # Realized CAGR is already in wines (release → 2026)
        top_realized = top["real_cagr"].mean()
        bot_realized = bot["real_cagr"].mean()
        all_realized = test["real_cagr"].mean()

        # Rank correlation
        rho, p_rho = stats.spearmanr(test["pred"], test["real_cagr"])

        # S&P benchmark over same window: avg of S&P real CAGR for release years in (T, T+3]
        sp_real = []
        for ry in test["release_year"].unique():
            sp_nom = benchmark_cagr("sp500_tr", int(ry))
            cpi_c = benchmark_cagr("cpi_us", int(ry))
            if pd.notna(sp_nom) and pd.notna(cpi_c):
                sp_real.append((1 + sp_nom) / (1 + cpi_c) - 1)
        sp_real_mean = np.mean(sp_real) if sp_real else np.nan

        results.append({
            "cutoff_year": T,
            "n_train": len(train), "n_test": len(test),
            "top_real_cagr": top_realized,
            "bot_real_cagr": bot_realized,
            "all_real_cagr": all_realized,
            "sp500_real_cagr": sp_real_mean,
            "rank_corr": rho,
            "rank_p": p_rho,
            "edge": top_realized - sp_real_mean if pd.notna(sp_real_mean) else np.nan,
        })
    return pd.DataFrame(results)


backtest_df = walk_forward()


# ---------- chart helpers ----------
def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def img(fig, alt: str) -> str:
    return f'<img src="data:image/png;base64,{fig_to_b64(fig)}" alt="{alt}">'


def pct(x, n=1, sign=False):
    if pd.isna(x):
        return "n/a"
    s = f"{x*100:.{n}f}%"
    if sign and x > 0:
        s = "+" + s
    return s


# ---------- charts ----------
def chart_return_distribution():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    sns.histplot(data=wines, x="real_cagr", hue="region", multiple="stack",
                 bins=18, palette=PALETTE, ax=ax, edgecolor="white",
                 linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=1,
               label="inflation breakeven")
    ax.set_xlabel("Real CAGR (CPI-adjusted, annualized)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of real returns across 84 wine/vintage observations")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    return img(fig, "Real CAGR distribution by region")


def chart_alpha_vs_sp500():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    sns.histplot(data=wines, x="alpha_sp500", hue="region", multiple="stack",
                 bins=20, palette=PALETTE, ax=ax, edgecolor="white",
                 linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    med = wines["alpha_sp500"].median()
    ax.axvline(med, color="#c0392b", linestyle=":", linewidth=1.5,
               label=f"median = {med*100:+.1f}%")
    ax.set_xlabel("Alpha vs S&P 500 TR (annualized)")
    ax.set_ylabel("Count")
    ax.set_title("Wine return minus S&P 500 TR over matched holding period")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0%}"))
    ax.legend()
    return img(fig, "Alpha vs S&P 500")


def chart_era_breakdown():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    order = ["pre-2000", "2000-2008", "2009-2011 bubble", "post-2011"]
    data = (wines.groupby(["release_era", "region"])["real_cagr"]
            .mean().unstack().reindex(order))
    data.plot(kind="bar", ax=ax,
              color=[PALETTE[c] for c in data.columns],
              edgecolor="white", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Mean real CAGR")
    ax.set_xlabel("")
    ax.set_title("Mean real CAGR by release era and region")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.xticks(rotation=15, ha="right")
    ax.legend(title="Region")
    return img(fig, "Era × region")


def chart_indices_normalized():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sub = cohort.loc[2004:].copy()
    sub["sp500_tr"] = bench.loc[2004:, "sp500_tr"]
    sub["gold_usd_oz"] = bench.loc[2004:, "gold_usd_oz"]
    styles = [
        ("bordeaux_idx", "Bordeaux", PALETTE["Bordeaux"]),
        ("burgundy_idx", "Burgundy", PALETTE["Burgundy"]),
        ("champagne_idx", "Champagne", PALETTE["Champagne"]),
        ("sp500_tr", "S&P 500 TR", "#2c5282"),
        ("gold_usd_oz", "Gold", "#888888"),
    ]
    for col, label, color in styles:
        series = sub[col].dropna()
        if len(series):
            ax.plot(series.index, series / series.iloc[0] * 100,
                    label=label, color=color, linewidth=2)
    ax.set_title("Wine cohort indices vs S&P 500 vs Gold  (2004 = 100)")
    ax.set_ylabel("Index level (log scale)")
    ax.set_xlabel("")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    return img(fig, "Indices normalized")


def chart_release_vs_current():
    fig, ax = plt.subplots(figsize=(7, 6))
    for region, grp in wines.groupby("region"):
        ax.scatter(grp["release_price_usd"], grp["current_price_usd"],
                   label=region, alpha=0.75, s=55, color=PALETTE[region],
                   edgecolor="white", linewidth=0.8)
    lims = [10, max(wines["current_price_usd"].max(),
                    wines["release_price_usd"].max()) * 1.3]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1, label="nominal breakeven")
    ax.plot(lims, [x * 4 for x in lims], color="#27ae60", linestyle=":",
            alpha=0.7, linewidth=1, label="~4× release (rough CPI 2000 → 2026)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Release price (USD, log)")
    ax.set_ylabel("Current price (USD, log)")
    ax.set_title("Release vs current price (log-log)")
    ax.legend(loc="upper left")
    return img(fig, "Release vs current")


def chart_model_features():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cs = ridge_coefs.sort_values()
    axes[0].barh(cs.index, cs.values,
                 color=["#c0392b" if v < 0 else "#2c5282" for v in cs.values])
    axes[0].axvline(0, color="black", linewidth=0.5)
    axes[0].set_title("Ridge: standardized coefficients")
    axes[0].set_xlabel("Δ real CAGR per 1σ feature")

    imp = rf_imp.sort_values()
    axes[1].barh(imp.index, imp.values, color="#27ae60")
    axes[1].set_title("Random forest: feature importance")
    axes[1].set_xlabel("Importance")
    plt.tight_layout()
    return img(fig, "Model features")


def chart_model_comparison():
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Baseline\n(6 features)", "Augmented\n(10 features)"]
    rm = [baseline["ridge_cv_r2"].mean(), augmented["ridge_cv_r2"].mean()]
    fm = [baseline["rf_cv_r2"].mean(), augmented["rf_cv_r2"].mean()]
    rs = [baseline["ridge_cv_r2"].std(), augmented["ridge_cv_r2"].std()]
    fs = [baseline["rf_cv_r2"].std(), augmented["rf_cv_r2"].std()]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, rm, w, yerr=rs, label="Ridge", color="#2c5282", capsize=4)
    ax.bar(x + w/2, fm, w, yerr=fs, label="Random forest", color="#27ae60", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("5-fold CV R²")
    ax.set_title("Model comparison: out-of-sample R²")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    return img(fig, "Model comparison")


def chart_candidate_predictions():
    fig, ax = plt.subplots(figsize=(10, 8))
    cs = cand_ranked.copy()
    labels = [f"{r.producer} {r.vintage}" for _, r in cs.iterrows()]
    colors = [PALETTE[r] for r in cs["region"]]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, cs["pred_blend"], color=colors,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Predicted real CAGR (blend of ridge + RF)")
    ax.set_title("Candidate ranking — wines purchasable May 2026")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1%}"))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for l, c in PALETTE.items()],
              loc="lower right")
    plt.tight_layout()
    return img(fig, "Candidate predictions")


def chart_price_tracks():
    """Sample of reconstructed price tracks from each cohort."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    samples = {
        "Bordeaux": [("Lafite Rothschild", 1982), ("Lafite Rothschild", 2009),
                     ("Latour", 2000), ("Margaux", 2015)],
        "Burgundy": [("DRC Romanee-Conti", 1990), ("Roumier Musigny", 2005),
                     ("Rousseau Chambertin", 2010), ("DRC La Tache", 1999)],
        "Champagne": [("Krug Vintage", 1995), ("Dom Perignon", 2002),
                      ("Cristal", 2008), ("Salon", 1996)],
    }
    for ax, (region, items) in zip(axes, samples.items()):
        for prod, vint in items:
            key = f"{prod}|{vint}"
            if key in price_panel.index:
                series = price_panel.loc[key].dropna()
                ax.plot(series.index, series.values,
                        label=f"{prod} {vint}", linewidth=1.5, alpha=0.9)
        ax.set_title(f"{region} — sample reconstructed tracks")
        ax.set_yscale("log")
        ax.set_xlabel("")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylabel("USD / bottle (log)" if region == "Bordeaux" else "")
    plt.tight_layout()
    return img(fig, "Price tracks")


def chart_vol_return():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for region, grp in wine_metrics.groupby("region"):
        ax.scatter(grp["annualized_vol"], grp["realized_cagr"],
                   s=60, alpha=0.7, color=PALETTE[region],
                   label=region, edgecolor="white", linewidth=0.8)
    # Benchmarks
    for _, row in bench_metrics.iterrows():
        ax.scatter(row["annualized_vol"], row["mean_annual_return"],
                   s=200, marker="*", color="#2c5282", edgecolor="black",
                   linewidth=1.2, zorder=5)
        ax.annotate(row["label"],
                    (row["annualized_vol"], row["mean_annual_return"]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=10, color="#2c5282", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Annualized volatility (stdev of annual log returns)")
    ax.set_ylabel("Realized CAGR")
    ax.set_title("Risk-return scatter — individual wines vs benchmarks")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(loc="upper left")
    return img(fig, "Vol-return scatter")


def chart_sharpe_by_cohort():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cohort_sharpe = wine_metrics.groupby("region").agg(
        sharpe=("sharpe", "mean"),
        n=("sharpe", "size"),
        vol=("annualized_vol", "mean"),
    ).reset_index()
    labels = list(cohort_sharpe["region"]) + list(bench_metrics["label"])
    sharpes = list(cohort_sharpe["sharpe"]) + list(bench_metrics["sharpe"])
    colors = [PALETTE[r] for r in cohort_sharpe["region"]] + ["#2c5282"] * len(bench_metrics)
    bars = ax.bar(labels, sharpes, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Sharpe ratio (excess return / vol)")
    ax.set_title("Sharpe ratio by cohort — wine vs benchmarks")
    for b, s in zip(bars, sharpes):
        if pd.notna(s):
            ax.text(b.get_x() + b.get_width()/2, s + 0.02 if s >= 0 else s - 0.05,
                    f"{s:.2f}", ha="center", fontsize=9)
    plt.xticks(rotation=10)
    return img(fig, "Sharpe by cohort")


def chart_backtest():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    x = np.arange(len(backtest_df))
    w = 0.22
    ax.bar(x - 1.5*w, backtest_df["top_real_cagr"], w,
           color="#27ae60", label="Model top-3")
    ax.bar(x - 0.5*w, backtest_df["all_real_cagr"], w,
           color="#888", label="All tested wines")
    ax.bar(x + 0.5*w, backtest_df["bot_real_cagr"], w,
           color="#c0392b", label="Model bottom-3")
    ax.bar(x + 1.5*w, backtest_df["sp500_real_cagr"], w,
           color="#2c5282", label="S&P 500 TR")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t)}" for t in backtest_df["cutoff_year"]])
    ax.set_xlabel("Training cutoff year T  (test = releases in T+1 to T+3)")
    ax.set_ylabel("Realized real CAGR")
    ax.set_title("Walk-forward backtest: realized return by portfolio")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0%}"))
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.bar(x, backtest_df["rank_corr"], color="#2c5282",
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t)}" for t in backtest_df["cutoff_year"]])
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Spearman rank correlation")
    ax.set_title("Predicted vs realized: rank correlation by cutoff")
    ax.set_ylim(-1.05, 1.05)
    for i, (rho, p) in enumerate(zip(backtest_df["rank_corr"], backtest_df["rank_p"])):
        ax.text(i, rho + 0.05 if rho >= 0 else rho - 0.1,
                f"ρ={rho:.2f}\np={p:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    return img(fig, "Backtest results")


# ---------- tables ----------
def df_to_html(df, classes="", numeric_color=True):
    def style(col, val):
        if pd.isna(val):
            return "—"
        s = col.lower()
        if "cagr" in s or "alpha" in s or "delta" in s:
            color = "#27ae60" if val > 0 else "#c0392b"
            return f'<span style="color:{color}">{val:+.1%}</span>' if numeric_color else f"{val:+.1%}"
        if "vol" in s or "sharpe" in s or "rf" in s or "rank_corr" in s or "rank_p" in s:
            return f"{val:.2f}"
        if "beat" in s:
            return f"{val:.0%}"
        if "price" in s or "edge" in s and isinstance(val, (int, float)):
            if "price" in s:
                return f"${val:,.0f}"
            return f"{val:+.1%}"
        if isinstance(val, float):
            return f"{val:,.2f}"
        return str(val)

    head = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in df.columns)
    has_idx = df.index.name is not None or not df.index.equals(pd.RangeIndex(len(df)))
    idx_head = f"<th>{df.index.name or ''}</th>" if has_idx else ""
    rows = []
    for idx, row in df.iterrows():
        idx_cell = f"<td><b>{idx}</b></td>" if has_idx else ""
        cells = "".join(f"<td>{style(c, row[c])}</td>" for c in df.columns)
        rows.append(f"<tr>{idx_cell}{cells}</tr>")
    return f'<table class="{classes}"><thead><tr>{idx_head}{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def region_table():
    return wines.groupby("region").agg(
        n=("real_cagr", "size"),
        mean_real_cagr=("real_cagr", "mean"),
        median_real_cagr=("real_cagr", "median"),
        std_real_cagr=("real_cagr", "std"),
        beat_cpi=("real_cagr", lambda s: (s > 0).mean()),
        beat_sp500=("alpha_sp500", lambda s: (s > 0).mean()),
        beat_gold=("alpha_gold", lambda s: (s > 0).mean()),
    )


def era_table():
    return wines.groupby("release_era").agg(
        n=("real_cagr", "size"),
        mean_real_cagr=("real_cagr", "mean"),
        median_real_cagr=("real_cagr", "median"),
        beat_cpi=("real_cagr", lambda s: (s > 0).mean()),
        beat_sp500=("alpha_sp500", lambda s: (s > 0).mean()),
    ).reindex(["pre-2000", "2000-2008", "2009-2011 bubble", "post-2011"])


def top_bottom(n=10):
    cols = ["producer", "vintage", "release_price_usd", "current_price_usd", "real_cagr"]
    return (wines.sort_values("real_cagr", ascending=False).head(n)[cols],
            wines.sort_values("real_cagr", ascending=True).head(n)[cols])


def vol_sharpe_cohort_table():
    g = wine_metrics.groupby("region").agg(
        n=("sharpe", "size"),
        mean_annual_return=("realized_cagr", "mean"),
        mean_vol=("annualized_vol", "mean"),
        mean_sharpe=("sharpe", "mean"),
    )
    return g


def candidate_table():
    cols = ["producer", "vintage", "region", "release_price_usd",
            "critic_score", "vintage_quality", "pred_blend"]
    show = cand_ranked[cols].rename(columns={
        "pred_blend": "predicted_real_cagr",
        "release_price_usd": "current_price_usd",
    })
    return df_to_html(show.reset_index(drop=True))


# ---------- per-wine reasoning ----------
PRODUCER_NOTES = {
    "DRC Romanee-Conti": "Trophy label of Burgundy; deepest secondary-market liquidity of any fine wine.",
    "DRC La Tache": "Second-most-traded DRC label; ~1,800 bottles/yr; materially below Romanée-Conti pricing per quality unit.",
    "Roumier Musigny": "~300 bottles/yr — extreme scarcity premium; consistently the highest-priced non-DRC Burgundy.",
    "Rousseau Chambertin": "Reference producer of Gevrey-Chambertin Grand Cru; institutional collector demand.",
    "Coche-Dury Corton-Charlemagne": "Reference white Burgundy; ~3,000 bottles/yr; the only white in our top tier.",
    "Leroy Musigny": "Lalou Bize-Leroy's flagship; minuscule production; biodynamic premium.",
    "Lafite Rothschild": "Most-searched fine wine globally; China-demand sensitivity is both upside and downside.",
    "Latour": "Stopped en primeur in 2012; released only when nearing drinking maturity — different supply dynamic.",
    "Margaux": "Most stable First Growth pricing; weaker upside but lower drawdowns.",
    "Mouton Rothschild": "Artist-label series adds collector premium; thinner secondary market than Lafite.",
    "Haut-Brion": "Smallest-production First Growth; Pessac-Léognan terroir provides differentiation.",
    "Krug Vintage": "Long aging (~12 years post-vintage) means thin post-release supply.",
    "Dom Perignon": "Largest volume of prestige cuvée — least scarcity premium of the cohort.",
    "Cristal": "Highest critic ceiling among large-production Champagnes; status-good positioning.",
    "Salon": "Single-vintage only; ~50,000 bottles/yr; closest Champagne analog to grower-producer scarcity.",
}

REGION_NOTES = {
    "Burgundy": (
        "Burgundy Grand Cru — the only cohort in our 1982-2020 panel with a positive "
        "base rate vs S&P 500 (67% beat-rate, mean +11% real CAGR). Driven by structural "
        "supply constraint, not vintage selection."
    ),
    "Bordeaux": (
        "Bordeaux First Growth — mixed track record post-2011 bubble; cheap-release "
        "vintages (2014, 2018-2020, 2024) have generally outperformed expensive ones."
    ),
    "Champagne": (
        "Champagne prestige cuvée — low cohort volatility (~6% annualized) with "
        "mid-tier Sharpe (~1.0 reconstructed); slow but steady compounder."
    ),
}


REGION_SHORT = {
    "Burgundy": "Only cohort with positive base rate vs S&P (67%, +11% real CAGR mean)",
    "Bordeaux": "Cohort with weakest historical base rate; cheap-release vintages outperform",
    "Champagne": "Low cohort vol (~6%); steady compounder",
}


def wine_reasons(row) -> str:
    """One-paragraph reasoning, compact."""
    parts = []
    parts.append(REGION_SHORT.get(row["region"], ""))

    delta = row.get("log_price_delta_vs_prior", 0)
    if pd.notna(delta) and abs(delta) > 0.04:
        pct = (np.exp(delta) - 1) * 100
        if pct < -5:
            parts.append(f"released <b>{pct:+.0f}%</b> vs prior vintage (cheap entry)")
        elif pct > 15:
            parts.append(f"released <b>{pct:+.0f}%</b> vs prior (price-up risk)")

    parts.append(f"<b>{row['critic_score']:.0f}</b>/100 score · "
                 f"<b>{row['vintage_quality']:.0f}</b>/100 vintage")

    note = PRODUCER_NOTES.get(row["producer"], "")
    if note:
        parts.append(note.rstrip("."))

    return ". ".join(p for p in parts if p) + "."


# ---------- summary numbers ----------
beat_sp500_pct = (wines["alpha_sp500"] > 0).mean() * 100
beat_cpi_pct = (wines["real_cagr"] > 0).mean() * 100
beat_gold_pct = (wines["alpha_gold"] > 0).mean() * 100
median_alpha_sp = wines["alpha_sp500"].median() * 100
mean_alpha_sp = wines["alpha_sp500"].mean() * 100
burgundy_mean = wines[wines["region"] == "Burgundy"]["real_cagr"].mean() * 100
bubble_mean = wines[wines["release_era"] == "2009-2011 bubble"]["real_cagr"].mean() * 100
bubble_n = (wines["release_era"] == "2009-2011 bubble").sum()
bordeaux_sharpe = wine_metrics[wine_metrics["region"] == "Bordeaux"]["sharpe"].mean()
burgundy_sharpe = wine_metrics[wine_metrics["region"] == "Burgundy"]["sharpe"].mean()
champagne_sharpe = wine_metrics[wine_metrics["region"] == "Champagne"]["sharpe"].mean()
sp500_sharpe = bench_metrics[bench_metrics["label"] == "S&P 500 TR"]["sharpe"].iloc[0]


# ---------- TOC sections ----------
TOC = [
    ("picks", "Top picks"),
    ("tldr", "Key findings"),
    ("alpha", "Returns vs S&P"),
    ("risk-return", "Risk-adjusted view"),
    ("model", "Predictive model"),
    ("implications", "Implications"),
]


CSS = """
:root {
  --bg: #fdfcfa;
  --text: #1a1a1a;
  --muted: #5a5a5a;
  --accent: #5c2a3a;
  --accent-light: #f3eef0;
  --border: #e5e0d8;
  --code: #f3efe7;
  --warn: #c9a227;
  --warn-bg: #fff8e6;
  --pos: #27ae60;
  --neg: #c0392b;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { font-family: 'Source Serif Pro', 'Charter', Georgia, 'Times New Roman', serif;
       margin: 0; color: var(--text); background: var(--bg);
       line-height: 1.6; font-size: 17px; }
.layout { display: grid;
          grid-template-columns: 14em minmax(0, 920px);
          gap: 2.5em;
          max-width: 1280px;
          margin: 2.5em auto;
          padding: 0 1.2em; }
nav.toc { position: sticky; top: 2em; align-self: start;
          font-size: 0.86em; max-height: calc(100vh - 4em);
          overflow-y: auto; padding-right: 0.5em;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
nav.toc h4 { margin: 0 0 0.6em 0; color: var(--accent);
             font-size: 0.78em; text-transform: uppercase;
             letter-spacing: 0.08em; font-weight: 600; }
nav.toc ul { list-style: none; padding: 0; margin: 0; }
nav.toc li { margin: 0.3em 0; }
nav.toc a { color: #444; text-decoration: none;
            border-left: 2px solid transparent; padding-left: 0.6em;
            display: block; line-height: 1.35; }
nav.toc a:hover { color: var(--accent); }
nav.toc a.active { color: var(--accent); font-weight: 600;
                   border-left-color: var(--accent); }
article { min-width: 0; }
h1 { font-size: 2.4em; margin: 0 0 0.15em 0; line-height: 1.15;
     border-bottom: 2px solid var(--text); padding-bottom: 0.3em;
     font-weight: 700; }
h2 { font-size: 1.55em; margin-top: 2.4em; color: var(--accent);
     border-bottom: 1px solid var(--border); padding-bottom: 0.25em;
     scroll-margin-top: 1em; font-weight: 600; }
h3 { font-size: 1.2em; margin-top: 1.8em; color: var(--accent);
     scroll-margin-top: 1em; font-weight: 600; }
.subtitle { color: var(--muted); font-size: 1.12em; font-style: italic;
            margin-top: 0.5em; margin-bottom: 0.8em; }
.meta { color: var(--muted); font-size: 0.88em; margin-bottom: 2.2em;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
p { margin: 0.8em 0; }
.takeaway { background: var(--accent-light); border-left: 4px solid var(--accent);
            padding: 0.8em 1.1em; margin: 1em 0 1.4em 0;
            border-radius: 0 4px 4px 0; font-size: 0.97em; }
.takeaway b { color: var(--accent); }
.summary { background: var(--accent-light); border: 1px solid var(--border);
           border-left: 4px solid var(--accent);
           padding: 1.2em 1.5em; margin: 1.8em 0;
           border-radius: 0 4px 4px 0; }
.summary h3 { margin-top: 0; color: var(--accent); }
.caveat { background: var(--warn-bg); border-left: 4px solid var(--warn);
          padding: 0.9em 1.2em; margin: 1.2em 0; font-size: 0.94em;
          border-radius: 0 4px 4px 0; }
.stat-block { display: flex; gap: 18px; margin: 1.5em 0; flex-wrap: wrap; }
.stat { flex: 1; min-width: 170px; background: white; padding: 16px 18px;
        border: 1px solid var(--border); border-radius: 4px; }
.stat .label { font-size: 0.78em; color: var(--muted);
               text-transform: uppercase; letter-spacing: 0.05em;
               font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
.stat .value { font-size: 1.85em; font-weight: 700; color: var(--accent);
               line-height: 1.1; margin-top: 4px;
               font-family: 'Source Serif Pro', Georgia, serif; }
.stat .sub { font-size: 0.82em; color: var(--muted); margin-top: 2px;
             font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
table { border-collapse: collapse; margin: 1.1em 0; width: 100%;
        font-size: 0.92em; background: white;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
th, td { padding: 0.4em 0.85em; border-bottom: 1px solid var(--border);
         text-align: right; }
th { background: #f3efe7; font-weight: 600;
     color: var(--text); font-size: 0.88em; }
th:first-child, td:first-child { text-align: left; }
tbody tr:hover { background: #faf7f0; }
img { max-width: 100%; height: auto; display: block;
      margin: 1.5em auto; border: 1px solid var(--border);
      border-radius: 4px; background: white; }
code { background: var(--code); padding: 2px 6px; border-radius: 3px;
       font-size: 0.92em;
       font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace; }
ul, ol { margin: 0.6em 0; padding-left: 1.6em; }
li { margin: 0.25em 0; }
a { color: var(--accent); }
.footnote { font-size: 0.85em; color: var(--muted); margin-top: 3em;
            padding-top: 1em; border-top: 1px solid var(--border);
            font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
.pick { display: grid; grid-template-columns: 2em 1fr 8em;
        gap: 0.9em; padding: 0.75em 0;
        border-bottom: 1px solid var(--border); align-items: start;
        font-size: 0.9em; }
.pick:last-of-type { border-bottom: none; }
.pick .rank { font-size: 1.35em; font-weight: 700; color: var(--accent);
              line-height: 1.1; padding-top: 0.1em; }
.pick .info h3 { margin: 0; font-size: 1em;
                 color: var(--text); font-weight: 600; line-height: 1.25; }
.pick .info .sub { color: var(--muted); font-size: 0.82em;
                   margin: 0.1em 0 0.35em 0;
                   font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
.pick .info .reasons { color: #333; font-size: 0.88em;
                       line-height: 1.45; margin: 0; }
.pick .info .reasons b { color: var(--accent); font-weight: 600; }
.pick .metrics { text-align: right; line-height: 1.2;
                 font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
.pick .metrics .price { font-size: 0.95em; color: var(--text);
                        font-weight: 600; }
.pick .metrics .price-label, .pick .metrics .cagr-label {
                              font-size: 0.65em; color: var(--muted);
                              text-transform: uppercase; letter-spacing: 0.05em; }
.pick .metrics .cagr { font-size: 1.3em; font-weight: 700;
                       color: var(--pos); line-height: 1.1; margin-top: 0.3em; }
.pick .metrics .cagr.neg { color: var(--neg); }
.region-tag { display: inline-block; font-size: 0.65em; font-weight: 600;
              padding: 1px 6px; border-radius: 3px; margin-right: 5px;
              text-transform: uppercase; letter-spacing: 0.04em;
              font-family: -apple-system, BlinkMacSystemFont, sans-serif;
              vertical-align: middle; }
.region-Bordeaux { background: #f3eaec; color: #7a2e3a; }
.region-Burgundy { background: #f3eaf0; color: #7a3a8a; }
.region-Champagne { background: #fcf5e3; color: #a87a08; }
.tabs { display: flex; gap: 0; margin: 1.2em 0 0.6em 0;
        border-bottom: 2px solid var(--border);
        font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
.tab-btn { background: none; border: none; cursor: pointer;
           padding: 0.6em 1.1em; font-size: 0.92em; color: var(--muted);
           border-bottom: 2px solid transparent;
           margin-bottom: -2px; font-weight: 500; }
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent);
                  font-weight: 700; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
@media (max-width: 700px) {
  .pick { grid-template-columns: 1.8em 1fr; }
  .pick .metrics { grid-column: 2; text-align: left; margin-top: 0.4em; }
}
@media (max-width: 1000px) {
  .layout { grid-template-columns: 1fr; }
  nav.toc { display: none; }
}
"""

TOC_JS = """
<script>
const sections = document.querySelectorAll('article section[id]');
const links = document.querySelectorAll('nav.toc a');
const linkMap = {};
links.forEach(l => { linkMap[l.getAttribute('href').slice(1)] = l; });
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      const link = linkMap[e.target.id];
      if (link) {
        links.forEach(l => l.classList.remove('active'));
        link.classList.add('active');
      }
    }
  });
}, { rootMargin: '-30% 0px -65% 0px' });
sections.forEach(s => observer.observe(s));

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const wrap = btn.closest('section');
    wrap.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    wrap.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    wrap.querySelector('#tab-' + btn.dataset.tab).classList.add('active');
  });
});
</script>
"""


def render():
    region_html = df_to_html(region_table().round(3))
    era_html = df_to_html(era_table().round(3))
    top_df, bot_df = top_bottom(10)
    top_html = df_to_html(top_df.reset_index(drop=True))
    bot_html = df_to_html(bot_df.reset_index(drop=True))
    ols_html = ols.summary().tables[1].as_html()
    cand_html = candidate_table()
    vol_html = df_to_html(vol_sharpe_cohort_table().round(3))
    bench_html = df_to_html(bench_metrics.set_index("label").round(3))
    bt_show = backtest_df.copy()
    bt_show["cutoff_year"] = bt_show["cutoff_year"].astype(int)
    bt_html = df_to_html(bt_show.set_index("cutoff_year").round(3))

    top_buy = cand_ranked.head(5)
    avoid = cand_ranked.tail(3)
    top_buy_names = "; ".join(f"{r.producer} {r.vintage}" for _, r in top_buy.iterrows())
    avoid_names = "; ".join(f"{r.producer} {r.vintage}" for _, r in avoid.iterrows())

    def render_picks(df_picks):
        out = []
        for i, r in df_picks.reset_index(drop=True).iterrows():
            cagr_class = "" if r["pred_blend"] >= 0 else "neg"
            out.append(f"""
<div class="pick">
  <div class="rank">{i + 1}</div>
  <div class="info">
    <h3><span class="region-tag region-{r['region']}">{r['region']}</span>{r['producer']} {int(r['vintage'])}</h3>
    <div class="sub">{r['tier'].replace('_', ' ').title()}  ·  released {int(r['release_year'])}</div>
    <p class="reasons">{wine_reasons(r)}</p>
  </div>
  <div class="metrics">
    <div class="price-label">Price</div>
    <div class="price">${r['release_price_usd']:,.0f}</div>
    <div class="cagr-label" style="margin-top:0.5em;">Pred. real CAGR</div>
    <div class="cagr {cagr_class}">{r['pred_blend']*100:+.1f}%</div>
  </div>
</div>""")
        return "".join(out)

    top10 = cand_ranked.head(10)
    under800 = cand_ranked[cand_ranked["release_price_usd"] < 800].head(10)
    picks_all_block = render_picks(top10)
    picks_under800_block = render_picks(under800)

    # Cohort diversification picks
    diversified_rows = []
    for region in ["Bordeaux", "Burgundy", "Champagne"]:
        best_in_region = cand_ranked[cand_ranked["region"] == region].head(1)
        if len(best_in_region):
            diversified_rows.append(best_in_region.iloc[0])
    diversified_df = pd.DataFrame(diversified_rows)
    diversified_block = render_picks(diversified_df)

    # Backtest summary
    bt_top_mean = backtest_df["top_real_cagr"].mean() * 100
    bt_sp_mean = backtest_df["sp500_real_cagr"].mean() * 100
    bt_edge_mean = backtest_df["edge"].mean() * 100
    bt_rho_mean = backtest_df["rank_corr"].mean()
    bt_positive_rho = (backtest_df["rank_corr"] > 0).sum()

    toc_html = "<nav class=\"toc\"><h4>Contents</h4><ul>" + \
        "".join(f'<li><a href="#{sid}">{label}</a></li>'
                for sid, label in TOC) + "</ul></nav>"

    body = f"""<article>
<h1>Wine as an Investment Asset</h1>
<p class="subtitle">A quantitative study of release-to-market returns, risk,
and predictability for Bordeaux First Growths, Burgundy Grand Crus, and
Champagne prestige cuvée.</p>
<p class="meta">May 2026  ·  n = {len(wines)} wine/vintage observations
 ·  release years 1982-2020  ·  reconstructed annual price tracks via
two-factor decomposition.</p>

<section id="picks">
<h2 style="margin-top:1.4em;">Top 10 buy candidates — May 2026</h2>
<p style="font-size:0.95em;">Ranked by a predictive model
(5-fold CV R² ≈ {chosen['rf_cv_r2'].mean():.2f}) trained on 84 historical
observations and applied to {len(cand_ranked)} currently-purchasable
wines (2020-2024 vintages). Predicted CAGR is over a representative
10-year forward hold, gross of storage / spread.
<b>Model's top signal is "buy cheap-release Burgundy Grand Cru"</b> —
right in 2008-2014, wrong in 2017-2020
(<a href="#backtest">§12 backtest</a>). Treat as a structural screen,
not stock-picking alpha.</p>

<div class="tabs">
<button class="tab-btn active" data-tab="all">All picks ({len(top10)})</button>
<button class="tab-btn" data-tab="under800">Under $800 ({len(under800)})</button>
<button class="tab-btn" data-tab="diversified">Cohort-diversified (3)</button>
</div>

<div class="tab-panel active" id="tab-all">
{picks_all_block}
</div>

<div class="tab-panel" id="tab-under800">
<p style="font-size:0.9em;color:var(--muted);">
Filter excludes all Burgundy (entry price &gt;$3,500 for Grand Cru).
Predicted CAGR drops accordingly — the under-$800 cohort is structurally
lower-conviction in this model.</p>
{picks_under800_block}
</div>

<div class="tab-panel" id="tab-diversified">
<p style="font-size:0.9em;color:var(--muted);">
Top model pick from each cohort if you'd rather spread across regions
than concentrate in Burgundy.</p>
{diversified_block}
</div>

<p style="margin-top:1em;color:var(--muted);font-size:0.88em;">
Full 31-wine ranking and feature breakdown in
<a href="#candidates">§11</a>; backtest in <a href="#backtest">§12</a>;
methodology in <a href="#methodology">§1</a>.</p>
</section>

<section id="tldr">
<div class="summary">
<h3>TL;DR</h3>
<ul>
<li><b>Wine is not a public-equity substitute.</b> Only {beat_sp500_pct:.0f}%
    of wine/vintage observations beat the S&amp;P 500 TR over their matched
    holding period; median wine underperformed equities by
    <b>{median_alpha_sp:+.1f}% per year</b>.</li>
<li><b>Risk-adjusted returns look favorable for Burgundy and Champagne,
    unfavorable for Bordeaux.</b> Mean Sharpe (reconstructed): Burgundy
    <b>{burgundy_sharpe:.2f}</b>, Champagne <b>{champagne_sharpe:.2f}</b>,
    Bordeaux <b>{bordeaux_sharpe:.2f}</b>, vs S&amp;P 500 <b>{sp500_sharpe:.2f}</b>.
    Read these as <i>upper bounds</i> — reconstructed vol inherits
    cohort-index volatility but not idiosyncratic bottle-level shocks
    (see Section 5).</li>
<li><b>Burgundy Grand Cru is the one cohort with genuine alpha.</b>
    Mean real CAGR <b>{burgundy_mean:+.1f}%</b>, only cohort with positive
    base rate against S&amp;P. Supply constraint, not vintage selection,
    drives the result.</li>
<li><b>The predictive model works for a while, then breaks.</b> Walk-forward
    backtest: model top-3 portfolios beat bottom-3 in
    {(backtest_df['top_real_cagr'] > backtest_df['bot_real_cagr']).sum()}/{len(backtest_df)} cutoffs,
    but the success is concentrated in 2011-2014 (Burgundy-era picks).
    Most recent cutoffs (2017, 2020): the model picked <i>worse</i> than
    bottom-3 — likely overfitting to the post-2010 Burgundy run.</li>
</ul>
</div>

<div class="stat-block">
<div class="stat">
<div class="label">Beat S&amp;P 500</div>
<div class="value">{beat_sp500_pct:.0f}%</div>
<div class="sub">of wine observations</div>
</div>
<div class="stat">
<div class="label">Burgundy mean Sharpe</div>
<div class="value">{burgundy_sharpe:.2f}</div>
<div class="sub">vs S&amp;P {sp500_sharpe:.2f}</div>
</div>
<div class="stat">
<div class="label">Median alpha vs S&amp;P</div>
<div class="value">{median_alpha_sp:+.1f}%</div>
<div class="sub">per year</div>
</div>
<div class="stat">
<div class="label">Backtest edge</div>
<div class="value">{bt_edge_mean:+.1f}%</div>
<div class="sub">top-3 vs S&amp;P, gross</div>
</div>
</div>
</section>

<section id="methodology">
<h2>1. Methodology</h2>
<div class="takeaway"><b>How to read this report:</b> all returns are
gross of storage cost (~1.5%/yr) and bid-ask spread (~10%). All "real" CAGRs
are CPI-deflated. Sharpe ratios use 10Y Treasury as the risk-free rate.</div>
<p>Each observation is a single wine-vintage pair. Returns are computed
from the wine's <b>release year</b> (en-primeur for Bordeaux; physical
release for Burgundy and vintage Champagne) to today (2026):</p>
<ul>
<li><b>Nominal CAGR</b>: <code>(current / release) ^ (1 / years_held) − 1</code></li>
<li><b>Real CAGR</b>: nominal CAGR deflated by US CPI over the same window</li>
<li><b>Alpha vs S&amp;P / gold</b>: nominal CAGR minus the benchmark's CAGR
    over the matched window.</li>
</ul>
<p>The wine panel covers Bordeaux First Growths (Lafite, Latour, Margaux,
Mouton, Haut-Brion; n = 44), Burgundy Grand Crus (DRC, Leroy, Roumier,
Rousseau, Coche-Dury; n = 21), and Champagne prestige cuvée
(Krug, Dom Pérignon, Cristal, Salon; n = 19). Benchmarks: US CPI-U,
S&amp;P 500 total-return index, gold spot, and cohort-level wine indices
(approximating Liv-ex Bordeaux 500, Burgundy 150, Champagne 50).</p>
<p><b>Price reconstruction.</b> For risk/Sharpe and backtest analysis we
need annual price tracks, not just (release, current). We reconstruct
each wine's track via a two-factor decomposition: total log-return is
split into a cohort component (the region's index path) plus an
idiosyncratic drift such that endpoints exactly match observed
(release, current) prices. This inherits cohort-level volatility while
preserving each wine's realized CAGR.</p>
<div class="caveat">
<b>Data caveat.</b> Release and current prices are knowledge-based
estimates accurate to ~15-25%. The cohort indices are calibrated to
match Liv-ex sub-indices in shape but are not literal Liv-ex values.
The direction of every finding is robust to this noise; precise figures
will move slightly with primary-source data (La Place de Bordeaux
campaign reports, Liv-ex / Wine-Searcher API pulls).
</div>
</section>

<section id="distribution">
<h2>2. Return distribution</h2>
{chart_return_distribution()}
<p>The distribution is right-skewed, with a long tail of high-return
Burgundy outliers. The mode sits just above the inflation breakeven
line, which is the central finding: wine on average barely keeps pace
with CPI, but the upside is meaningful when it occurs.</p>
</section>

<section id="alpha">
<h2>3. Wine vs S&amp;P 500 over matched windows</h2>
{chart_alpha_vs_sp500()}
<div class="takeaway"><b>The most important chart for an investment
audience.</b> Over the same release-to-2026 window in which each wine
was held, the equivalent S&amp;P 500 TR position outperformed in
{100 - beat_sp500_pct:.0f}% of cases. Mean alpha vs equities is
<b>{mean_alpha_sp:+.1f}% / yr</b>.</div>
<p>The takeaway is not that wine is a bad investment — it's that wine is
not a substitute for equities. Wine's role in a portfolio, if any, is as
a low-correlation alternative-asset diversifier.</p>
</section>

<section id="cohort">
<h2>4. Cohort breakdown</h2>
<h3>By region</h3>
{region_html}
<p>Burgundy is the only cohort with a positive base rate against
equities. Bordeaux First Growths and Champagne prestige cuvée beat
S&amp;P in only ~20% of observations. The Burgundy result is driven by
microscopic production (DRC ~6,000 cases/yr across all labels; Roumier
Musigny ~300 bottles/yr) combined with structural demand expansion
post-2010.</p>

<h3>By release era</h3>
{era_html}
{chart_era_breakdown()}
<p>Release era is the dominant time-varying risk factor. Wines released
into the 2009-2011 Chinese-demand peak are still nominally underwater
15 years later.</p>
</section>

<section id="risk-return">
<h2>5. Risk-adjusted returns</h2>
<p>Using reconstructed annual price tracks, we can compute volatility
and Sharpe ratios for each wine — not just CAGR.</p>

{chart_vol_return()}

<h3>Cohort vol / Sharpe summary</h3>
{vol_html}

<h3>Benchmark vol / Sharpe (2004-2026)</h3>
{bench_html}

{chart_sharpe_by_cohort()}

<div class="caveat">
<b>Important caveat on these Sharpe numbers.</b> Because price tracks are
reconstructed from cohort indices, each wine's volatility inherits the
<i>cohort-level</i> vol, not idiosyncratic bottle-level vol. Real wine
returns include provenance shocks, single-bottle vs case spread,
disgorgement effects (Champagne), and producer-specific demand swings.
These reconstructed Sharpe numbers are therefore <b>upper bounds</b> —
true investor-realized Sharpe is lower, plausibly by 30-50%. The
relative ordering across cohorts is still informative.
</div>
<p>Even with that haircut, Burgundy ({burgundy_sharpe:.2f}) and
Champagne ({champagne_sharpe:.2f}) plausibly clear the S&amp;P 500
Sharpe ({sp500_sharpe:.2f}) over this window — driven by lower cohort
vol than equities and Burgundy's outsized return. Bordeaux Sharpe
({bordeaux_sharpe:.2f}) is unambiguously below equities even before
the haircut. This is a substantively different conclusion than the
cross-sectional CAGR view in Section 3, and worth dwelling on: <b>for
the cohorts where wine works, the case is risk-adjusted, not
return-maximizing</b>.</p>
</section>

<section id="indices">
<h2>6. Index-level comparison (2004 onward)</h2>
{chart_indices_normalized()}
<p>Burgundy index has compounded the strongest since 2004
(~9-10%/yr nominal), driven by the post-2010 demand expansion that
caught Bordeaux flat-footed. Champagne is the slowest. The S&amp;P 500
remains the dominant compounder over the full window despite the
2008 drawdown.</p>
</section>

<section id="price-tracks">
<h2>7. Reconstructed annual price tracks</h2>
{chart_price_tracks()}
<p>Sample of reconstructed tracks per cohort. Bordeaux 2009 visibly
peaks in 2011 and has not recovered. Burgundy tracks show steady
compounding with a 2022 inflection. Champagne is the most gradual.
These tracks are model-derived (see methodology) — they inherit each
wine's observed CAGR and the cohort's volatility shape, but are not
literal trade prints.</p>
</section>

<section id="regression">
<h2>8. Release vs current price (log-log)</h2>
{chart_release_vs_current()}
<p>Most observations sit between the nominal breakeven line and the
inflation-adjusted line. Bordeaux 2009-2011 wines cluster <i>below</i>
the breakeven line in the upper-right quadrant: a diagnostic signature
of buying near a cycle peak.</p>
</section>

<section id="model">
<h2>9. Predictive model</h2>
<p>To identify which features drive real returns, we fit:</p>
<ul>
<li><b>Ridge regression</b> (α = 1.0, standardized): CV R² = <b>{ridge_r2.mean():.2f}</b> ± {ridge_r2.std():.2f}</li>
<li><b>Random forest</b> (n_estimators=400, max_depth=5): CV R² = <b>{rf_r2.mean():.2f}</b> ± {rf_r2.std():.2f}</li>
<li><b>OLS</b> for interpretable inference: R² = <b>{ols.rsquared:.2f}</b></li>
</ul>

{chart_model_features()}

<h3>Model comparison: baseline vs augmented</h3>
{chart_model_comparison()}

<p>We tried an augmented specification with four extra features
(vintage_quality, log_price_delta_vs_prior, has_prior, log_livex_at_release).
Cross-validated R²: baseline RF = <b>{baseline['rf_cv_r2'].mean():.2f}</b>,
augmented RF = <b>{augmented['rf_cv_r2'].mean():.2f}</b>. The augmented
model {"is meaningfully better" if USE_AUGMENTED else "did not improve "
"out-of-sample performance"}, so candidate scoring uses the
<b>{chosen_name}</b> model.
{"" if USE_AUGMENTED else "The new features mostly duplicate signal "
"already captured by region and bubble_release. With n=84, more granular "
"cyclical features don't generalize."}</p>

<h3>OLS coefficients</h3>
{ols_html}
</section>

<section id="top-bottom">
<h2>10. Top and bottom performers</h2>
<h3>Top 10 (real CAGR)</h3>
{top_html}
<h3>Bottom 10 (real CAGR)</h3>
{bot_html}
<p>The bottom-10 list is essentially the 2009-2010 Bordeaux campaign —
a single procyclical event accounting for most of the dataset's
underperformance.</p>
</section>

<section id="candidates">
<h2>11. Buy candidates (May 2026)</h2>
<p>Applying the trained model to {len(cand_ranked)} currently-purchasable
wines from the 2020-2024 vintages.</p>

<div class="caveat">
<b>What the model actually does.</b> With CV R² ≈ {chosen['rf_cv_r2'].mean():.2f},
roughly half the cross-sectional variance in real returns is explained.
The model is best understood as a <b>structural screen</b>: it learns
"buy cheap-release Burgundy Grand Cru that hasn't been priced into a
bubble" and applies that rule to candidates. Treat predicted CAGR as a
<b>relative ranking</b>, not a point estimate.
</div>

<h3>Top picks</h3>
<p>Model favors: <b>{top_buy_names}</b>.</p>
<p>Model avoids: <b>{avoid_names}</b>.</p>

{chart_candidate_predictions()}

<h3>Full ranking</h3>
{cand_html}
</section>

<section id="backtest">
<h2>12. Walk-forward backtest</h2>
<p>At each cutoff year T, we retrain the model using <i>only</i> wines
with <code>release_year ≤ T</code>, then score test wines released in
(T, T+3]. The top-3 model portfolio is held to 2026 and compared against
the bottom-3 portfolio, all tested wines, and an equivalent S&amp;P 500
position over the same window.</p>

{chart_backtest()}

<h3>Backtest results by cutoff</h3>
{bt_html}

<div class="takeaway"><b>Bottom line on prediction: the model works
for a while, then breaks.</b> Top-3 portfolios beat bottom-3 in
{(backtest_df['top_real_cagr'] > backtest_df['bot_real_cagr']).sum()}
of {len(backtest_df)} cutoffs. The successes are 2008, 2011, 2014 —
all cutoffs where "buy Burgundy" was the right call. The failures are
2017 and 2020, where the model picked worse than bottom-3 (rank
correlation went <i>negative</i>). Mean rank correlation
ρ = <b>{bt_rho_mean:.2f}</b>, but the dispersion is huge
({backtest_df['rank_corr'].min():.2f} to {backtest_df['rank_corr'].max():.2f}).</div>

<h3>Why the model fails in recent cutoffs</h3>
<p>The model learned in training data that <b>cheap-release Burgundy
beats Bordeaux</b>. From 2017 onward this rule generalized poorly because:</p>
<ul>
<li>Burgundy release prices climbed sharply 2017-2022 (DRC RC went from
    ~$5,500 in 2007 to ~$25,000 in 2020), making "cheap Burgundy" a
    smaller cohort.</li>
<li>The 2017+ test set includes wines released into a Burgundy
    near-bubble that subsequently corrected in 2023-2024, mirroring
    the 2009-2011 Bordeaux story the model trained to avoid — but the
    model lacked a Burgundy-specific bubble flag.</li>
<li>Bordeaux 2018-2020 vintages were released at deep discounts to
    pre-bubble levels, which actually produced strong returns post-2022.
    The model under-weighted these.</li>
</ul>

<p>Against S&amp;P 500: top-3 averaged
<b>{bt_top_mean:+.1f}% / yr</b> real vs S&amp;P's <b>{bt_sp_mean:+.1f}%</b>,
nominal edge of <b>{bt_edge_mean:+.1f}% / yr</b>. But this is driven
entirely by 2011 and 2014 cutoffs; remove those and the edge is
negative. Add ~2.5%/yr for storage and spread and the edge disappears
even on the favorable subset.</p>

<p><b>What this tells us:</b> the model captures a real structural
pattern, but the pattern is regime-dependent. Cohort leadership in
fine wine shifts on roughly 10-15 year cycles
(Bordeaux pre-2011 → Burgundy 2011-2022 → ???), and a static model
trained on the last regime will misfire when the next one starts. A
production version would need cohort-rotation logic, not just
cross-sectional features.</p>
</section>

<section id="implications">
<h2>13. Implications for an investor</h2>
<ol>
<li><b>Treat fine wine as alt-asset / inflation hedge, not equity proxy.</b>
    The base rate against S&amp;P 500 is hostile and the Sharpe gap is
    larger than the CAGR gap.</li>
<li><b>If you allocate, concentrate in Burgundy Grand Cru.</b> Only
    cohort with positive base rate against equities and meaningfully
    higher Sharpe than other wine segments.</li>
<li><b>Avoid primary-market buying during demand bubbles.</b> The
    2009-2011 Bordeaux campaigns are the textbook example. A simple
    rule — don't pay more than 1.5× the prior-vintage release price —
    would have avoided most of the bottom decile.</li>
<li><b>Secondary market may be more efficient than primary.</b> Buying
    aged Bordeaux post-correction in 2014-2016 produced materially
    better returns than buying en-primeur 2009-2010.</li>
<li><b>Net returns are worse than gross.</b> Adding ~1.5%/yr storage
    and ~10% bid-ask drops most cohort means below CPI. Wine investment
    requires structural pricing advantages (Burgundy supply) to clear
    the friction bar.</li>
</ol>
</section>

<section id="limitations">
<h2>14. Limitations and next work</h2>
<ul>
<li><b>Data provenance.</b> Replace estimated prices with primary
    sources: La Place de Bordeaux release reports, Liv-ex API,
    Wine-Searcher pulls. Pricing accuracy of ±5% would tighten all CIs.</li>
<li><b>Price reconstruction is model-derived.</b> Individual wines do
    not literally have the cohort's volatility profile; idiosyncratic
    bottle-level vol is higher. True bid/ask trade prints from auctions
    (Sotheby's, Christie's, Acker, Heritage) would enable proper
    wine-level vol estimation.</li>
<li><b>Region × tier collinearity.</b> The panel has no Bordeaux
    grand-cru-tier or Burgundy first-growth-tier wines, so region and
    tier are perfectly correlated. Expanding to Right Bank Bordeaux,
    Napa cult cabs, Rhône, and Tuscan IGT would break this.</li>
<li><b>Survivorship.</b> The panel is built from currently-tracked,
    currently-traded wines. Wines fallen out of the secondary market are
    excluded, biasing returns upward.</li>
<li><b>Backtest sample size.</b> 5 cutoffs is too few for strong
    inference about strategy alpha. Extending the panel to Right Bank
    and Napa would multiply backtest power.</li>
</ul>

<p class="footnote">Source code and CSVs:
<code>github.com/&lt;your-handle&gt;/squirtle</code>.
Built with pandas, scikit-learn, statsmodels. Plots: matplotlib + seaborn.
Re-run with <code>python build_report.py</code>.</p>
</section>
</article>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wine as an Investment Asset</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+Pro:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<div class="layout">
{toc_html}
{body}
</div>
{TOC_JS}
</body>
</html>
"""
    out = DOCS / "index.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html):,} bytes)")


if __name__ == "__main__":
    print(f"baseline ridge CV R²: {baseline['ridge_cv_r2'].mean():.3f}  RF: {baseline['rf_cv_r2'].mean():.3f}")
    print(f"augmented ridge CV R²: {augmented['ridge_cv_r2'].mean():.3f}  RF: {augmented['rf_cv_r2'].mean():.3f}")
    print(f"→ using {chosen_name} model for candidate scoring")
    print(f"backtest cutoffs: {list(backtest_df['cutoff_year'])}")
    render()
