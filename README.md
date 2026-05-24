# Wine as an Investment — quantitative study

Self-contained analysis of fine-wine returns vs. equities, gold, and CPI
across Bordeaux First Growths, Burgundy Grand Crus, and Champagne
prestige cuvée.

## Layout

```
data/
  wines.csv          # 84 wine/vintage observations, hand-curated
  benchmarks.csv     # annual CPI, S&P 500 TR, gold, 10Y, Liv-ex 100
analysis.py          # source for the Jupyter notebook (jupytext format)
analysis.ipynb       # executed notebook with inline outputs
build_report.py      # rebuilds the HTML report
docs/
  index.html         # the final report (self-contained, GitHub Pages root)
outputs/             # PNG plots from the notebook
```

## Rebuild the report

```sh
python3 -m venv .venv
.venv/bin/pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter jupytext
.venv/bin/python build_report.py     # → docs/index.html
.venv/bin/jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis.ipynb
```

## Publish to GitHub Pages

1. Push this repo to GitHub.
2. Repo Settings → Pages → Source: **Deploy from a branch**, branch
   `main`, folder `/docs`.
3. Report will be live at
   `https://<your-handle>.github.io/<repo-name>/`.

The `docs/index.html` file is fully self-contained — all CSS and plots
are inlined as base64 — so no asset paths to manage.

## Caveats on data

Prices in `data/wines.csv` are knowledge-based estimates accurate to
~15-25%. The *direction* of every finding in the report is robust to
that noise, but precise CAGR numbers will move with primary-source data.
To make this rigorous, replace with:

- **Release prices:** La Place de Bordeaux annual campaign reports;
  Wine Spectator vintage release archives.
- **Current prices:** Liv-ex API or careful Wine-Searcher pulls
  (Wine-Searcher blocks unauthorized scraping).

Storage cost (~1.5%/yr) and bid-ask spread (~10%) are **not** netted
from returns — figures are gross.
