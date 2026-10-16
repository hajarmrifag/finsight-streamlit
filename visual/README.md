# Visual baseline

Screenshots of every page at three viewports, used to see exactly what a change does to the interface.

| Viewport | Size |
| --- | --- |
| desktop | 1440 × 900 |
| tablet | 820 × 1180 |
| mobile | 390 × 844 |

Each image is the full page height, or the viewport height if the page is shorter. States: Overview, Evidence analyst (empty), Evidence analyst (quick answer open), Campaign scenario, Methodology. `baseline/report.json` records per-page content height, horizontal overflow, Streamlit exceptions, and console errors.

## Setup (once)

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m playwright install chromium
```

## Commands

```sh
python scripts/visual_baseline.py capture   # write visual/current/
python scripts/visual_baseline.py compare   # capture, then diff against visual/baseline/ (exit 1 on regression)
python scripts/visual_baseline.py update    # capture, then replace visual/baseline/
```

`compare` fails when more than 0.2% of a page's pixels change (a pixel counts as changed when its brightness differs by more than 16/255) or an image changes size. Differences are written to `visual/diff/`. Every mode also exits 1 on horizontal overflow, a Streamlit exception or a browser console error, and `update` refuses to overwrite the baseline in that case. The script exits early if port 8599 is already in use. After an intentional design change, review the new screenshots and run `update`.

## Limits

- Baselines were captured on macOS. Font rendering differs between operating systems, so compare only on the same platform; the script is not wired into CI for that reason.
- Streamlit runs with the toolbar in minimal mode, animations disabled, and a fixed locale and timezone.
- The semantic-search path is not captured because it depends on a model download.
