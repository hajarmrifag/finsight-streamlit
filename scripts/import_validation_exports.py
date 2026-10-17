"""Copy the validation-split research exports into data/research.json.

The churn-reactivation-engine pipeline writes three CSV files. This script reads them (it never
writes to that repository) and replaces the matching blocks under `validation_analysis`:

    python scripts/import_validation_exports.py                  # update data/research.json
    python scripts/import_validation_exports.py --check          # exit 1 if the file is out of date
    python scripts/import_validation_exports.py --exports PATH   # CSV folder (default: sibling repo)

The model comparison block is not imported because the pipeline does not export it as a file; it is
transcribed from the research README and marked `values_rounded`.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import Research  # noqa: E402

REPORT = ROOT / 'data' / 'research.json'
DEFAULT_EXPORTS = ROOT.parent / 'churn-reactivation-engine' / 'data' / 'processed'
REPO = 'churn-reactivation-engine'


def read_csv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def build_blocks(exports):
    targeting = read_csv(exports / 'validation_targeting_metrics.csv')
    importance = read_csv(exports / 'validation_feature_importance.csv')
    profile = read_csv(exports / 'validation_risk_segment_profile.csv')
    return {
        'targeting_curve': {
            'protocol': 'original', 'split': 'validation',
            'points': [{
                'target_fraction': float(r['target_fraction']), 'targeted_rows': int(r['targeted_rows']),
                'score_threshold': float(r['score_threshold']), 'precision': float(r['precision']),
                'recall': float(r['recall']), 'lift': float(r['lift']),
                'positives_captured': int(r['positives_captured']),
            } for r in targeting],
            'source': f'{REPO}: data/processed/validation_targeting_metrics.csv, written by src/evaluate_thresholds.py',
        },
        'feature_importance': {
            'protocol': 'original', 'split': 'validation', 'metric': 'average_precision_decrease', 'repeats': 3,
            'rows': [{'feature': r['feature'], 'importance_mean': float(r['importance_mean']),
                      'importance_std': float(r['importance_std'])} for r in importance],
            'source': f'{REPO}: data/processed/validation_feature_importance.csv, written by src/analyze_feature_importance.py',
        },
        'risk_segment_profile': {
            'protocol': 'original', 'split': 'validation',
            'rows': [{'feature': r['feature'], **{k: float(r[k]) for k in (
                'high_risk_median', 'other_median', 'high_risk_mean', 'other_mean', 'standardized_mean_diff')}}
                for r in profile],
            'source': f'{REPO}: data/processed/validation_risk_segment_profile.csv, written by src/profile_risk_segments.py',
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exports', type=Path, default=DEFAULT_EXPORTS)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not args.exports.is_dir():
        sys.exit(f'Export folder not found: {args.exports}')

    report = json.loads(REPORT.read_text())
    blocks = build_blocks(args.exports)
    stale = [name for name, block in blocks.items() if report['validation_analysis'].get(name) != block]
    if args.check:
        print('Out of date: ' + ', '.join(stale) if stale else 'research.json matches the exports.')
        sys.exit(1 if stale else 0)
    report['validation_analysis'].update(blocks)
    Research.model_validate(report)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print('Updated: ' + ', '.join(stale) if stale else 'Already up to date.')


if __name__ == '__main__':
    main()
