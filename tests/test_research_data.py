"""Integrity checks for data/research.json, the README, and the in-app research notes."""
import copy
import json
import math
import re
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from research import Research, json_schema_text, load_report, SCHEMA_PATH  # noqa: E402

REPORT = json.loads((ROOT / 'data' / 'research.json').read_text())
KNOWLEDGE = json.loads((ROOT / 'data' / 'knowledge.json').read_text())
README = (ROOT / 'README.md').read_text()
research = Research.model_validate(REPORT)
original = research.evaluation('original_test')
purged = research.evaluation('purged_test')


def broken(change):
    """A copy of the report with `change` applied, for checking that bad data is rejected."""
    data = copy.deepcopy(REPORT)
    change(data)
    return data


def evaluation(data, evaluation_id):
    return next(e for e in data['evaluations'] if e['id'] == evaluation_id)


# Schema and file integrity

def test_research_file_matches_schema():
    assert load_report(ROOT / 'data' / 'research.json') == REPORT


def test_published_json_schema_is_current():
    assert SCHEMA_PATH.read_text() == json_schema_text(), 'Run `python research.py` to regenerate data/research.schema.json'


def test_validation_errors_are_caught_by_the_app_loader():
    assert issubclass(ValidationError, ValueError)  # app.py handles (OSError, ValueError)


REJECTED = [
    ('recall disagrees with counts', lambda d: evaluation(d, 'original_test').update(recall=0.5), 'recall 0.5 does not match'),
    ('lift disagrees with counts', lambda d: evaluation(d, 'original_test').update(lift=9.0), 'lift 9.0 does not match'),
    ('captured exceeds positives', lambda d: evaluation(d, 'original_test').update(positives_captured=500), 'cases captured cannot exceed'),
    ('wrong top-10% row count', lambda d: evaluation(d, 'original_test').update(targeted_rows=1000), 'top-10% rule'),
    ('purged id with original protocol', lambda d: evaluation(d, 'purged_test').update(protocol='original'), 'does not match protocol'),
    ('purged run trains on overlapping windows', lambda d: evaluation(d, 'purged_test').update(last_training_snapshot='1998-06-01'), 'must not train on outcome windows'),
    ('original run without overlap', lambda d: evaluation(d, 'original_test').update(last_training_snapshot='1998-01-01'), 'defined by its overlapping'),
    ('purging removes no rows', lambda d: evaluation(d, 'purged_test').update(training_rows=69789), 'purging must remove'),
    ('evaluation rows differ from split', lambda d: d['splits'][2].update(rows=13178), 'differ from the test split'),
    ('original training rows differ from splits', lambda d: d['splits'][0].update(rows=43960), 'train plus validation'),
    ('missing evaluation', lambda d: d['evaluations'].pop(), 'each of original_test'),
    ('duplicate evaluation', lambda d: d['evaluations'].append(copy.deepcopy(d['evaluations'][0])), 'each of original_test'),
    ('unknown field', lambda d: d.update(surprise=1), 'Extra inputs'),
    ('wrong schema version', lambda d: d.update(schema_version=1), 'Input should be 2'),
    ('bad dataset hash', lambda d: d.update(dataset_sha256='abc'), 'String should match'),
    ('monthly accounts do not sum', lambda d: d['months'][0].update(accounts=4392, observed_disengagement_rate=81 / 4392), 'sum to the test rows'),
    ('monthly cases do not sum', lambda d: d['months'][0].update(observed_disengagement_rate=59 / 4391), 'sum to the test positives'),
    ('monthly rate is not whole cases', lambda d: d['months'][0].update(observed_disengagement_rate=0.0187), 'whole number of cases'),
    ('latest snapshot differs from final month', lambda d: d.update(high_risk_accounts=1), 'final monthly cohort'),
    ('test period differs from split', lambda d: d.update(test_period=['1998-07-01', '1998-12-01']), 'test period must equal'),
    ('splits overlap in time', lambda d: d['splits'][1].update(period=['1997-06-01', '1998-06-01']), 'chronological'),
    ('curve row count wrong', lambda d: d['validation_analysis']['targeting_curve']['points'][0].update(targeted_rows=100), 'do not match the validation split'),
    ('curve precision wrong', lambda d: d['validation_analysis']['targeting_curve']['points'][1].update(precision=0.9), 'not consistent with its counts'),
    ('curve not monotonic', lambda d: d['validation_analysis']['targeting_curve']['points'].reverse(), 'widen monotonically'),
    ('curve labeled as test', lambda d: d['validation_analysis']['targeting_curve'].update(split='test'), "Input should be 'validation'"),
    ('feature missing from importance', lambda d: d['validation_analysis']['feature_importance']['rows'].pop(), 'exactly once'),
    ('importance not sorted', lambda d: d['validation_analysis']['feature_importance']['rows'].reverse(), 'sorted from most'),
    ('comparison disagrees with curve', lambda d: d['validation_analysis']['model_comparison']['models'][2].update(top10_precision=0.2), 'disagree on validation top-10% precision'),
    ('comparison lift disagrees with precision', lambda d: d['validation_analysis']['model_comparison']['models'][0].update(top10_lift=9.0), 'lift does not match its precision'),
]


@pytest.mark.parametrize('name, change, message', REJECTED, ids=[r[0] for r in REJECTED])
def test_inconsistent_data_is_rejected_for_the_right_reason(name, change, message):
    with pytest.raises(ValidationError, match=re.escape(message)):
        Research.model_validate(broken(change))


# Original and purged evaluations stay distinguishable

def test_protocols_are_explicit_and_distinct():
    assert (original.protocol, original.split) == ('original', 'test')
    assert (purged.protocol, purged.split) == ('purged_temporal', 'test')
    assert research.evaluation('purged_validation').split == 'validation'
    assert original.outcome_windows_overlap and not purged.outcome_windows_overlap
    assert not research.evaluation('purged_validation').outcome_windows_overlap
    assert set(research.protocols) == {'original', 'purged_temporal'}


def test_purged_evaluation_is_more_conservative_and_uses_fewer_rows():
    assert purged.training_rows < original.training_rows
    assert purged.roc_auc < original.roc_auc and purged.lift < original.lift
    assert (purged.rows, purged.positives) == (original.rows, original.positives)


def test_derived_fields_are_declared_and_match_the_published_rule():
    assert original.derived_fields == [] and not original.values_rounded
    for e in (purged, research.evaluation('purged_validation')):
        assert e.values_rounded and e.derived_fields == ['targeted_rows']
        assert e.targeted_rows == math.ceil(round(e.rows * 0.1, 6))
        # The derived row count reproduces the published precision to its four decimals.
        assert round(e.positives_captured / e.targeted_rows, 4) == e.precision


def test_split_counts_and_prevalence():
    assert [(s.name, s.rows, s.positives) for s in research.splits] == [
        ('train', 43959, 1131), ('validation', 25830, 737), ('test', 13177, 199)]
    assert research.split('test').prevalence == pytest.approx(0.0151, abs=5e-5)
    assert sum(m.accounts for m in research.months) == 13177
    assert [m.cases for m in research.months] == [81, 59, 59]


def test_pooled_ranking_and_monthly_cohorts_are_different_policies():
    # The caption on the Overview page says so; the counts differ by rounding of the per-month ceil rule.
    monthly = sum(m.high_risk for m in research.months)
    assert monthly != original.targeted_rows
    assert abs(monthly - original.targeted_rows) <= len(research.months)


# Validation-split data prepared for future charts

def test_validation_analysis_is_only_validation_split():
    analysis = research.validation_analysis
    for block in (analysis.targeting_curve, analysis.feature_importance, analysis.risk_segment_profile, analysis.model_comparison):
        assert (block.protocol, block.split) == ('original', 'validation')


def test_targeting_curve_policy_point_matches_the_research_readme_and_calculator():
    from core import campaign
    point = next(p for p in research.validation_analysis.targeting_curve.points if p.target_fraction == 0.1)
    assert (point.targeted_rows, point.positives_captured) == (2583, 457)
    assert round(point.precision, 4) == 0.1769 and round(point.lift, 2) == 6.20
    # The research README reports about 56.52 value units per save at 2.00 cost and a 20% save rate.
    scenario = campaign(point.targeted_rows, point.precision, 0.20, 2.0, 100.0)
    assert round(scenario['break_even'], 2) == 56.52


def test_feature_importance_covers_every_model_feature():
    rows = research.validation_analysis.feature_importance.rows
    assert len(rows) == research.feature_count == 100
    assert rows[0].feature == 'cash_withdrawals_90d'


def test_validation_exports_match_the_research_repository_when_available():
    import import_validation_exports as importer
    if not importer.DEFAULT_EXPORTS.is_dir():
        pytest.skip('churn-reactivation-engine exports are not checked out next to this repository')
    blocks = importer.build_blocks(importer.DEFAULT_EXPORTS)
    assert {name: REPORT['validation_analysis'][name] for name in blocks} == blocks


# README consistency

def readme_value(pattern):
    match = re.search(pattern, README)
    assert match, f'README no longer matches: {pattern}'
    return match.groups()


def test_readme_results_table_matches_data():
    e = original
    assert readme_value(r'\| ROC-AUC \| ([\d.]+) \| ([\d.]+) \|') == (f'{e.roc_auc:.4f}', f'{purged.roc_auc:.4f}')
    assert readme_value(r'\| Average precision \| ([\d.]+) \| ([\d.]+) \|') == (f'{e.pr_auc:.4f}', f'{purged.pr_auc:.4f}')
    assert readme_value(r'\| Precision in the top 10% \| ([\d.]+)% \| ([\d.]+)% \|') == (f'{e.precision:.2%}'[:-1], f'{purged.precision:.2%}'[:-1])
    assert readme_value(r'\| Top-10% lift \| ([\d.]+)× \| ([\d.]+)× \|') == (f'{e.lift:.2f}', f'{purged.lift:.2f}')
    for ev in (e, purged):
        assert f'{ev.positives_captured} of {ev.positives} ({ev.recall:.2%})' in README
    assert f'| Training rows | {e.training_rows:,} | {purged.training_rows:,} |' in README
    assert f'| Last training snapshot | {e.last_training_snapshot} | {purged.last_training_snapshot} |' in README


def test_readme_sample_sizes_and_prose_match_data():
    e = original
    assert readme_value(r'\*\*([\d,]+) held-out account-month observations\*\*, including \*\*(\d+) disengagement cases\*\*') == (f'{e.rows:,}', str(e.positives))
    assert readme_value(r'puts (\d+) of the (\d+) observed cases in the top ([\d,]+) observations') == (str(e.positives_captured), str(e.positives), f'{e.targeted_rows:,}')
    assert readme_value(r'purged locked test retained ROC-AUC ([\d.]+) and top-10% lift ([\d.]+)x') == (f'{purged.roc_auc:.4f}', f'{purged.lift:.2f}')


def test_documentation_avoids_em_dashes():
    for path in (ROOT / 'README.md', ROOT / 'visual' / 'README.md'):
        assert '\u2014' not in path.read_text(), f'em dash in {path.name}'


# In-app research notes

NOTES = {item['id']: item['text'] for item in KNOWLEDGE}


def test_research_notes_are_well_formed():
    assert len(NOTES) == len(KNOWLEDGE)
    for item in KNOWLEDGE:
        assert set(item) == {'id', 'title', 'text', 'source', 'vector'}
        assert all(item[k].strip() for k in ('title', 'text', 'source'))
        assert len(item['vector']) == 384
        assert math.sqrt(sum(v * v for v in item['vector'])) == pytest.approx(1.0, abs=1e-3)


def test_performance_note_matches_data():
    text, e = NOTES['performance'], original
    for fragment in (f'ROC-AUC {e.roc_auc:.4f}', f'PR-AUC {e.pr_auc:.4f}', f'lift {e.lift:.2f}x', f'{e.targeted_rows:,} of {e.rows:,}',
                     f'captured {e.positives_captured} of {e.positives} cases ({e.recall:.2%} recall)', f'Precision was {e.precision:.2%}'):
        assert fragment in text, fragment


def test_counts_and_training_notes_match_data():
    latest = research.months[-1]
    assert f'In the {latest.snapshot} snapshot there are {latest.accounts:,} eligible accounts' in NOTES['counts']
    assert f'highest-risk 10% contains {latest.high_risk} accounts' in NOTES['counts']
    assert f'{research.feature_count} features and {original.training_rows:,}' in NOTES['training']


def test_limitations_notes_quote_the_purged_result():
    for text in (NOTES['limits'], *research.limitations):
        if 'purged' in text:
            assert f'ROC-AUC {purged.roc_auc:.4f}' in text and f'lift {purged.lift:.2f}x' in text
    assert any('purged' in text for text in research.limitations)


def test_behavior_notes_match_latest_snapshot_profile():
    for row in research.latest_snapshot_profile:
        text = NOTES[row.feature]
        assert f'median {row.high_risk_median:,.2f}, compared with {row.other_median:,.2f}' in text
