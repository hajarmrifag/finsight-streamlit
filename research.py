"""Validated schema for the aggregate research artifacts that FinSight displays.

`data/research.json` is the only place the app reads model results from. Loading it through
`load_report` rejects malformed files and internally inconsistent numbers (for example a
recall that does not equal cases captured divided by cases), so a bad edit fails at load time
and in the test suite instead of reaching the interface.

Two evaluation protocols are kept apart by design:

* `original`: the development period is every snapshot before the evaluation period. The last
  training snapshots have 90-day outcome windows that extend into the evaluation period.
* `purged_temporal`: training snapshots whose 90-day outcome window extends into the
  evaluation period are removed first.
"""
import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 2
OUTCOME_DAYS = 90
TARGET_FRACTION = 0.10


def targeted_count(rows, fraction):
    """Rows in the top `fraction` of a ranking, using the research pipeline's ceil rule."""
    return math.ceil(round(rows * fraction, 6))


def _ordered(period):
    if period[0] > period[1]:
        raise ValueError('period start must not be after its end')
    return period


DateRange = Annotated[tuple[date, date], AfterValidator(_ordered)]
Share = Annotated[float, Field(ge=0, le=1)]
Count = Annotated[int, Field(ge=0)]
PositiveCount = Annotated[int, Field(gt=0)]


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class Split(Strict):
    name: Literal['train', 'validation', 'test']
    period: DateRange
    rows: PositiveCount
    positives: Count

    @model_validator(mode='after')
    def positives_fit(self):
        if self.positives > self.rows:
            raise ValueError('positives cannot exceed rows')
        return self

    @property
    def prevalence(self):
        return self.positives / self.rows


EVALUATION_KEYS = {
    'original_test': ('original', 'test'),
    'purged_test': ('purged_temporal', 'test'),
    'purged_validation': ('purged_temporal', 'validation'),
}


class Evaluation(Strict):
    """One model evaluation. `id` states the protocol and split so results cannot be mixed up."""
    id: Literal['original_test', 'purged_test', 'purged_validation']
    protocol: Literal['original', 'purged_temporal']
    split: Literal['validation', 'test']
    evaluation_period: DateRange
    training_rows: PositiveCount
    last_training_snapshot: date
    rows: PositiveCount
    positives: Count
    roc_auc: Share
    pr_auc: Share
    targeted_fraction: Literal[0.1]
    targeted_rows: PositiveCount
    positives_captured: Count
    precision: Share
    recall: Share
    lift: Annotated[float, Field(gt=0)]
    values_rounded: bool
    derived_fields: list[Literal['targeted_rows']] = []
    source: Annotated[str, Field(min_length=1)]

    @property
    def prevalence(self):
        return self.positives / self.rows

    @property
    def outcome_windows_overlap(self):
        """True when the last training snapshot's outcome window reaches into the evaluation period."""
        return self.last_training_snapshot + timedelta(days=OUTCOME_DAYS) > self.evaluation_period[0]

    @model_validator(mode='after')
    def consistent(self):
        if EVALUATION_KEYS[self.id] != (self.protocol, self.split):
            raise ValueError(f'id {self.id} does not match protocol {self.protocol} and split {self.split}')
        if self.protocol == 'purged_temporal' and self.outcome_windows_overlap:
            raise ValueError('a purged evaluation must not train on outcome windows that overlap the evaluation period')
        if self.protocol == 'original' and not self.outcome_windows_overlap:
            raise ValueError('the original protocol is defined by its overlapping outcome windows')
        if not self.positives_captured <= min(self.positives, self.targeted_rows) <= self.rows:
            raise ValueError('cases captured cannot exceed the cases present or the rows targeted')
        if self.targeted_rows != targeted_count(self.rows, self.targeted_fraction):
            raise ValueError('targeted rows do not match the top-10% rule')
        # Published values are rounded (4 decimals for rates, 2 for lift); full-precision values are exact.
        rate_tol, lift_tol = (5.1e-5, 5.1e-3) if self.values_rounded else (1e-9, 1e-9)
        checks = {
            'precision': (self.precision, self.positives_captured / self.targeted_rows, rate_tol),
            'recall': (self.recall, self.positives_captured / self.positives, rate_tol),
            'lift': (self.lift, (self.positives_captured / self.targeted_rows) / self.prevalence,
                     lift_tol + (rate_tol / self.prevalence if self.values_rounded else 0)),
        }
        for name, (stated, derived, tol) in checks.items():
            if abs(stated - derived) > tol:
                raise ValueError(f'{name} {stated} does not match the counts ({derived:.6f})')
        return self


class Month(Strict):
    snapshot: date
    accounts: PositiveCount
    high_risk: PositiveCount
    observed_disengagement_rate: Share

    @model_validator(mode='after')
    def consistent(self):
        if self.high_risk != targeted_count(self.accounts, TARGET_FRACTION):
            raise ValueError('high-risk accounts do not match the top-10% rule')
        cases = self.observed_disengagement_rate * self.accounts
        if abs(cases - round(cases)) > 1e-6:
            raise ValueError('observed rate does not correspond to a whole number of cases')
        return self

    @property
    def cases(self):
        return round(self.observed_disengagement_rate * self.accounts)


class ProfileRow(Strict):
    feature: Annotated[str, Field(min_length=1)]
    high_risk_median: float
    other_median: float


class TargetingPoint(Strict):
    target_fraction: Annotated[float, Field(gt=0, le=1)]
    targeted_rows: PositiveCount
    score_threshold: Share
    precision: Share
    recall: Share
    lift: Annotated[float, Field(gt=0)]
    positives_captured: Count


class TargetingCurve(Strict):
    """Precision, recall and lift at several targeting depths, on the validation split only."""
    protocol: Literal['original']
    split: Literal['validation']
    points: Annotated[list[TargetingPoint], Field(min_length=2)]
    source: Annotated[str, Field(min_length=1)]


class FeatureImportanceRow(Strict):
    feature: Annotated[str, Field(min_length=1)]
    importance_mean: float
    importance_std: Annotated[float, Field(ge=0)]


class FeatureImportance(Strict):
    """Permutation importance: the drop in validation average precision when a feature is shuffled."""
    protocol: Literal['original']
    split: Literal['validation']
    metric: Literal['average_precision_decrease']
    repeats: PositiveCount
    rows: Annotated[list[FeatureImportanceRow], Field(min_length=1)]
    source: Annotated[str, Field(min_length=1)]


class SegmentProfileRow(Strict):
    feature: Annotated[str, Field(min_length=1)]
    high_risk_median: float
    other_median: float
    high_risk_mean: float
    other_mean: float
    standardized_mean_diff: float


class RiskSegmentProfile(Strict):
    """Top-10% risk group against the other 90% on the validation split. Associations only."""
    protocol: Literal['original']
    split: Literal['validation']
    rows: Annotated[list[SegmentProfileRow], Field(min_length=1)]
    source: Annotated[str, Field(min_length=1)]


class ModelScore(Strict):
    model: Literal['Logistic Regression', 'Random Forest', 'Gradient Boosting']
    roc_auc: Share
    pr_auc: Share
    top10_precision: Share
    top10_lift: Annotated[float, Field(gt=0)]


class ModelComparison(Strict):
    """Candidate models scored on the validation split. Rounded as published in the research README."""
    protocol: Literal['original']
    split: Literal['validation']
    values_rounded: Literal[True]
    models: Annotated[list[ModelScore], Field(min_length=2)]
    source: Annotated[str, Field(min_length=1)]


class ValidationAnalysis(Strict):
    """Research outputs from the validation split. These are never test-set results."""
    targeting_curve: TargetingCurve
    feature_importance: FeatureImportance
    risk_segment_profile: RiskSegmentProfile
    model_comparison: ModelComparison


class Research(Strict):
    schema_version: Literal[2]
    model: str
    model_status: Literal['trained']
    dataset: str
    source_url: Annotated[str, Field(pattern=r'^https://')]
    dataset_sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    research_origin: str
    feature_count: PositiveCount
    development_period: DateRange
    test_period: DateRange
    protocols: dict[Literal['original', 'purged_temporal'], Annotated[str, Field(min_length=1)]]
    splits: list[Split]
    evaluations: list[Evaluation]
    latest_snapshot: date
    latest_accounts: PositiveCount
    high_risk_accounts: PositiveCount
    months: Annotated[list[Month], Field(min_length=1)]
    latest_snapshot_profile: Annotated[list[ProfileRow], Field(min_length=1)]
    target_definition: Annotated[str, Field(min_length=1)]
    limitations: Annotated[list[str], Field(min_length=1)]
    validation_analysis: ValidationAnalysis

    def evaluation(self, evaluation_id):
        return next(e for e in self.evaluations if e.id == evaluation_id)

    def split(self, name):
        return next(s for s in self.splits if s.name == name)

    @model_validator(mode='after')
    def consistent(self):
        splits = {s.name: s for s in self.splits}
        if [s.name for s in self.splits] != ['train', 'validation', 'test']:
            raise ValueError('splits must be train, validation, test in order')
        if not splits['train'].period[1] < splits['validation'].period[0] <= splits['validation'].period[1] < splits['test'].period[0]:
            raise ValueError('splits must be chronological and must not overlap')
        if self.development_period != (splits['train'].period[0], splits['validation'].period[1]):
            raise ValueError('development period must span the train and validation splits')
        if self.test_period != splits['test'].period:
            raise ValueError('test period must equal the test split')

        evaluations = {e.id: e for e in self.evaluations}
        if len(evaluations) != len(self.evaluations) or set(evaluations) != set(EVALUATION_KEYS):
            raise ValueError('evaluations must contain each of original_test, purged_test, purged_validation once')
        if set(self.protocols) != {'original', 'purged_temporal'}:
            raise ValueError('both evaluation protocols must be described')
        for e in self.evaluations:
            split = splits[e.split]
            if (e.rows, e.positives, e.evaluation_period) != (split.rows, split.positives, split.period):
                raise ValueError(f'{e.id} rows, positives or period differ from the {e.split} split')
        original, purged_test, purged_validation = (evaluations[k] for k in ('original_test', 'purged_test', 'purged_validation'))
        if original.training_rows != splits['train'].rows + splits['validation'].rows:
            raise ValueError('original training rows must equal train plus validation rows')
        if original.last_training_snapshot != self.development_period[1]:
            raise ValueError('original training must end at the end of the development period')
        if not purged_test.training_rows < original.training_rows:
            raise ValueError('purging must remove training rows')
        if not purged_validation.training_rows < splits['train'].rows:
            raise ValueError('purging must remove training rows')

        if [m.snapshot for m in self.months] != sorted({m.snapshot for m in self.months}):
            raise ValueError('months must be unique and in order')
        if not all(self.test_period[0] <= m.snapshot <= self.test_period[1] for m in self.months):
            raise ValueError('monthly cohorts must fall inside the test period')
        if sum(m.accounts for m in self.months) != original.rows:
            raise ValueError('monthly accounts must sum to the test rows')
        if sum(m.cases for m in self.months) != original.positives:
            raise ValueError('monthly cases must sum to the test positives')
        last = self.months[-1]
        if (self.latest_snapshot, self.latest_accounts, self.high_risk_accounts) != (last.snapshot, last.accounts, last.high_risk):
            raise ValueError('latest snapshot figures must match the final monthly cohort')

        analysis = self.validation_analysis
        validation = splits['validation']
        points = analysis.targeting_curve.points
        previous = None
        for p in points:
            if p.targeted_rows != targeted_count(validation.rows, p.target_fraction):
                raise ValueError(f'targeted rows at {p.target_fraction} do not match the validation split')
            if not (abs(p.precision - p.positives_captured / p.targeted_rows) < 1e-9
                    and abs(p.recall - p.positives_captured / validation.positives) < 1e-9
                    and abs(p.lift - p.precision / validation.prevalence) < 1e-9):
                raise ValueError(f'targeting point {p.target_fraction} is not consistent with its counts')
            if previous and not (p.target_fraction > previous.target_fraction
                                 and p.positives_captured >= previous.positives_captured
                                 and p.score_threshold <= previous.score_threshold):
                raise ValueError('targeting points must widen monotonically')
            previous = p
        names = [row.feature for row in analysis.feature_importance.rows]
        if len(names) != len(set(names)) or len(names) != self.feature_count:
            raise ValueError('feature importance must list each model feature exactly once')
        importances = [row.importance_mean for row in analysis.feature_importance.rows]
        if importances != sorted(importances, reverse=True):
            raise ValueError('feature importance must be sorted from most to least important')
        gradient = next((m for m in analysis.model_comparison.models if m.model == 'Gradient Boosting'), None)
        at_policy = next((p for p in points if p.target_fraction == TARGET_FRACTION), None)
        if gradient and at_policy and abs(gradient.top10_precision - at_policy.precision) > 5.1e-5:
            raise ValueError('model comparison and targeting curve disagree on validation top-10% precision')
        for m in analysis.model_comparison.models:
            if abs(m.top10_lift - m.top10_precision / validation.prevalence) > 5.1e-3 + 5.1e-5 / validation.prevalence:
                raise ValueError(f'{m.model} lift does not match its precision')
        return self


def load_report(path):
    """Read and validate the research file; returns the plain JSON dict the app renders from."""
    raw = json.loads(Path(path).read_text())
    Research.model_validate(raw)
    return raw


SCHEMA_PATH = Path(__file__).resolve().parent / 'data' / 'research.schema.json'


def json_schema_text():
    return json.dumps(Research.model_json_schema(), indent=2, sort_keys=True) + '\n'


if __name__ == '__main__':  # regenerate the published JSON Schema: python research.py
    SCHEMA_PATH.write_text(json_schema_text())
    print(f'Wrote {SCHEMA_PATH}')
