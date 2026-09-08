"""Evidence retrieval and scenario math shared by the interface and tests."""
import math
import re

STOP = set('a an the how what why which when is are was were of to for with in on at do does can could would should me tell about many much'.split())


def word_search(question, knowledge):
    tokens = set(re.findall(r'[a-z0-9]+', question.lower())) - STOP
    if not tokens:
        return []
    matches = []
    for item in knowledge:
        words = set(re.findall(r'[a-z0-9]+', (item['title'] + ' ' + item['text']).lower()))
        score = len(tokens & words) / len(tokens)
        if score >= .3:
            matches.append(dict(item, score=score))
    return sorted(matches, key=lambda m: m['score'], reverse=True)[:3]


def semantic_search(question, knowledge, encoder):
    vector = encoder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
    matches = []
    for item in knowledge:
        if len(vector) != len(item['vector']):
            raise ValueError('Embedding dimensions differ')
        score = float(sum(float(a) * b for a, b in zip(vector, item['vector'])))
        if score >= .3:
            matches.append(dict(item, score=score))
    return sorted(matches, key=lambda m: m['score'], reverse=True)[:3]


def campaign(targeted, precision, save_rate, contact_cost, retained_value):
    values = (targeted, precision, save_rate, contact_cost, retained_value)
    if not all(math.isfinite(v) for v in values) or targeted < 1 or targeted > 1_000_000 or int(targeted) != targeted:
        raise ValueError('Enter a whole number of accounts between 1 and 1,000,000.')
    if not (0 <= precision <= 1 and 0 <= save_rate <= 1 and 0 <= contact_cost <= 1_000_000 and 0 <= retained_value <= 10_000_000):
        raise ValueError('Enter valid scenario assumptions.')
    saves = targeted * precision * save_rate
    cost = targeted * contact_cost
    return dict(expected_saves=saves, cost=cost, net=saves*retained_value-cost,
                break_even=cost/saves if saves else None)
