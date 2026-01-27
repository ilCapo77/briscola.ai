"""
Test per opponent mix (RL training).

Obiettivo:
- parsing robusto di `name:weight,...`
- normalizzazione e gestione duplicati
- campionamento riproducibile
"""

from __future__ import annotations

import numpy as np
import pytest

from briscola_ai.ai.training.opponent_mix import parse_opponent_mix, sample_opponent_name


def test_parse_opponent_mix_normalizes_and_merges_duplicates() -> None:
    items = parse_opponent_mix("heuristic_v1:2, random:1, heuristic_v1:1")
    # ordine stabile (sorted per name)
    assert [i.name for i in items] == ["heuristic_v1", "random"]
    probs = {i.name: i.prob for i in items}
    # heuristic_v1 weight=3, random weight=1 -> 0.75/0.25
    assert probs["heuristic_v1"] == pytest.approx(0.75)
    assert probs["random"] == pytest.approx(0.25)
    assert sum(i.prob for i in items) == pytest.approx(1.0)


def test_parse_opponent_mix_allows_implicit_weight() -> None:
    items = parse_opponent_mix("a,b")
    probs = [i.prob for i in items]
    assert probs[0] == pytest.approx(0.5)
    assert probs[1] == pytest.approx(0.5)


def test_parse_opponent_mix_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError):
        parse_opponent_mix("")
    with pytest.raises(ValueError):
        parse_opponent_mix("x:0")
    with pytest.raises(ValueError):
        parse_opponent_mix("x:-1")
    with pytest.raises(ValueError):
        parse_opponent_mix("x:not_a_number")


def test_sample_opponent_name_is_reproducible_with_seed() -> None:
    items = parse_opponent_mix("a:1,b:1,c:1")
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    seq1 = [sample_opponent_name(items, rng=rng1) for _ in range(20)]
    seq2 = [sample_opponent_name(items, rng=rng2) for _ in range(20)]
    assert seq1 == seq2
