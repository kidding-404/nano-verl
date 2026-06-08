from __future__ import annotations

from typing import Any

from math_verify import LatexExtractionConfig, parse, verify


_BOXED_EXTRACTION_CONFIG = [LatexExtractionConfig()]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any,
) -> float:
    _ = data_source, extra_info
    pred = parse(solution_str, extraction_config=_BOXED_EXTRACTION_CONFIG, fallback_mode="no_fallback")
    gold = parse(str(ground_truth))
    if not pred or not gold:
        return 0.0
    return 1.0 if verify(gold, pred) else 0.0
