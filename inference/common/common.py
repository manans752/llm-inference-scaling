"""Backward-compatible exports for legacy scripts."""

from inference.common.evaluation import (
    answers_match,
    evaluate_output,
    extract_final_answer,
    is_valid_answer,
    low_confidence,
    majority_vote,
    normalize_text as normalise,
)

