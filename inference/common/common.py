import re
from collections import Counter

MODEL_NAME = "microsoft/phi-2"

def extract_final_answer(text: str):
    match = re.search(r"Final Answer:\s*(.*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # fallback: skip junk lines like "Question:" or "Problem:"
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        if line.lower().startswith("question"):
            continue
        if line.lower().startswith("problem"):
            continue
        return line

    return None

def normalise(ans: str):  # yes = YES
    if ans is None:
        return None
    return ans.strip().lower()

def answers_match(pred: str, target_answer: str):  # ensure semantically equal answers are not marked false
    if pred is None or target_answer is None:
        return False

    pred = pred.strip().lower()
    target_answer = target_answer.strip().lower()

    # target_answer is numeric
    if target_answer.replace(".", "", 1).isdigit():
        # extract first number from prediction
        match = re.search(r"-?\d+(\.\d+)?", pred)
        if match:
            pred_num = match.group(0)
            return pred_num == target_answer
        return False

    # target_answer is text
    pred = re.sub(r"[^a-z0-9]", "", pred)
    target_answer = re.sub(r"[^a-z0-9]", "", target_answer)

    return pred == target_answer

def is_valid_answer(ans: str):
    if ans is None:
        return False

    ans = ans.strip().lower()

    # reject placeholders
    if "<answer>" in ans or "<option>" in ans:
        return False

    # reject reasoning-like outputs
    bad_starts = ["to find", "let x", "step", "a farmer", "question"]
    if any(ans.startswith(b) for b in bad_starts):
        return False

    return True

def majority_vote(predictions):
    predictions = [p for p in predictions if p is not None and p != ""]
    if len(predictions) == 0:
        return None

    vote = Counter(predictions)
    return vote.most_common(1)[0][0]

def low_confidence(output: str):
    ans = extract_final_answer(output)
    if ans is None:
        return True

    ans = ans.strip().lower()

    if len(ans) == 0:
        return True

    # placeholder / junk outputs
    bad_markers = [
        "<show",
        "<answer>",
        "<option>",
        "step",
        "solution",
        "divide",
        "let x",
        "|question_end|"
    ]
    if any(b in ans for b in bad_markers):
        return True

    # answer too long (rambling)
    if len(ans.split()) > 6:
        return True
