_MODEL_COSTS = {
    "gpt-4o":       {"input": 0.0025,  "output": 0.01},
    "gpt-4o-mini":  {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":  {"input": 0.01,    "output": 0.03},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    costs = _MODEL_COSTS.get(model, _MODEL_COSTS["gpt-4o"])
    return round((input_tokens * costs["input"] + output_tokens * costs["output"]) / 1000, 6)


def count_tokens_approx(text: str) -> int:
    return max(1, len(text) // 4)
