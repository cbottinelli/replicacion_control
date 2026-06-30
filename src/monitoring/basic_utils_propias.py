import tiktoken
import re
from typing import Any

OAIChatLog = dict[str, Any]

def parse_first_float(text: str) -> float | None:
    match = re.search(r"\b\d+(?:\.\d+)?\b", text)
    if match is None:
        return None
    return float(match.group(0))

def get_encoder_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError as e:
        raise ValueError(
            f"Could not determine tokenizer for model {model!r}. "
            "Please specify an explicit encoder."
        ) from e

# Basada en single_tokenize del repo, pero parametrizando el modelo.
def single_tokenize(token: str, model: str) -> int:
    encoder = get_encoder_for_model(model)
    tokens = encoder.encode(token)
    assert len(tokens) == 1, (f"Expected single token, got {len(tokens)} for {token!r}")
    return tokens[0]

def get_yes_no(x: str):
    x = x.lower()
    y = "yes" in x
    n = "no" in x
    if y == n:
        return None
    return y

def add_dicts_safe(*dicts: dict):
    res = dict()
    for d in dicts:
        for k, v in d.items():
            if k in res:
                raise ValueError(f"Key {k} already in dict")
            res[k] = v
    return res