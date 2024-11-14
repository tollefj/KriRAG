# ------------------------------------------------------------------------------
# File: llm.py
# Description: llm tools for KriRAG. Relies on a openai-comptaible api, default through llama.cpp server docker container.
#
# License: Apache License 2.0
# For license details, refer to the LICENSE file in the project root.
#
# Contributors:
# - Tollef Jørgensen (Initial Development, 2024)
# ------------------------------------------------------------------------------

import json
import re

import requests

HOSTNAME = "0.0.0.0"
PORT = 8000
url = f"http://{HOSTNAME}:{PORT}/completions/v1"  # llama.cpp server
headers = {"Content-Type": "application/json"}

schema = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
        "reason": {"type": "string"},
        "score": {"type": "integer", "enum": [0, 1, 2, 3]},
        "summary": {"type": "string"},
    },
    "required": ["questions", "reason", "score", "summary"],
}

schema_summ = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
    },
    "required": ["summary"],
}

schema_findings = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
        },
        "references": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": ["summary", "references"],
}


schemas = {
    "default": schema,
    "summary": schema_summ,
    "findings": schema_findings,
}


def pred(
    instruction,
    max_tokens=1000,
    use_schema: str = "default",
    temp=0.0,  # temperature. 0: deterministic, 1+: random
    # min_p=0.1,  # minimum probability
    # max_p=0.9,  # maximum probability
    # top_p=0.9,  # nucleus sampling
    # top_k=40,  # consider top k tokens at each generation step
    evaluate: bool = False,  # apply eval
):
    if len(instruction) == 0:
        raise ValueError("Instruction cannot be empty")

    data = {
        "prompt": instruction,
        "n_predict": max_tokens,
        "temperature": temp,
        "repeat_penalty": 1.2,  # 1.1 default,
    }
    if use_schema:
        data["json_schema"] = schemas[use_schema]

    response = requests.post(url, headers=headers, data=json.dumps(data)).json()
    response = response["content"]
    if evaluate:
        return parse_llm_output(response)
    return response


def parse_llm_output(response: str):
    if not response:
        return response
    # spacing!
    response = response.replace("\n", " ")
    response = response.replace("\t", " ")
    response = re.sub(r"\s+", " ", response)
    response = response.strip()
    # markdown ticks
    response = response.replace("```python", "")
    response = response.replace("```json", "")
    response = response.replace("```", "")

    response = response.replace("false", "False")
    response = response.replace("true", "True")
    response = response.replace("null", "None")

    obj = eval(response)
    if isinstance(obj, dict):
        # unify keys in case of capitalization.
        obj = {k.lower(): v for k, v in obj.items()}
    return obj


def ask_llm(
    query: str,
    text: str,
    extra: str = "",
    doc_id: str = "ID",
    temp: float = 0.0,
    tokens: int = 150,
    prompt_source: dict = None,  # which dict in "llm_config"
    lang: str = "en",
    verbose: bool = False,
) -> dict:
    text = re.sub(r"\.{3,}", "...", text)

    if extra:
        extra = f"You have info from previous interrogations: '{extra}'. Use this info to guide your reasoning if relevant."

    instruction = prompt_source[lang].format(
        query=query,
        text=text,
        extra=extra,
        doc_id=doc_id,
    )
    if verbose:
        print("Instruction", instruction)

    output = pred(
        instruction=instruction,
        temp=temp,
        max_tokens=tokens,
        use_schema="default",
    )
    if verbose:
        print("-*-" * 40)
        print(output)
        print("-*-" * 40)
    try:
        output = parse_llm_output(output)
    except SyntaxError as e:
        print("SyntaxError. Returning raw output")
    return output
