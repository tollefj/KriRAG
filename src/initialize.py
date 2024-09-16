import os
from typing import List

import nltk
import pandas as pd

LANG = "norwegian"

valid_exts = [".txt", ".json", ".jsonl"]


def load_documents(
    doc_path: str,
    lang: str = LANG,
) -> pd.DataFrame:
    # iterate all files in the folder
    doc_paths = []
    for _file in os.listdir(doc_path):
        if _file.endswith(tuple(valid_exts)):
            doc_paths.append(os.path.join(doc_path, _file))

    parsed_data = []
    for dp in doc_paths:
        with open(dp, "r", encoding="utf-8") as f:
            _doc = f.readlines()
            parsed_data.extend(parse_document(_doc, lang))

    df = pd.DataFrame(parsed_data)
    return df


def load_single_document(docs: List[str], lang: str = LANG) -> pd.DataFrame:
    # docs is a list of strings/sentences
    parsed_data = parse_document(docs, lang)
    df = pd.DataFrame(parsed_data)
    return df


def parse_document(docs: List[str], lang: str = LANG) -> List[str]:
    docs = [d.replace("\n", "") for d in docs]

    def sentencize(text):
        return nltk.sent_tokenize(text, language=lang)

    parsed_data = []
    sentences = []
    paragraph_sent_map = []
    for d_id, paragraph in enumerate(docs):
        d_sents = sentencize(paragraph)
        paragraph_sent_map.extend([d_id] * len(d_sents))
        sentences.extend(d_sents)

    for s_id, sent in enumerate(sentences):
        _d_id = paragraph_sent_map[s_id]
        parsed_data.append(
            {
                "id": f"DOC_{_d_id + 1}",  # for LLM referencing
                "page_id": _d_id,
                "sent_id": s_id,
                "text": sent,
            }
        )

    return parsed_data
