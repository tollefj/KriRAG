import json
import logging
import os
from collections import defaultdict
from typing import Literal

import pandas as pd

CASETYPE = Literal["open-case"]


def get_keys(investigation: CASETYPE = "open-case") -> str:
    match investigation:
        case "open-case":
            return "dokument", "RAG QUERY"


def load_truth(investigation: CASETYPE = "open-case") -> dict:
    truth_path = f"data/ground_truth/{investigation}.csv"
    if investigation == "open-case":
        truth_df = pd.read_csv(truth_path, index_col=0).reset_index(drop=True)
    else:
        truth_df = pd.read_csv(truth_path)

    truth_dict = {}
    doc_key, rag_key = get_keys(investigation)
    truth_df = truth_df.dropna(subset=[rag_key])

    for rag_query in truth_df[rag_key].unique():
        tmp = truth_df[truth_df[rag_key] == rag_query][doc_key].values.tolist()
        tmp = [x for x in tmp if pd.notna(x)]
        if len(tmp) == 0:
            continue
        truth_dict[rag_query] = tmp

    # sort on information-need (IB1.2, IB3.3, etc.)
    truth_dict = dict(sorted(truth_dict.items()))
    return truth_dict


# simple fix to find the most likely query (in the ground truth)
def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_metrics(ground_truth, model_predictions, decimals=3, k=-1):
    ground_truth_set = set(ground_truth)
    model_predictions_set = set(model_predictions)

    TP = len(ground_truth_set & model_predictions_set)
    FP = len(model_predictions_set - ground_truth_set)
    FN = len(ground_truth_set - model_predictions_set)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    postfix = f"@{k}" if k > 0 else ""

    if k > 0:
        metrics = {
            f"P{postfix}": round(precision, decimals),
            f"R{postfix}": round(recall, decimals),
        }
    else:
        metrics = {
            f"P{postfix}": round(precision, decimals),
            f"R{postfix}": round(recall, decimals),
            f"F1{postfix}": round(f1, decimals),
            # f"Acc{postfix}": round(accuracy, decimals),
        }
    return metrics


def case_metrics(
    investigation: CASETYPE,
    min_score=0,  # filter on score (0, 1, 2 or 3 - low to high)
    top_k=-1,  # top k results to consider
    replace_above_k=True,  # whether to replace all values above k with -1
    decimals=2,
    root_folder=".",
    verbose: bool = False,
):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.CRITICAL)
    if verbose:
        logger.setLevel(logging.INFO)

    truth_dict = load_truth(investigation)
    all_metrics = {}

    for folder in sorted(os.listdir(root_folder)):
        folder = os.path.join(root_folder, folder)
        if not investigation.lower() in folder.lower() or not os.path.isdir(folder):
            logger.warning(f"Skipping folder: {folder}")
            continue
        logger.info(f"Processing folder: {folder}")
        output_folder = folder
        llm_answers = {}
        for llm_output in os.listdir(output_folder):
            llm_output_path = os.path.join(output_folder, llm_output)
            with open(llm_output_path, "r", encoding="utf-8") as f:
                tmp = []
                for line in f.readlines():
                    tmp.append(json.loads(line))
                llm_answers[llm_output] = tmp

        folder_metrics = {}

        for llm_file, llm_answer in llm_answers.items():
            logger.info(f"Processing file: {llm_file}")
            logger.info(f"Total answers: {len(llm_answer)}")
            valid_answers = []
            for answer in llm_answer:
                if "llm_output" in answer:
                    if not isinstance(answer["llm_output"], dict):
                        continue
                    if (
                        "score" in answer["llm_output"].keys()
                        and answer["llm_output"]["score"] >= min_score
                    ):
                        valid_answers.append(answer)

            # now filter on the top_k results
            if top_k > 0:
                valid_answers = valid_answers[:top_k]

            llm_answer = valid_answers
            if len(llm_answer) == 0:
                logger.warning(f"No answers found for file: {llm_file}")
                continue
            found_ids = [x["id"] for x in llm_answer]
            # from FILE_123 -> 123.
            # This depends on how you format your data!
            if investigation == "open-case":
                found_ids = sorted([int(x.split("_")[-1]) for x in found_ids])
            else:
                filter_case_id = lambda x: x.split("-")[0].strip()
                found_ids = sorted([filter_case_id(x) for x in found_ids])

            # hacky method to detect the ground-truth query from a user query
            query = llm_answer[0]["query"]
            best_query = None
            best_query_dist = 999
            for ground_truth_query in truth_dict.keys():
                dist = levenshtein(query, ground_truth_query)
                if dist < best_query_dist:
                    best_query_dist = dist
                    best_query = ground_truth_query

            ground_truth_matches = truth_dict[best_query]
            metrics = get_metrics(
                ground_truth_matches, found_ids, k=top_k, decimals=decimals
            )
            if replace_above_k and len(llm_answer) < top_k:
                metrics = {k: -1 for k in metrics}

            folder_metrics[query] = {
                "query": query,
                "IB_query": best_query,
                "metrics": metrics,
                "num_retrieved": len(found_ids),
            }
        all_metrics[output_folder] = folder_metrics
    return all_metrics
