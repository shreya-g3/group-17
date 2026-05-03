import json
import csv
import re
from pathlib import Path


LABEL_TYPES = ["participants", "interventions", "outcomes"]
STRATEGIES = {"0", "A", "B"}
DEFAULT_EPOCHS = 3


def remove_eval_suffix(stem):

    if stem.endswith("_eval"):
        return stem[:-5]
    if stem.endswith("eval"):
        return stem[:-4].rstrip("_")
    return stem


def parse_eval_filename(json_path, default_epochs=DEFAULT_EPOCHS):

    path = Path(json_path)
    stem = remove_eval_suffix(path.stem)

    meta = {
        "source_file": path.name,
        "run_name": stem,
        "pipeline": "",
        "strategy": "",
        "epoch": default_epochs,
        "label_type": "",
    }


    label_type = None
    prefix = None

    for lab in LABEL_TYPES:
        if stem.endswith(lab):
            label_type = lab
            prefix = stem[: -len(lab)].rstrip("_")
            break

    if label_type is None:
        raise ValueError(
            f"Cannot parse label_type from filename: {path.name}. "
            f"Expected one of {LABEL_TYPES} at the end."
        )

    meta["label_type"] = label_type


    rest = prefix.strip("_")


    m = re.fullmatch(r"(.+?)_?([0AB])_?([1-9]\d*)", rest)
    if m:
        meta["pipeline"] = m.group(1).rstrip("_")
        meta["strategy"] = m.group(2)
        meta["epoch"] = int(m.group(3))
        return meta


    m = re.fullmatch(r"(.+?)_?([0AB])", rest)
    if m:
        meta["pipeline"] = m.group(1).rstrip("_")
        meta["strategy"] = m.group(2)
        meta["epoch"] = default_epochs
        return meta


    m = re.fullmatch(r"(.+?)_?([1-9]\d*)", rest)
    if m:
        meta["pipeline"] = m.group(1).rstrip("_")
        meta["strategy"] = ""
        meta["epoch"] = int(m.group(2))
        return meta

    meta["pipeline"] = rest
    meta["strategy"] = ""
    meta["epoch"] = default_epochs
    return meta


def flatten_eval_json(data, file_meta):
    token = data.get("token_level", {})
    span = data.get("span_level", {})
    exact = span.get("exact", {})
    overlap = span.get("overlap", {})

    return {

        "pipeline": file_meta["pipeline"],
        "strategy": file_meta["strategy"],
        "epoch": file_meta["epoch"],
        "label_type": file_meta["label_type"],
        "source_file": file_meta["source_file"],
        "run_name": file_meta["run_name"],


        "docs": data.get("docs", 0),

 
        "token_accuracy": token.get("accuracy", 0.0),
        "token_macro_f1": token.get("macro avg", {}).get("f1-score", 0.0),
        "token_weighted_f1": token.get("weighted avg", {}).get("f1-score", 0.0),

        "B_precision": token.get("B", {}).get("precision", 0.0),
        "B_recall": token.get("B", {}).get("recall", 0.0),
        "B_f1": token.get("B", {}).get("f1-score", 0.0),

        "I_precision": token.get("I", {}).get("precision", 0.0),
        "I_recall": token.get("I", {}).get("recall", 0.0),
        "I_f1": token.get("I", {}).get("f1-score", 0.0),


        "exact_precision": exact.get("precision", 0.0),
        "exact_recall": exact.get("recall", 0.0),
        "exact_f1": exact.get("f1", 0.0),
        "exact_gold": exact.get("gold_spans", 0),
        "exact_pred": exact.get("pred_spans", 0),
        "exact_matched": exact.get("matched", 0),

        "overlap_precision": overlap.get("precision", 0.0),
        "overlap_recall": overlap.get("recall", 0.0),
        "overlap_f1": overlap.get("f1", 0.0),
        "overlap_gold": overlap.get("gold_spans", 0),
        "overlap_pred": overlap.get("pred_spans", 0),
        "overlap_matched": overlap.get("matched", 0),
    }


def export_csv(results_eval_dir="results_eval", output_csv="results_summary.csv"):
    results_eval_dir = Path(results_eval_dir)
    files = sorted(results_eval_dir.glob("*_eval.json"))

    if not files:
        print(f" No *_eval.json files found in {results_eval_dir}")
        return

    rows = []

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)

            file_meta = parse_eval_filename(f)
            row = flatten_eval_json(data, file_meta)
            rows.append(row)

        except Exception as e:
            print(f"[Skipped] {f.name}: {e}")

    if not rows:
        print("No valid rows exported.")
        return

    rows = sorted(
        rows,
        key=lambda x: (
            x["pipeline"],
            x["strategy"],
            x["epoch"],
            x["label_type"],
            x["source_file"],
        )
    )

    fieldnames = [
        "pipeline", "strategy", "epoch", "label_type",
        "source_file", "run_name", "docs",

        "token_accuracy", "token_macro_f1", "token_weighted_f1",
        "B_precision", "B_recall", "B_f1",
        "I_precision", "I_recall", "I_f1",

        "exact_precision", "exact_recall", "exact_f1",
        "exact_gold", "exact_pred", "exact_matched",

        "overlap_precision", "overlap_recall", "overlap_f1",
        "overlap_gold", "overlap_pred", "overlap_matched",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)




if __name__ == "__main__":
    export_csv(
        results_eval_dir="results_eval",
        output_csv="results_summary.csv"
    )