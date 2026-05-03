import json
from pathlib import Path
from sklearn.metrics import classification_report


def load_result_json(json_path):
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = ["label_type", "gold_labels", "pred_labels"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"JSON 缺少必要字段: {key}")

    return data


def flatten_labels(labels):
    flat = []
    for doc in labels:
        flat.extend(doc)
    return flat


def labels_to_spans(doc_labels):

    spans = []
    start = None

    for i, lab in enumerate(doc_labels):
        if lab == "B":
            if start is not None:
                spans.append((start, i))
            start = i

        elif lab == "I":
            if start is None:
                start = i

        else:  # O
            if start is not None:
                spans.append((start, i))
                start = None

    if start is not None:
        spans.append((start, len(doc_labels)))

    return spans


def get_all_spans(all_labels):
    return [labels_to_spans(doc_labels) for doc_labels in all_labels]


def safe_divide(a, b):
    return a / b if b > 0 else 0.0


def compute_prf(num_match, num_pred, num_gold):
    precision = safe_divide(num_match, num_pred)
    recall = safe_divide(num_match, num_gold)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def evaluate_token_level(test_labels, pred_labels):
    y_true_flat = flatten_labels(test_labels)
    y_pred_flat = flatten_labels(pred_labels)

    report_dict = classification_report(
        y_true_flat,
        y_pred_flat,
        digits=4,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        y_true_flat,
        y_pred_flat,
        digits=4,
        zero_division=0
    )

    return report_dict, report_text


def evaluate_span_exact(test_labels, pred_labels):
    gold_spans_all = get_all_spans(test_labels)
    pred_spans_all = get_all_spans(pred_labels)

    gold_total = 0
    pred_total = 0
    exact_match = 0

    for gold_spans, pred_spans in zip(gold_spans_all, pred_spans_all):
        gold_set = set(gold_spans)
        pred_set = set(pred_spans)

        gold_total += len(gold_set)
        pred_total += len(pred_set)
        exact_match += len(gold_set & pred_set)

    precision, recall, f1 = compute_prf(exact_match, pred_total, gold_total)

    return {
        "gold_spans": gold_total,
        "pred_spans": pred_total,
        "matched": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def spans_overlap_with_threshold(span_pred, span_gold, threshold=0.5):
    s1, e1 = span_pred
    s2, e2 = span_gold

    inter = max(0, min(e1, e2) - max(s1, s2))
    gold_len = e2 - s2

    if gold_len == 0:
        return False

    overlap_ratio = inter / gold_len
    return overlap_ratio >= threshold


def evaluate_span_overlap(test_labels, pred_labels, threshold=0.5):
    gold_spans_all = get_all_spans(test_labels)
    pred_spans_all = get_all_spans(pred_labels)

    gold_total = 0
    pred_total = 0
    overlap_match = 0

    for gold_spans, pred_spans in zip(gold_spans_all, pred_spans_all):
        gold_total += len(gold_spans)
        pred_total += len(pred_spans)

        matched_gold = set()

        for pred_span in pred_spans:
            for gold_idx, gold_span in enumerate(gold_spans):
                if gold_idx in matched_gold:
                    continue

                if spans_overlap_with_threshold(pred_span, gold_span, threshold):
                    overlap_match += 1
                    matched_gold.add(gold_idx)
                    break

    precision, recall, f1 = compute_prf(overlap_match, pred_total, gold_total)

    return {
        "threshold": threshold,
        "gold_spans": gold_total,
        "pred_spans": pred_total,
        "matched": overlap_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_eval_result(data, overlap_threshold=0.5):
    label_type = data["label_type"]
    pipeline_name = data.get("pipeline_name", "unknown_pipeline")
    gold_labels = data["gold_labels"]
    pred_labels = data["pred_labels"]

    token_report_dict, token_report_text = evaluate_token_level(gold_labels, pred_labels)
    span_exact = evaluate_span_exact(gold_labels, pred_labels)
    span_overlap = evaluate_span_overlap(gold_labels, pred_labels, threshold=overlap_threshold)

    result = {
        "pipeline_name": pipeline_name,
        "label_type": label_type,
        "docs": len(gold_labels),
        "token_level": token_report_dict,
        "span_level": {
            "exact": span_exact,
            "overlap": span_overlap
        }
    }

    pretty_text = []
    pretty_text.append("=" * 60)
    pretty_text.append(f"Evaluating pipeline: {pipeline_name}")
    pretty_text.append(f"Label type        : {label_type}")
    pretty_text.append(f"Docs              : {len(gold_labels)}")
    pretty_text.append("=" * 60)

    pretty_text.append("\nClassification Report (token-level):")
    pretty_text.append(token_report_text)

    pretty_text.append("\nSpan-level EXACT match:")
    pretty_text.append(f"Gold spans      : {span_exact['gold_spans']}")
    pretty_text.append(f"Pred spans      : {span_exact['pred_spans']}")
    pretty_text.append(f"Exact matched   : {span_exact['matched']}")
    pretty_text.append(f"Precision       : {span_exact['precision']:.4f}")
    pretty_text.append(f"Recall          : {span_exact['recall']:.4f}")
    pretty_text.append(f"F1              : {span_exact['f1']:.4f}")

    pretty_text.append(f"\nSpan-level OVERLAP match (threshold={overlap_threshold}):")
    pretty_text.append(f"Gold spans      : {span_overlap['gold_spans']}")
    pretty_text.append(f"Pred spans      : {span_overlap['pred_spans']}")
    pretty_text.append(f"Matched         : {span_overlap['matched']}")
    pretty_text.append(f"Precision       : {span_overlap['precision']:.4f}")
    pretty_text.append(f"Recall          : {span_overlap['recall']:.4f}")
    pretty_text.append(f"F1              : {span_overlap['f1']:.4f}")

    return result, "\n".join(pretty_text)


def evaluate_from_json(json_path, overlap_threshold=0.5):
    data = load_result_json(json_path)
    return build_eval_result(data, overlap_threshold=overlap_threshold)


def save_eval_result(input_json_path, eval_result, eval_text, save_dir="results_eval"):
    input_json_path = Path(input_json_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = input_json_path.stem  
    json_save_path = save_dir / f"{stem}_eval.json"
    txt_save_path = save_dir / f"{stem}_eval.txt"

    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    with open(txt_save_path, "w", encoding="utf-8") as f:
        f.write(eval_text)

    print(f"[Saved] {json_save_path}")
    print(f"[Saved] {txt_save_path}")


def run_all_evaluations(results_dir="results", save_dir="results_eval", overlap_threshold=0.5):
    results_dir = Path(results_dir)

    json_files = sorted(results_dir.glob("llm*.json"))
    if not json_files:
        print(f"在 {results_dir} 下没有找到 json 文件。")
        return

    print(f"共找到 {len(json_files)} 个 json 文件。")

    for json_path in json_files:
        print(f"\n>>> Processing: {json_path.name}")
        try:
            eval_result, eval_text = evaluate_from_json(
                json_path,
                overlap_threshold=overlap_threshold
            )
            print(eval_text)
            save_eval_result(
                input_json_path=json_path,
                eval_result=eval_result,
                eval_text=eval_text,
                save_dir=save_dir
            )
        except Exception as e:
            print(f"[Failed] {json_path.name}: {e}")


if __name__ == "__main__":
    run_all_evaluations(
        results_dir="results",
        save_dir="results_eval",
        overlap_threshold=0.5
    )