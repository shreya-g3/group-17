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
    """
    BIO -> [(start, end), ...]
    end 为开区间
    """
    spans = []
    start = None

    for i, lab in enumerate(doc_labels):
        if lab == 'B':
            if start is not None:
                spans.append((start, i))
            start = i

        elif lab == 'I':
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

    print("\nClassification Report (token-level):")
    print(classification_report(y_true_flat, y_pred_flat, digits=4))


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

    print("\nSpan-level EXACT match:")
    print(f"Gold spans      : {gold_total}")
    print(f"Pred spans      : {pred_total}")
    print(f"Exact matched   : {exact_match}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1              : {f1:.4f}")


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

    print(f"\nSpan-level OVERLAP match (threshold={threshold}):")
    print(f"Gold spans      : {gold_total}")
    print(f"Pred spans      : {pred_total}")
    print(f"Matched         : {overlap_match}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1              : {f1:.4f}")


def evaluate_from_json(json_path, overlap_threshold=0.5):
    data = load_result_json(json_path)

    label_type = data["label_type"]
    pipeline_name = data.get("pipeline_name", "unknown_pipeline")
    gold_labels = data["gold_labels"]
    pred_labels = data["pred_labels"]

    print("=" * 60)
    print(f"Evaluating pipeline: {pipeline_name}")
    print(f"Label type        : {label_type}")
    print(f"Docs              : {len(gold_labels)}")
    print("=" * 60)

    evaluate_token_level(gold_labels, pred_labels)
    evaluate_span_exact(gold_labels, pred_labels)
    evaluate_span_overlap(gold_labels, pred_labels, threshold=overlap_threshold)


if __name__ == "__main__":
    json_path = "results/baseline_participants.json"
    evaluate_from_json(json_path, overlap_threshold=0.5)