import json
import html
from pathlib import Path


# =========================================================
# 1. 读取 JSON
# =========================================================
def load_result_json(json_path):
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = ["label_type", "doc_ids", "tokens", "gold_labels", "pred_labels"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"JSON 缺少必要字段: {key}")

    return data


# =========================================================
# 2. 基础配置
# =========================================================
def get_color(label_type):
    color_map = {
        "participants": "#cfe8ff",
        "interventions": "#d9f7be",
        "outcomes": "#ffe7ba",
        "comparator": "#e4d1ff",
    }
    return color_map.get(label_type, "#cfe8ff")


# =========================================================
# 3. 渲染单篇文档
# =========================================================
def render_single_label_doc(tokens, labels, color="#cfe8ff"):
    parts = []
    in_span = False
    just_closed = False

    for tok, lab in zip(tokens, labels):
        tok = html.escape(tok)

        if lab == "B":
            if in_span:
                parts.append("</span>")
            if parts and not just_closed:
                parts.append(" ")
            parts.append(
                f'<span style="background-color:{color}; padding:2px 4px; '
                f'border-radius:4px; margin:0 1px; display:inline-block;">'
            )
            parts.append(tok)
            in_span = True
            just_closed = False

        elif lab == "I":
            if not in_span:
                if parts and not just_closed:
                    parts.append(" ")
                parts.append(
                    f'<span style="background-color:{color}; padding:2px 4px; '
                    f'border-radius:4px; margin:0 1px; display:inline-block;">'
                )
                in_span = True
            parts.append(" " + tok)
            just_closed = False

        else:  # O
            if in_span:
                parts.append("</span>")
                in_span = False
                just_closed = True
            else:
                just_closed = False

            if parts and not just_closed:
                parts.append(" ")
            parts.append(tok)

    if in_span:
        parts.append("</span>")

    return "".join(parts)


# =========================================================
# 4. span 评估工具（文档级）
# =========================================================
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


def safe_divide(a, b):
    return a / b if b > 0 else 0.0


def compute_prf(num_match, num_pred, num_gold):
    precision = safe_divide(num_match, num_pred)
    recall = safe_divide(num_match, num_gold)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def spans_overlap_with_threshold(span_pred, span_gold, threshold=0.5):
    s1, e1 = span_pred
    s2, e2 = span_gold

    inter = max(0, min(e1, e2) - max(s1, s2))
    gold_len = e2 - s2

    if gold_len == 0:
        return False

    overlap_ratio = inter / gold_len
    return overlap_ratio >= threshold


def doc_exact_f1(gold_labels, pred_labels):
    gold_spans = labels_to_spans(gold_labels)
    pred_spans = labels_to_spans(pred_labels)

    gold_set = set(gold_spans)
    pred_set = set(pred_spans)
    match = len(gold_set & pred_set)

    _, _, f1 = compute_prf(match, len(pred_set), len(gold_set))
    return f1


def doc_overlap_f1(gold_labels, pred_labels, threshold=0.5):
    gold_spans = labels_to_spans(gold_labels)
    pred_spans = labels_to_spans(pred_labels)

    matched_gold = set()
    overlap_match = 0

    for pred_span in pred_spans:
        for gold_idx, gold_span in enumerate(gold_spans):
            if gold_idx in matched_gold:
                continue
            if spans_overlap_with_threshold(pred_span, gold_span, threshold=threshold):
                overlap_match += 1
                matched_gold.add(gold_idx)
                break

    _, _, f1 = compute_prf(overlap_match, len(pred_spans), len(gold_spans))
    return f1


# =========================================================
# 5. 构造 case
# =========================================================
def build_cases_from_json(json_path):
    data = load_result_json(json_path)

    doc_ids = data["doc_ids"]
    tokens = data["tokens"]
    gold_labels = data["gold_labels"]
    pred_labels = data["pred_labels"]
    label_type = data["label_type"]
    pipeline_name = data.get("pipeline_name", "unknown_pipeline")

    n = min(len(doc_ids), len(tokens), len(gold_labels), len(pred_labels))

    cases = []
    for i in range(n):
        L = min(len(tokens[i]), len(gold_labels[i]), len(pred_labels[i]))
        cases.append({
            "doc_id": doc_ids[i],
            "tokens": tokens[i][:L],
            "gold": gold_labels[i][:L],
            "pred": pred_labels[i][:L],
        })

    return label_type, pipeline_name, cases


# =========================================================
# 6. 打分、排序、挑例子
# =========================================================
def add_scores_to_cases(cases, overlap_threshold=0.5):
    scored = []

    for case in cases:
        exact_f1 = doc_exact_f1(case["gold"], case["pred"])
        overlap_f1 = doc_overlap_f1(case["gold"], case["pred"], threshold=overlap_threshold)

        new_case = dict(case)
        new_case["exact_f1"] = exact_f1
        new_case["overlap_f1"] = overlap_f1
        scored.append(new_case)

    return scored


def rank_cases(cases, overlap_threshold=0.5):
    scored = add_scores_to_cases(cases, overlap_threshold=overlap_threshold)

    scored_sorted = sorted(
        scored,
        key=lambda x: (x["exact_f1"], x["overlap_f1"]),
        reverse=True
    )

    good_cases = scored_sorted[:4]
    bad_cases = scored_sorted[-4:]

    return good_cases, bad_cases, scored_sorted


# =========================================================
# 7. HTML 渲染
# =========================================================
def render_case_html(case, label_type="participants"):
    color = get_color(label_type)

    gold_html = render_single_label_doc(case["tokens"], case["gold"], color=color)
    pred_html = render_single_label_doc(case["tokens"], case["pred"], color=color)

    return f"""
    <div style="margin-bottom: 28px; padding: 16px; border: 1px solid #ddd; border-radius: 8px;">
        <h3 style="margin: 0 0 10px 0;">Document: {html.escape(str(case["doc_id"]))}</h3>
        <p style="margin: 4px 0; color: #444;">
            exact_f1 = {case["exact_f1"]:.4f}, overlap_f1 = {case["overlap_f1"]:.4f}
        </p>

        <div style="margin-bottom: 10px;">
            <b>Gold</b>
            <div style="line-height: 1.9; margin-top: 6px;">{gold_html}</div>
        </div>

        <div>
            <b>Prediction</b>
            <div style="line-height: 1.9; margin-top: 6px;">{pred_html}</div>
        </div>
    </div>
    """


def save_ranked_visualization_html(
    good_cases,
    bad_cases,
    output_path="prediction_vis_ranked.html",
    label_type="participants",
    title="PICO Extraction Visualization"
):
    good_html = "".join(render_case_html(case, label_type=label_type) for case in good_cases)
    bad_html = "".join(render_case_html(case, label_type=label_type) for case in bad_cases)

    page = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>{html.escape(title)}</title>
    </head>
    <body style="font-family: Arial, sans-serif; margin: 24px;">
        <h1>{html.escape(title)}</h1>
        <p style="color: #555;">Field: {html.escape(label_type)}</p>

        <h2 style="margin-top: 32px;">4 Good Cases</h2>
        {good_html}

        <h2 style="margin-top: 40px;">4 Bad Cases</h2>
        {bad_html}
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"HTML 已保存到: {output_path}")


# =========================================================
# 8. 一键运行
# =========================================================
def run_visualization_from_json(json_path, output_path=None, overlap_threshold=0.5):
    label_type, pipeline_name, cases = build_cases_from_json(json_path)
    good_cases, bad_cases, _ = rank_cases(cases, overlap_threshold=overlap_threshold)

    if output_path is None:
        output_path = f"{pipeline_name}_{label_type}_ranked_vis.html"

    save_ranked_visualization_html(
        good_cases,
        bad_cases,
        output_path=output_path,
        label_type=label_type,
        title=f"{pipeline_name} {label_type} visualization"
    )


if __name__ == "__main__":
    json_path = "results/baseline_participants.json"
    run_visualization_from_json(
        json_path=json_path,
        output_path=None,
        overlap_threshold=0.5
    )
    json_path = "results/sentence_filter_participants.json"
    run_visualization_from_json(
        json_path=json_path,
        output_path=None,
        overlap_threshold=0.5
    )
    json_path = "results/crf_participants.json"
    run_visualization_from_json(
        json_path=json_path,
        output_path=None,
        overlap_threshold=0.5
    )