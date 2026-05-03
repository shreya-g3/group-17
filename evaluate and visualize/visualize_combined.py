import json
import html
import argparse
from pathlib import Path
from collections import OrderedDict


FIELDS = ["participants", "interventions", "outcomes"]

COLORS = {
    "participants": "#cfe8ff",    # blue
    "interventions": "#d9f7be",   # green
    "outcomes": "#ffe7ba",        # orange
}

SHORT = {
    "participants": "P",
    "interventions": "I",
    "outcomes": "O",
}



def load_result_json(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required = ["label_type", "doc_ids", "tokens", "gold_labels", "pred_labels"]
    for key in required:
        if key not in data:
            raise ValueError(f"{path} missing required key: {key}")

    return data


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

        else:
            if start is not None:
                spans.append((start, i))
                start = None

    if start is not None:
        spans.append((start, len(doc_labels)))

    return spans


def exact_f1(gold_labels, pred_labels):
    gold_spans = set(labels_to_spans(gold_labels))
    pred_spans = set(labels_to_spans(pred_labels))

    matched = len(gold_spans & pred_spans)

    precision = matched / len(pred_spans) if pred_spans else 0.0
    recall = matched / len(gold_spans) if gold_spans else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def overlap_match(pred_span, gold_span, threshold=0.5):
    ps, pe = pred_span
    gs, ge = gold_span

    inter = max(0, min(pe, ge) - max(ps, gs))
    gold_len = ge - gs

    if gold_len <= 0:
        return False

    return (inter / gold_len) >= threshold


def overlap_f1(gold_labels, pred_labels, threshold=0.5):
    gold_spans = labels_to_spans(gold_labels)
    pred_spans = labels_to_spans(pred_labels)

    matched_gold = set()
    matched = 0

    for pred_span in pred_spans:
        for gold_idx, gold_span in enumerate(gold_spans):
            if gold_idx in matched_gold:
                continue

            if overlap_match(pred_span, gold_span, threshold=threshold):
                matched += 1
                matched_gold.add(gold_idx)
                break

    precision = matched / len(pred_spans) if pred_spans else 0.0
    recall = matched / len(gold_spans) if gold_spans else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def build_doc_map(data):

    doc_map = OrderedDict()

    n = min(
        len(data["doc_ids"]),
        len(data["tokens"]),
        len(data["gold_labels"]),
        len(data["pred_labels"]),
    )

    for i in range(n):
        doc_id = data["doc_ids"][i]
        tokens = data["tokens"][i]
        gold = data["gold_labels"][i]
        pred = data["pred_labels"][i]

        L = min(len(tokens), len(gold), len(pred))

        doc_map[doc_id] = {
            "tokens": tokens[:L],
            "gold": gold[:L],
            "pred": pred[:L],
        }

    return doc_map




def merge_three_results(participants_path, interventions_path, outcomes_path):
    data_by_field = {
        "participants": load_result_json(participants_path),
        "interventions": load_result_json(interventions_path),
        "outcomes": load_result_json(outcomes_path),
    }

    maps = {
        field: build_doc_map(data)
        for field, data in data_by_field.items()
    }

    common_doc_ids = (
        set(maps["participants"].keys())
        & set(maps["interventions"].keys())
        & set(maps["outcomes"].keys())
    )

    ordered_doc_ids = [
        doc_id
        for doc_id in maps["participants"].keys()
        if doc_id in common_doc_ids
    ]

    pipeline_name = data_by_field["participants"].get("pipeline_name", "combined_run")

    docs = []

    for doc_id in ordered_doc_ids:
        # Use participant tokens as the base; truncate all fields to common length.
        base_tokens = maps["participants"][doc_id]["tokens"]
        min_len = min(
            len(maps["participants"][doc_id]["tokens"]),
            len(maps["interventions"][doc_id]["tokens"]),
            len(maps["outcomes"][doc_id]["tokens"]),
        )

        doc = {
            "doc_id": doc_id,
            "tokens": base_tokens[:min_len],
            "gold": {},
            "pred": {},
            "scores": {},
        }

        exact_sum = 0.0
        overlap_sum = 0.0

        for field in FIELDS:
            gold = maps[field][doc_id]["gold"][:min_len]
            pred = maps[field][doc_id]["pred"][:min_len]

            doc["gold"][field] = gold
            doc["pred"][field] = pred

            ex = exact_f1(gold, pred)
            ov = overlap_f1(gold, pred)

            doc["scores"][field] = {
                "exact_f1": ex,
                "overlap_f1": ov,
            }

            exact_sum += ex
            overlap_sum += ov

        doc["scores"]["avg_exact_f1"] = exact_sum / len(FIELDS)
        doc["scores"]["avg_overlap_f1"] = overlap_sum / len(FIELDS)

        docs.append(doc)

    return pipeline_name, docs



def render_combined_tokens(tokens, labels_by_field):

    parts = []

    for idx, tok in enumerate(tokens):
        tok_html = html.escape(str(tok))

        active_fields = []

        for field in FIELDS:
            labels = labels_by_field[field]
            lab = labels[idx] if idx < len(labels) else "O"

            if lab in ("B", "I"):
                active_fields.append(field)

        if not active_fields:
            parts.append(tok_html)
            continue

        # Normal case: one active field
        if len(active_fields) == 1:
            field = active_fields[0]
            color = COLORS[field]
            short = SHORT[field]

            parts.append(
                f'<span style="background:{color}; '
                f'padding:2px 4px; border-radius:4px; '
                f'margin:0 1px; display:inline-block;">'
                f'{tok_html}<sup style="font-size:9px; margin-left:2px;">{short}</sup>'
                f'</span>'
            )

        # Rare case: multiple field labels overlap at same token
        else:
            shorts = "/".join(SHORT[f] for f in active_fields)
            gradient_parts = []

            for i, field in enumerate(active_fields):
                start = int(i * 100 / len(active_fields))
                end = int((i + 1) * 100 / len(active_fields))
                gradient_parts.append(f"{COLORS[field]} {start}%, {COLORS[field]} {end}%")

            gradient = "linear-gradient(135deg, " + ", ".join(gradient_parts) + ")"

            parts.append(
                f'<span style="background:{gradient}; '
                f'padding:2px 4px; border-radius:4px; '
                f'margin:0 1px; display:inline-block;">'
                f'{tok_html}<sup style="font-size:9px; margin-left:2px;">{shorts}</sup>'
                f'</span>'
            )

    return " ".join(parts)


def render_legend():
    items = []

    for field in FIELDS:
        items.append(
            f'<span style="background:{COLORS[field]}; '
            f'padding:4px 8px; border-radius:6px; margin-right:8px; '
            f'display:inline-block;">'
            f'{SHORT[field]} = {field}'
            f'</span>'
        )

    return "".join(items)


def render_score_table(scores):
    rows = []

    for field in FIELDS:
        s = scores[field]
        rows.append(
            f"""
            <tr>
                <td>{html.escape(field)}</td>
                <td>{s["exact_f1"]:.3f}</td>
                <td>{s["overlap_f1"]:.3f}</td>
            </tr>
            """
        )

    rows.append(
        f"""
        <tr style="font-weight:bold;">
            <td>average</td>
            <td>{scores["avg_exact_f1"]:.3f}</td>
            <td>{scores["avg_overlap_f1"]:.3f}</td>
        </tr>
        """
    )

    return f"""
    <table style="border-collapse:collapse; margin:8px 0 14px 0; font-size:13px;">
        <tr>
            <th>Field</th>
            <th>Exact F1</th>
            <th>Overlap F1</th>
        </tr>
        {''.join(rows)}
    </table>
    """


def add_table_css(html_text):
    return html_text.replace(
        "<table",
        '<table'
    ).replace(
        "<th>",
        '<th style="border:1px solid #ccc; padding:4px 8px; background:#f6f6f6;">'
    ).replace(
        "<td>",
        '<td style="border:1px solid #ccc; padding:4px 8px;">'
    )


def render_doc_block(doc):
    gold_html = render_combined_tokens(doc["tokens"], doc["gold"])
    pred_html = render_combined_tokens(doc["tokens"], doc["pred"])
    score_html = add_table_css(render_score_table(doc["scores"]))

    return f"""
    <div style="margin-bottom:32px; padding:18px; border:1px solid #ddd; border-radius:10px;">
        <h2 style="margin:0 0 8px 0;">Document: {html.escape(str(doc["doc_id"]))}</h2>

        <div style="margin:6px 0 8px 0;">
            {render_legend()}
        </div>

        {score_html}

        <div style="margin-bottom:14px;">
            <div style="font-weight:bold; margin-bottom:6px;">Gold</div>
            <div style="line-height:2.1;">{gold_html}</div>
        </div>

        <div>
            <div style="font-weight:bold; margin-bottom:6px;">Prediction</div>
            <div style="line-height:2.1;">{pred_html}</div>
        </div>
    </div>
    """


def select_docs(docs, mode="worst", top_k=5):
    if mode == "worst":
        return sorted(
            docs,
            key=lambda x: (x["scores"]["avg_exact_f1"], x["scores"]["avg_overlap_f1"])
        )[:top_k]

    if mode == "best":
        return sorted(
            docs,
            key=lambda x: (x["scores"]["avg_exact_f1"], x["scores"]["avg_overlap_f1"]),
            reverse=True
        )[:top_k]

    if mode == "first":
        return docs[:top_k]

    if mode == "mixed":
        worst = sorted(
            docs,
            key=lambda x: (x["scores"]["avg_exact_f1"], x["scores"]["avg_overlap_f1"])
        )[: max(1, top_k // 2)]

        best = sorted(
            docs,
            key=lambda x: (x["scores"]["avg_exact_f1"], x["scores"]["avg_overlap_f1"]),
            reverse=True
        )[: top_k - len(worst)]

        seen = set()
        selected = []

        for d in worst + best:
            if d["doc_id"] not in seen:
                selected.append(d)
                seen.add(d["doc_id"])

        return selected

    raise ValueError("mode must be one of: worst, best, first, mixed")


def save_combined_html(pipeline_name, docs, output_path, mode="worst", top_k=5):
    selected_docs = select_docs(docs, mode=mode, top_k=top_k)

    blocks = "".join(render_doc_block(doc) for doc in selected_docs)

    title = f"{pipeline_name} combined P/I/O visualization"

    page = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>{html.escape(title)}</title>
    </head>
    <body style="font-family:Arial, sans-serif; margin:24px;">
        <h1 style="margin-top:0;">{html.escape(title)}</h1>

        <p style="color:#555;">
            Participants, interventions and outcomes are shown together on the same document text.
            Gold and prediction are displayed separately. Selection mode: <b>{html.escape(mode)}</b>,
            top_k = {top_k}.
        </p>

        <div style="margin:12px 0 20px 0;">
            {render_legend()}
        </div>

        {blocks}
    </body>
    </html>
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"[Saved] {output_path}")



def find_field_file(results_dir, run_prefix, field):

    results_dir = Path(results_dir)

    candidates = [
        results_dir / f"{run_prefix}{field}.json",
        results_dir / f"{run_prefix}_{field}.json",
        results_dir / f"{run_prefix}-{field}.json",
    ]

    for c in candidates:
        if c.exists():
            return c

    # fallback: fuzzy search
    matches = sorted(results_dir.glob(f"*{run_prefix}*{field}*.json"))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print(f"[Warning] Multiple matches for {run_prefix} + {field}:")
        for m in matches:
            print("  ", m)
        print("[Using first match]")
        return matches[0]

    raise FileNotFoundError(f"Cannot find JSON for run_prefix={run_prefix}, field={field}")


def run_from_prefix(results_dir, run_prefix, output_path, mode="worst", top_k=5):
    participants_path = find_field_file(results_dir, run_prefix, "participants")
    interventions_path = find_field_file(results_dir, run_prefix, "interventions")
    outcomes_path = find_field_file(results_dir, run_prefix, "outcomes")

    print("[Using files]")
    print("participants :", participants_path)
    print("interventions:", interventions_path)
    print("outcomes     :", outcomes_path)

    pipeline_name, docs = merge_three_results(
        participants_path=participants_path,
        interventions_path=interventions_path,
        outcomes_path=outcomes_path,
    )

    save_combined_html(
        pipeline_name=f"{run_prefix}",
        docs=docs,
        output_path=output_path,
        mode=mode,
        top_k=top_k,
    )


def run_from_three_paths(participants_path, interventions_path, outcomes_path, output_path, mode="worst", top_k=5):
    pipeline_name, docs = merge_three_results(
        participants_path=participants_path,
        interventions_path=interventions_path,
        outcomes_path=outcomes_path,
    )

    save_combined_html(
        pipeline_name=pipeline_name,
        docs=docs,
        output_path=output_path,
        mode=mode,
        top_k=top_k,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create one combined P/I/O visualization for a specific model/run."
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing result JSON files."
    )

    parser.add_argument(
        "--run_prefix",
        type=str,
        default=None,
        help="Run prefix, e.g. separate_A_5. The script will auto-find P/I/O files."
    )

    parser.add_argument("--participants", type=str, default=None)
    parser.add_argument("--interventions", type=str, default=None)
    parser.add_argument("--outcomes", type=str, default=None)

    parser.add_argument(
        "--output",
        type=str,
        default="combined_vis.html",
        help="Output HTML path."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="worst",
        choices=["worst", "best", "first", "mixed"],
        help="Which documents to show."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of documents to show."
    )

    args = parser.parse_args()

    if args.run_prefix is not None:
        run_from_prefix(
            results_dir=args.results_dir,
            run_prefix=args.run_prefix,
            output_path=args.output,
            mode=args.mode,
            top_k=args.top_k,
        )
        return

    if args.participants and args.interventions and args.outcomes:
        run_from_three_paths(
            participants_path=args.participants,
            interventions_path=args.interventions,
            outcomes_path=args.outcomes,
            output_path=args.output,
            mode=args.mode,
            top_k=args.top_k,
        )
        return

    raise ValueError(
        "Use either --run_prefix OR provide --participants, --interventions and --outcomes."
    )


if __name__ == "__main__":
    main()