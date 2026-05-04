import json
from pathlib import Path
from separate_pipeline import run_separate, run_separate_with_sentence_filter
from dataloader_utils import get_doc_ids, get_all

def save_results_json(
    pipeline_name,
    label_type,
    doc_ids,
    tokens,
    gold_labels,
    pred_labels,
    save_dir="results",
    extra_meta=None
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "pipeline_name": pipeline_name,
        "label_type": label_type,
        "doc_ids": doc_ids,
        "tokens": tokens,
        "gold_labels": gold_labels,
        "pred_labels": pred_labels,
    }
    if extra_meta is not None:
        result["meta"] = extra_meta
    save_path = save_dir / f"{pipeline_name}_{label_type}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {save_path}")

if __name__ == '__main__':
    label_type = 'participants'
    use_pre_trained = 0
    use_pre_trained_decomposed = 1   # this part uses GloVe

    test_doc_ids = get_doc_ids(split="test", label_type=label_type)
    _, test_tokens, _ = get_all(test_doc_ids, label_type=label_type, split="test")

    # 1. Baseline
    # =========================
    print("\n===== Baseline: direct token extraction =====")
    test_labels_base, pred_labels_base = run_separate(
        label_type=label_type,
        use_pre_trained=use_pre_trained
    )
    save_results_json(
        pipeline_name="baseline",
        label_type=label_type,
        doc_ids=test_doc_ids,
        tokens=test_tokens,
        gold_labels=test_labels_base,
        pred_labels=pred_labels_base,
        extra_meta={
            "use_pre_trained": use_pre_trained
        }
    )

    # 2. Sentence filter + token extraction
    # =========================
    print("\n===== Decomposed: sentence filter + token extraction =====")
    test_labels_sf, pred_labels_sf = run_separate_with_sentence_filter(
        label_type=label_type,
        use_gold_sent=False,
        use_pre_trained=use_pre_trained_decomposed,
    )


    save_results_json(
        pipeline_name="sentence_filter",
        label_type=label_type,
        doc_ids=test_doc_ids,
        tokens=test_tokens,
        gold_labels=test_labels_sf,
        pred_labels=pred_labels_sf,
        extra_meta={
            "use_pre_trained": use_pre_trained,
            "use_gold_sent": False
        }
    )
