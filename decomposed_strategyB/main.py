import json
from pathlib import Path

from separate_pipeline import run_separate, run_separate_with_sentence_filter
from separate_crf import run_separate_crf
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

    print(f"结果已保存到: {save_path}")


if __name__ == '__main__':
    label_type = 'participants'
    use_pre_trained = 0
    use_pre_trained_decomposed = 1   # this part uses GloVe


    # 原始 test 文档信息（baseline / CRF 可直接对齐原 test 集）
    test_doc_ids = get_doc_ids(split="test", label_type=label_type)
    _, test_tokens, _ = get_all(test_doc_ids, label_type=label_type, split="test")

    # =========================
    # 1. baseline
    # =========================
    print("\n===== baseline: 直接 token extraction =====")
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

    # =========================
    # 2. sentence filter + token extraction
    # =========================
    print("\n===== decomposed: sentence filter + token extraction =====")
    test_labels_sf, pred_labels_sf = run_separate_with_sentence_filter(
        label_type=label_type,
        use_gold_sent=False,
        use_pre_trained=use_pre_trained_decomposed,
    )

    # 这里先仍然保存原始 test_doc_ids / test_tokens
    # 但要注意：sentence filter 实际上预测的是“过滤后的文档”
    # 如果后面 evaluate / visualize 要严格对应过滤后文本，
    # 最好把 run_separate_with_sentence_filter 里重建后的 doc_ids/tokens 也一起返回或单独保存。
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

    # =========================
    # 3. CRF
    # =========================
    print("\n===== baseline with CRF =====")
    test_labels_crf, pred_labels_crf = run_separate_crf(
        label_type=label_type,
        use_pre_trained=use_pre_trained
    )

    save_results_json(
        pipeline_name="crf",
        label_type=label_type,
        doc_ids=test_doc_ids,
        tokens=test_tokens,
        gold_labels=test_labels_crf,
        pred_labels=pred_labels_crf,
        extra_meta={
            "use_pre_trained": use_pre_trained
        }
    )