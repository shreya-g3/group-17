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

    save_path = save_dir / f"{pipeline_name}_A_5_{label_type}.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {save_path}")


def run(label_type = 'interventions'):
    use_pre_trained = 0


    test_doc_ids = get_doc_ids(split="test", label_type=label_type)
    _, test_tokens, _ = get_all(test_doc_ids, label_type=label_type, split="test")

    #separate

    test_labels_base, pred_labels_base = run_separate(
        label_type=label_type,
        use_pre_trained=use_pre_trained
    )

    save_results_json(
        pipeline_name="separate",
        label_type=label_type,
        doc_ids=test_doc_ids,
        tokens=test_tokens,
        gold_labels=test_labels_base,
        pred_labels=pred_labels_base,
        extra_meta={
            "use_pre_trained": use_pre_trained
        }
    )


    # sentence filter + token extraction


    test_labels_sf, pred_labels_sf = run_separate_with_sentence_filter(
        label_type=label_type,
        use_gold_sent=False,
        use_pre_trained=use_pre_trained,
    )

    save_results_json(
        pipeline_name="decomposed",
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


    # CRF


    # test_labels_crf, pred_labels_crf = run_separate_crf(
    #     label_type=label_type,
    #     use_pre_trained=use_pre_trained
    # )

    # save_results_json(
    #     pipeline_name="crf",
    #     label_type=label_type,
    #     doc_ids=test_doc_ids,
    #     tokens=test_tokens,
    #     gold_labels=test_labels_crf,
    #     pred_labels=pred_labels_crf,
    #     extra_meta={
    #         "use_pre_trained": use_pre_trained
    #     }
    # )
if __name__ == "__main__":
    for label in ['participants', 'interventions', 'outcomes']:
        run(label_type=label)