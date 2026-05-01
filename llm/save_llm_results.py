import json
from pathlib import Path
from separate_llm import run_separate_llm
from dataloader_utils import get_doc_ids, get_all


def save_for_pipeline(pipeline_name, label_type, mode):
    print(f"\n{'='*60}")
    print(f"Running {pipeline_name} for {label_type}")
    print(f"{'='*60}")

    gold_labels, pred_labels = run_separate_llm(
        label_type=label_type,
        mode=mode,
        use_cached=True,
        allow_fresh=True,
    )

    doc_ids = get_doc_ids(split="test", label_type=label_type)
    _, tokens, _ = get_all(doc_ids, label_type, "test")

    result = {
        "pipeline_name": pipeline_name,
        "label_type": label_type,
        "doc_ids": doc_ids,
        "tokens": tokens,
        "gold_labels": gold_labels,
        "pred_labels": pred_labels,
    }

    Path("results").mkdir(exist_ok=True)
    save_path = Path("results") / f"{pipeline_name}_B{label_type}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    # All 3 fields × 2 modes = 6 result JSONs
    for label in ['participants', 'interventions', 'outcomes']:
        for mode in ['zero_shot', 'few_shot']:
            pipeline_name = f"llm_{mode}"
            save_for_pipeline(pipeline_name, label, mode)

    print("\nAll 6 LLM result files saved.")
    print("Now run: python evaluate.py")