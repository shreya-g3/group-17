from separate_pipeline import run_separate, run_separate_with_sentence_filter
from joint_pipeline import run_joint
from evaluate import evaluate

if __name__ == '__main__':

    label_type = 'participants'

    print("\n===== baseline: 直接 token extraction =====")
    test_labels_base, pred_labels_base = run_separate(label_type=label_type, use_pre_trained=0)
    evaluate(test_labels_base, pred_labels_base)

    print("\n===== decomposed: sentence filter + token extraction =====")
    test_labels_sf, pred_labels_sf = run_separate_with_sentence_filter(
        label_type=label_type,
        use_gold_sent=False,
        use_pre_trained=0,
    )
    evaluate(test_labels_sf, pred_labels_sf)
