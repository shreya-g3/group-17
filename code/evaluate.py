from dataloader_utils import flatten_labels
from sklearn.metrics import classification_report

def evaluate(test_labels, pred_labels):
    y_true_flat = flatten_labels(test_labels)
    y_pred_flat = flatten_labels(pred_labels)

    print("\nClassification Report (token-level):")
    print(classification_report(y_true_flat, y_pred_flat))