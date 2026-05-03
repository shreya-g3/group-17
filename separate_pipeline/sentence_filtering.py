from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from data_utils import build_sentence_dataset_with_meta


def run_sentence_filter_with_meta(label_type='participants'):
    train_data = build_sentence_dataset_with_meta(label_type, split='train')
    test_data = build_sentence_dataset_with_meta(label_type, split='test')

    X_train = [x['text'] for x in train_data]
    y_train = [x['y'] for x in train_data]

    X_test = [x['text'] for x in test_data]
    y_test = [x['y'] for x in test_data]

    vectorizer = TfidfVectorizer(lowercase=True, min_df=2, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    y_prob = clf.predict_proba(X_test_vec)[:, 1]

    print(f"\nSentence filtering for {label_type}")
    print(classification_report(y_test, y_pred, digits=4))
    print("keep ratio =", sum(y_pred) / len(y_pred))

    for item, pred, prob in zip(test_data, y_pred, y_prob):
        item['pred'] = int(pred)
        item['prob'] = float(prob)

    return train_data, test_data, vectorizer, clf


def rebuild_docs_from_kept_sentences(sentence_data, use_gold=False, fallback_top1=True):
    grouped = defaultdict(list)
    for item in sentence_data:
        grouped[item['doc_id']].append(item)

    doc_ids = []
    filtered_tokens = []
    filtered_labels = []
    kept_items_by_doc = []

    for doc_id in sorted(grouped.keys()):
        items = sorted(grouped[doc_id], key=lambda x: x['sent_id'])

        kept = []
        for item in items:
            keep = item['y'] if use_gold else item.get('pred', 0)
            if keep == 1:
                kept.append(item)

        if len(kept) == 0 and fallback_top1 and (not use_gold):
            best_item = max(items, key=lambda x: x.get('prob', 0.0))
            kept = [best_item]

        doc_tokens = []
        doc_labels = []

        for item in kept:
            doc_tokens.extend(item['sent_tokens'])
            doc_labels.extend(item['sent_labels'])

        doc_ids.append(doc_id)
        filtered_tokens.append(doc_tokens)
        filtered_labels.append(doc_labels)
        kept_items_by_doc.append(kept)

    return doc_ids, filtered_tokens, filtered_labels, kept_items_by_doc