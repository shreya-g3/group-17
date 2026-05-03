from lstm_model import BiLSTMTagger, train_model, predict
from dataloader_utils import decode_predictions, get_all, get_doc_ids, apply_mask_to_labels
from data_utils import build_token_dataloader_single, build_token_dataloader_from_docs
from sentence_filtering import run_sentence_filter_with_meta, rebuild_docs_from_kept_sentences

def align_filtered_predictions_to_original_docs(
    original_doc_ids,
    original_tokens,
    original_labels,
    filtered_doc_ids,
    kept_items_by_doc,
    filtered_pred_labels,
):
    aligned_gold = []
    aligned_pred = []

    filtered_map = {}
    for doc_id, kept_items, pred_labels in zip(filtered_doc_ids, kept_items_by_doc, filtered_pred_labels):
        filtered_map[doc_id] = {
            "kept_items": kept_items,
            "pred_labels": pred_labels
        }

    for doc_id, doc_tokens, doc_gold in zip(original_doc_ids, original_tokens, original_labels):
        doc_len = len(doc_tokens)

        aligned_gold.append(doc_gold[:])

    
        doc_pred = ['O'] * doc_len

        if doc_id in filtered_map:
            kept_items = filtered_map[doc_id]["kept_items"]
            pred_seq = filtered_map[doc_id]["pred_labels"]

            offset = 0
            for item in kept_items:
                start = item["start"]
                end = item["end"]
                seg_len = end - start

                seg_pred = pred_seq[offset: offset + seg_len]
                doc_pred[start:end] = seg_pred
                offset += seg_len

        aligned_pred.append(doc_pred)

    return aligned_gold, aligned_pred
def run_separate(label_type='participants', use_pre_trained=1):
    id2label = {0: 'O', 1: 'B', 2: 'I'}
    print("Step 7: 开始初始化模型...")
    len_vocab, train_loader, test_loader, X_test_mask, test_labels, em = build_token_dataloader_single(label_type, use_pre_trained=use_pre_trained)

    model = BiLSTMTagger(
        vocab_size=len_vocab,
        emb_dim=50,
        hidden_dim=64,
        num_labels=3,
        embedding_matrix=em
    )

    print("Step 7 已完成：模型初始化完成")
    print("Step 8: 开始训练模型...")
    model = train_model(model, train_loader, epochs=5)

    print("Step 8 已完成：模型训练完成")
    print("Step 9: 开始预测...")
    pred_ids = predict(model, test_loader)

    print("Step 9 已完成：预测完成")
    print(f"预测得到的文档数: {len(pred_ids)}")

    print("Step 10: 开始解码 BIO 标签...")
    pred_labels = decode_predictions(pred_ids, id2label, X_test_mask)
    test_labels = apply_mask_to_labels(test_labels, X_test_mask)
    print("Step 10 已完成")
    return test_labels, pred_labels


def run_separate_with_sentence_filter(label_type='participants', use_gold_sent=False, use_pre_trained=1):
    id2label = {0: 'O', 1: 'B', 2: 'I'}

    print("Step 1: 读取原始 train 数据...")
    train_doc_ids = get_doc_ids(split="train", label_type=label_type)
    train_labels, train_tokens, _ = get_all(train_doc_ids, label_type, 'train')

    print("Step 1.5: 读取原始 test 数据...")
    original_test_doc_ids = get_doc_ids(split="test", label_type=label_type)
    original_test_labels, original_test_tokens, _ = get_all(original_test_doc_ids, label_type, 'test')

    print("Step 2: 运行 sentence filter...")
    _, test_sent_data, _, _ = run_sentence_filter_with_meta(label_type)

    print("Step 3: 重建过滤后的 test 文档...")
    filtered_doc_ids, filtered_test_tokens, filtered_test_labels, kept_items_by_doc = rebuild_docs_from_kept_sentences(
        test_sent_data,
        use_gold=use_gold_sent,
        fallback_top1=True
    )


    print("Step 4: 构建 token dataloader...")
    len_vocab, train_loader, test_loader, X_test_mask, test_labels, em = build_token_dataloader_from_docs(
        train_tokens=train_tokens,
        train_labels=train_labels,
        test_tokens=filtered_test_tokens,
        test_labels=filtered_test_labels,
        use_pre_trained=use_pre_trained,
    )

    print("Step 5: 初始化 token 模型...")
    model = BiLSTMTagger(
        vocab_size=len_vocab,
        emb_dim=50,
        hidden_dim=64,
        num_labels=3,
        embedding_matrix=em
    )

    print("Step 6: 训练 token 模型...")
    model = train_model(model, train_loader, epochs=5)

    print("Step 7: 在过滤后的 test 上预测...")
    pred_ids = predict(model, test_loader)

    print("Step 8: 解码 BIO 标签...")
    pred_labels_filtered = decode_predictions(pred_ids, id2label, X_test_mask)
    test_labels_filtered = apply_mask_to_labels(test_labels, X_test_mask)

    print("Step 9: 对齐回原始文档...")
    aligned_gold, aligned_pred = align_filtered_predictions_to_original_docs(
        original_doc_ids=original_test_doc_ids,
        original_tokens=original_test_tokens,
        original_labels=original_test_labels,
        filtered_doc_ids=filtered_doc_ids,
        kept_items_by_doc=kept_items_by_doc,
        filtered_pred_labels=pred_labels_filtered
    )

    return aligned_gold, aligned_pred