from lstm_model import BiLSTMTagger, train_model, predict
from dataloader_utils import decode_predictions, get_all, get_doc_ids, apply_mask_to_labels
from data_utils import build_token_dataloader_single, build_token_dataloader_from_docs
from sentence_filtering import run_sentence_filter_with_meta, rebuild_docs_from_kept_sentences


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
    model = train_model(model, train_loader, epochs=3)

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

    print("Step 2: 运行 sentence filter...")
    _, test_sent_data, _, _ = run_sentence_filter_with_meta(label_type)

    print("Step 3: 重建过滤后的 test 文档...")
    filtered_doc_ids, filtered_test_tokens, filtered_test_labels = rebuild_docs_from_kept_sentences(
        test_sent_data,
        use_gold=use_gold_sent,
        fallback_top1=True
    )

    print(f"过滤后 test 文档数: {len(filtered_doc_ids)}")
    print(f"过滤后第一篇 token 数: {len(filtered_test_tokens[0])}")

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
    model = train_model(model, train_loader, epochs=3)

    print("Step 7: 在过滤后的 test 上预测...")
    pred_ids = predict(model, test_loader)

    print("Step 8: 解码 BIO 标签...")
    pred_labels = decode_predictions(pred_ids, id2label, X_test_mask)
    test_labels = apply_mask_to_labels(test_labels, X_test_mask)
    return test_labels, pred_labels