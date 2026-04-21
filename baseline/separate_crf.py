from lstm_crf_model import BiLSTMCRFTagger, train_model_crf, predict_crf
from data_utils import build_token_dataloader_single_crf


def run_separate_crf(label_type='participants', use_pre_trained=1):
    id2label = {0: 'O', 1: 'B', 2: 'I'}

    print("Step 1: 构建 CRF dataloader...")
    len_vocab, train_loader, test_loader, test_labels, em = \
        build_token_dataloader_single_crf(
            label_type=label_type,
            use_pre_trained=use_pre_trained
        )

    print("Step 2: 初始化 BiLSTM-CRF 模型...")
    model = BiLSTMCRFTagger(
        vocab_size=len_vocab,
        emb_dim=50,
        hidden_dim=64,
        num_labels=3,
        embedding_matrix=em
    )

    print("Step 3: 训练模型...")
    model = train_model_crf(model, train_loader, epochs=3)

    print("Step 4: 在 test 上预测...")
    pred_ids = predict_crf(model, test_loader)

    print("Step 5: 解码 BIO 标签...")
    pred_labels = []
    for doc_pred in pred_ids:
        pred_labels.append([id2label[int(x)] for x in doc_pred])

    return test_labels, pred_labels