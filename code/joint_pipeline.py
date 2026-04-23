from lstm_model import BiLSTMTagger, train_model, predict
from dataloader_utils import decode_predictions, apply_mask_to_labels
from data_utils import build_token_dataloader_joint

def run_joint(use_pre_trained=1):
    print("Step 8: 开始初始化 joint 模型...")
    id2label = {
    0: 'O',
    1: 'B-P',
    2: 'I-P',
    3: 'B-I',
    4: 'I-I',
    5: 'B-O',
    6: 'I-O'
}
    len_vocab, train_loader, test_loader, X_test_mask, test_labels, em = \
        build_token_dataloader_joint(use_pre_trained=use_pre_trained)

    model = BiLSTMTagger(
        vocab_size=len_vocab,
        emb_dim=50,
        hidden_dim=64,
        num_labels=7,
        embedding_matrix=em
    )

    print("Step 9: 开始训练 joint 模型...")
    model = train_model(model, train_loader, epochs=3)

    print("Step 10: 开始预测...")
    pred_ids = predict(model, test_loader)

    print("Step 11: 开始解码标签...")
    pred_labels = decode_predictions(pred_ids, id2label, X_test_mask)
    test_labels = apply_mask_to_labels(test_labels, X_test_mask)
    return test_labels, pred_labels