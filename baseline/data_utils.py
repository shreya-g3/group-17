from dataloader_utils import get_all, build_vocab, encode_labels, encode_tokens
from dataloader_utils import pad_sequences, get_doc_ids
from dataloader_utils import split_into_sentences, sentence_label_from_bio
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
def build_token_dataloader_from_docs(train_tokens, train_labels, test_tokens, test_labels, label2id=None, use_pre_trained=1):
    vocab = build_vocab(train_tokens)
    if label2id == None:
        label2id = {'O': 0, 'B': 1, 'I': 2}
    X_train = encode_tokens(train_tokens, vocab)
    X_test = encode_tokens(test_tokens, vocab)

    y_train = encode_labels(train_labels, label2id)
    y_test = encode_labels(test_labels, label2id)

    X_train_pad, X_train_mask = pad_sequences(X_train, pad_value=vocab['<PAD>'], max_len=None)
    X_test_pad, X_test_mask = pad_sequences(X_test, pad_value=vocab['<PAD>'], max_len=len(X_train_pad[0]))

    y_train_pad, _ = pad_sequences(y_train, pad_value=-100, max_len=len(X_train_pad[0]))
    y_test_pad, _ = pad_sequences(y_test, pad_value=-100, max_len=len(X_train_pad[0]))

    X_train_pad = torch.tensor(X_train_pad)
    y_train_pad = torch.tensor(y_train_pad)

    X_test_pad = torch.tensor(X_test_pad)
    y_test_pad = torch.tensor(y_test_pad)

    train_loader = DataLoader(TensorDataset(X_train_pad, y_train_pad), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_pad, y_test_pad), batch_size=32)

    if use_pre_trained:
        embedding_matrix = build_embedding_matrix(vocab, "./glove.6B/glove.6B.50d.txt")
    else:
        embedding_matrix = None

    return len(vocab), train_loader, test_loader, X_test_mask, test_labels, embedding_matrix
def build_embedding_matrix(vocab, embedding_path, embedding_dim=50):

    embedding_matrix = np.random.randn(len(vocab), embedding_dim)

    with open(embedding_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0].lower()
            vec = np.array(parts[1:], dtype=float)

            if word in vocab:
                embedding_matrix[vocab[word]] = vec

    return embedding_matrix
def build_token_dataloader_single(label_type='participants', use_pre_trained=1):
    train_doc_ids = get_doc_ids(split="train", label_type=label_type)
    test_doc_ids = get_doc_ids(split="test", label_type=label_type)

    train_labels, train_tokens, _ = get_all(train_doc_ids, label_type, 'train')
    test_labels, test_tokens, _ = get_all(test_doc_ids, label_type, 'test')

    label2id = {'O': 0, 'B': 1, 'I': 2}

    len_vocab, train_loader, test_loader, X_test_pad, test_labels, embedding_matrix = \
        build_token_dataloader_from_docs(
            train_tokens, train_labels,
            test_tokens, test_labels,
            label2id=label2id,
            use_pre_trained=use_pre_trained,
        )

    return len_vocab, train_loader, test_loader, X_test_pad, test_labels, embedding_matrix

def build_sentence_dataset(label_type='participants', split='train'):
    doc_ids = get_doc_ids(split=split, label_type=label_type)
    labels, tokens, _ = get_all(doc_ids, label_type=label_type, split=split)

    X_sent = []
    y_sent = []

    for doc_tokens, doc_labels in zip(tokens, labels):
        sent_pairs = split_into_sentences(doc_tokens, doc_labels)

        for sent_tokens, sent_labels in sent_pairs:
            X_sent.append(" ".join(sent_tokens))
            y_sent.append(sentence_label_from_bio(sent_labels))

    return X_sent, y_sent
def build_sentence_dataset_with_meta(label_type='participants', split='train'):
    doc_ids = get_doc_ids(split=split, label_type=label_type)
    labels, tokens, _ = get_all(doc_ids, label_type=label_type, split=split)

    data = []

    for doc_id, doc_tokens, doc_labels in zip(doc_ids, tokens, labels):
        cur_tokens = []
        cur_labels = []
        start_idx = 0
        sent_id = 0

        for i, tok in enumerate(doc_tokens):
            cur_tokens.append(tok)
            cur_labels.append(doc_labels[i])

            if tok in ['.', '?', '!']:
                data.append({
                    'doc_id': doc_id,
                    'sent_id': sent_id,
                    'start': start_idx,
                    'end': i + 1,
                    'text': " ".join(cur_tokens),
                    'sent_tokens': cur_tokens[:],
                    'sent_labels': cur_labels[:],
                    'y': sentence_label_from_bio(cur_labels)
                })
                cur_tokens = []
                cur_labels = []
                start_idx = i + 1
                sent_id += 1

        if cur_tokens:
            data.append({
                'doc_id': doc_id,
                'sent_id': sent_id,
                'start': start_idx,
                'end': len(doc_tokens),
                'text': " ".join(cur_tokens),
                'sent_tokens': cur_tokens[:],
                'sent_labels': cur_labels[:],
                'y': sentence_label_from_bio(cur_labels)
            })

    return data
def build_token_dataloader_from_docs_crf(
    train_tokens,
    train_labels,
    test_tokens,
    test_labels,
    label2id=None,
    use_pre_trained=1
):
    vocab = build_vocab(train_tokens)

    if label2id is None:
        label2id = {'O': 0, 'B': 1, 'I': 2}

    X_train = encode_tokens(train_tokens, vocab)
    X_test = encode_tokens(test_tokens, vocab)

    y_train = encode_labels(train_labels, label2id)
    y_test = encode_labels(test_labels, label2id)

    X_train_pad, X_train_mask = pad_sequences(X_train, pad_value=vocab['<PAD>'], max_len=None)
    X_test_pad, X_test_mask = pad_sequences(X_test, pad_value=vocab['<PAD>'], max_len=len(X_train_pad[0]))

    y_train_pad, _ = pad_sequences(y_train, pad_value=0, max_len=len(X_train_pad[0]))
    y_test_pad, _ = pad_sequences(y_test, pad_value=0, max_len=len(X_train_pad[0]))

    X_train_pad = torch.tensor(X_train_pad, dtype=torch.long)
    y_train_pad = torch.tensor(y_train_pad, dtype=torch.long)
    X_train_mask = torch.tensor(X_train_mask, dtype=torch.bool)

    X_test_pad = torch.tensor(X_test_pad, dtype=torch.long)
    y_test_pad = torch.tensor(y_test_pad, dtype=torch.long)
    X_test_mask = torch.tensor(X_test_mask, dtype=torch.bool)

    train_loader = DataLoader(
        TensorDataset(X_train_pad, y_train_pad, X_train_mask),
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test_pad, y_test_pad, X_test_mask),
        batch_size=32
    )

    if use_pre_trained:
        embedding_matrix = build_embedding_matrix(vocab, "./glove.6B/glove.6B.50d.txt")
    else:
        embedding_matrix = None

    max_len = len(X_train_pad[0])
    test_labels_trunc = [doc_labels[:max_len] for doc_labels in test_labels]

    return len(vocab), train_loader, test_loader, test_labels_trunc, embedding_matrix
def build_token_dataloader_single_crf(label_type='participants', use_pre_trained=1):
    train_doc_ids = get_doc_ids(split="train", label_type=label_type)
    test_doc_ids = get_doc_ids(split="test", label_type=label_type)

    train_labels, train_tokens, _ = get_all(train_doc_ids, label_type, 'train')
    test_labels, test_tokens, _ = get_all(test_doc_ids, label_type, 'test')

    label2id = {'O': 0, 'B': 1, 'I': 2}

    len_vocab, train_loader, test_loader, test_labels, embedding_matrix = \
        build_token_dataloader_from_docs_crf(
            train_tokens=train_tokens,
            train_labels=train_labels,
            test_tokens=test_tokens,
            test_labels=test_labels,
            label2id=label2id,
            use_pre_trained=use_pre_trained
        )

    return len_vocab, train_loader, test_loader, test_labels, embedding_matrix