from pathlib import Path
import numpy as np

DATA_DIR = Path("../ebm_nlp_2_00")

docs_dir = DATA_DIR / "documents"

def get_doc_ids(split="train", label_type="participants"):

    if split == "test":
        split = "test/gold"

    train_dir = (
        DATA_DIR
        / "annotations"
        / "aggregated"
        / "hierarchical_labels"
        / label_type  # assuming that the split is the same for all entity types, we can just look at one of them
        / split
    )

    doc_ids = [p.stem.split(".")[0] for p in train_dir.glob("*.AGGREGATED.ann")]
    return sorted(doc_ids)

def load_labels_for_doc(doc_id, label_type="participants", split="train"):
    if split == "test":
        split = "test/gold"

    ann_path = DATA_DIR / "annotations" / "aggregated" / "hierarchical_labels" / label_type / split/ f"{doc_id}.AGGREGATED.ann"

    if not ann_path.exists():
        print(ann_path, "does not exist!")
        return None

    with open(ann_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f]

    return labels

def load_labels(doc_ids, label_type="participants", split="train"):
    labels = []
    for doc_id in doc_ids:
        doc_labels = load_labels_for_doc(doc_id, label_type, split)
        if doc_labels is not None:
            labels.append(doc_labels)
    return labels

def hierarchical_to_bio(tags):

    bio = []
    prev = 0

    for t in tags:
        t = int(t)  # ensure it's an integer
        if t == 0:
            bio.append("O")
        else:
            if prev == 0:
                bio.append("B")
            else:
                bio.append("I")
        prev = t

    return bio

def convert_all_labels_to_bio(labels):
    for i, doc_labels in enumerate(labels):
        labels[i] = hierarchical_to_bio(doc_labels)
    return labels
def load_document(doc_id):
    doc_path = DATA_DIR / "documents" / f"{doc_id}.tokens"
    with open(doc_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def load_documents(doc_ids):
    documents = []
    for doc_id in doc_ids:
        doc = load_document(doc_id)
        documents.append(doc)
    return documents
def bio_to_span(token, label):
    res = []
    now = []
    for tok, lab in zip(token, label):
        if lab == 'B':
            if now:
                res.append(now)
                now = []
            now = [tok]
        elif lab == 'I':
            if now:
                now.append(tok)
            else:
                now=[tok]
        else:
            if now:
                res.append(now)
                now = []

    if now:
        res.append(now)
        now = []

    return res

def bios_to_spans(tokens, labels):
    res = []
    for tok, lab in zip(tokens, labels):
        tmp = bio_to_span(tok, lab)
        res.append(tmp)
    return res
def get_all(doc_ids, label_type='participants', split='train'):
    labels = load_labels(doc_ids, label_type=label_type, split=split)
    labels = convert_all_labels_to_bio(labels)
    tokens = load_documents(doc_ids)
    spans = bios_to_spans(tokens, labels)
    return labels, tokens, spans
def encode_labels(labels, label2id):
    encoded = []
    for doc_labels in labels:
        encoded.append([label2id[x] for x in doc_labels])
    return encoded
def build_vocab(all_tokens, min_freq=1):
    freq = {}
    for doc in all_tokens:
        for tok in doc:
            tok = tok.lower()
            freq[tok] = freq.get(tok, 0) + 1

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for tok, c in freq.items():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab
def encode_tokens(tokens, vocab):
    encoded = []
    for doc in tokens:
        encoded.append([vocab.get(tok.lower(), vocab['<UNK>']) for tok in doc])
    return encoded
def pad_sequences(seqs, pad_value=0, max_len=None):
    if max_len is None:
        max_len = max(len(x) for x in seqs)

    padded = []
    masks = []
    for seq in seqs:
        seq = seq[:max_len]
        pad_len = max_len - len(seq)
        padded.append(seq + [pad_value] * pad_len)
        masks.append([1] * len(seq) + [0] * pad_len)
    return padded, masks
def decode_predictions(pred_ids, id2label, masks):
    all_preds = []

    for doc_preds, doc_mask in zip(pred_ids, masks):
        cur_labels = []

        for p, m in zip(doc_preds, doc_mask):
            if m == 1:  # 只保留真实token
                cur_labels.append(id2label[int(p)])

        all_preds.append(cur_labels)

    return all_preds
def flatten_labels(labels):
    flat = []
    for doc in labels:
        flat.extend(doc)
    return flat
def split_into_sentences(tokens, labels=None):
    sentences = []
    cur_tokens = []
    cur_labels = [] if labels is not None else None

    for i, tok in enumerate(tokens):
        cur_tokens.append(tok)
        if labels is not None:
            cur_labels.append(labels[i])

        if tok in ['.', '?', '!']:
            if labels is not None:
                sentences.append((cur_tokens, cur_labels))
            else:
                sentences.append(cur_tokens)

            cur_tokens = []
            if labels is not None:
                cur_labels = []

    # 处理最后一句没标点结束的情况
    if cur_tokens:
        if labels is not None:
            sentences.append((cur_tokens, cur_labels))
        else:
            sentences.append(cur_tokens)

    return sentences
def sentence_label_from_bio(sent_labels):
    for lab in sent_labels:
        if lab in ['B', 'I']:
            return 1
    return 0
def apply_mask_to_labels(labels, masks):
    new_labels = []
    for doc_labels, doc_mask in zip(labels, masks):
        cur = []
        for lab, m in zip(doc_labels, doc_mask):
            if m == 1:
                cur.append(lab)
        new_labels.append(cur)
    return new_labels
