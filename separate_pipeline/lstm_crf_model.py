import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTMCRFTagger(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, embedding_matrix=None, pad_idx=0):
        super().__init__()

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float),
                freeze=False,
                padding_idx=pad_idx
            )
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, labels=None, mask=None):
        x = self.embedding(input_ids)          # (B, L, E)
        x, _ = self.lstm(x)                    # (B, L, 2H)
        emissions = self.classifier(x)         # (B, L, C)

        if labels is not None:
            log_likelihood = self.crf(emissions, labels, mask=mask, reduction='mean')
            loss = -log_likelihood
            return loss
        else:
            pred_ids = self.crf.decode(emissions, mask=mask)
            return pred_ids


def train_model_crf(model, train_loader, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in train_loader:
            input_ids, labels, mask = batch

            loss = model(input_ids, labels=labels, mask=mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model


def predict_crf(model, data_loader):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, labels, mask = batch
            pred_ids = model(input_ids, labels=None, mask=mask)
            all_preds.extend(pred_ids)

    return all_preds