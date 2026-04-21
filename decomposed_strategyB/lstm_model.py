import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)
        pt = torch.exp(-loss)
        focal_loss = ((1 - pt) ** self.gamma) * loss
        return focal_loss.mean()

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, embedding_matrix=None,pad_idx=0):
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

    def forward(self, input_ids):
        x = self.embedding(input_ids)          # (B, L, E)
        x, _ = self.lstm(x)                    # (B, L, 2H)
        logits = self.classifier(x)            # (B, L, C)
        return logits

def train_model(model, train_loader, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # weights = torch.tensor([1.0, 5.0, 5.0])  # O, B, I
    # criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    #baseline
    #criterion = nn.CrossEntropyLoss(ignore_index=-100)
    # #Focal loss
    criterion = FocalLoss(gamma=2)


    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            input_ids, labels = batch

            logits = model(input_ids)

            loss = criterion(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model

def predict(model, data_loader):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, _ = batch

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())

    return all_preds