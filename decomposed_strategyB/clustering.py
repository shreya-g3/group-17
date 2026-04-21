from dataloader_utils import get_doc_ids, get_all, split_into_sentences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


# Load training documents for all three fields
doc_ids = get_doc_ids(split='train', label_type='participants')
labels, tokens, _ = get_all(doc_ids, label_type='participants', split='train')

# Split documents into individual sentences
all_sentences = []
all_true_labels = []

for doc_tokens, doc_labels in zip(tokens, labels):
    sentences = split_into_sentences(doc_tokens, doc_labels)
    for sent_tokens, sent_labels in sentences:
        all_sentences.append(" ".join(sent_tokens))
        has_label = any(l in ['B', 'I'] for l in sent_labels)
        all_true_labels.append(1 if has_label else 0)


# Convert sentences to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2))
X = vectorizer.fit_transform(all_sentences)

# Run k-means with 3 clusters (one per field ideally)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)
cluster_labels = kmeans.labels_


# Reduce to 2D using PCA so we can plot it
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X.toarray())

# Plot
plt.figure(figsize=(8, 5))
colors = ['red', 'blue', 'green']
for i in range(3):
    mask = cluster_labels == i
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                c=colors[i], label=f'Cluster {i}', 
                alpha=0.4, s=10)

plt.title('K-means clustering of sentences (k=3)')
plt.xlabel('PCA dimension 1')
plt.ylabel('PCA dimension 2')
plt.legend()
plt.tight_layout()
plt.savefig('clustering_plot.png')
plt.show()
print("Plot saved as clustering_plot.png")


print(f"\nTotal sentences: {len(all_sentences)}")
print(f"Cluster sizes:")
for i in range(3):
    count = sum(cluster_labels == i)
    print(f"  Cluster {i}: {count} sentences")