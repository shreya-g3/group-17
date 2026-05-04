"""
clustering.py (v2 - all 3 PICO fields)
---------------------------------------
Preliminary clustering analysis for Task 2: Structured Information Extraction.

Coursework requirement:
"Before extraction, use k-means or HAC on sentence embeddings to see whether
natural clusters correspond to schema fields."

This script:
1. Loads sentences from ALL THREE fields (Participants, Interventions, Outcomes)
2. Labels each sentence by its TRUE field
3. Runs K-means with k=3 (one cluster per schema field)
4. Also runs HAC (Hierarchical Agglomerative Clustering) for comparison
5. Measures how well clusters align with true fields (ARI, Homogeneity)
6. Produces a two-panel plot comparing cluster assignment vs true field labels

USAGE:
    python clustering.py
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, homogeneity_score

from dataloader_utils import get_doc_ids, get_all, split_into_sentences


# ============================================================
# Configuration
# ============================================================
FIELDS = ['participants', 'interventions', 'outcomes']
FIELD_COLORS = {
    'participants':  '#4e79a7',   # blue
    'interventions': '#f28e2b',   # orange
    'outcomes':      '#59a14f',   # green
}
CLUSTER_COLORS = ['#e15759', '#76b7b2', '#edc948']  # red, teal, yellow
MAX_DOCS = 1000   # use first 1000 docs for speed


# ============================================================
# Step 1: Find common doc_ids across all 3 fields
# ============================================================
print("Step 1: Finding common doc_ids across all 3 PICO fields...")

ids_per_field = {}
for field in FIELDS:
    ids_per_field[field] = set(get_doc_ids(split='train', label_type=field))

# Intersection: only docs that have annotations for ALL three fields
common_ids = sorted(
    ids_per_field['participants']
    & ids_per_field['interventions']
    & ids_per_field['outcomes']
)

doc_ids = common_ids[:MAX_DOCS]
print(f"  Total common docs: {len(common_ids)}, using first {len(doc_ids)}")


# ============================================================
# Step 2: Load sentences from all 3 fields
# ============================================================
print("\nStep 2: Loading PICO-containing sentences from all 3 fields...")

all_sentences = []
all_field_labels = []  # 0=participants, 1=interventions, 2=outcomes

for field_idx, field in enumerate(FIELDS):
    print(f"  Loading {field}...")
    labels, tokens, _ = get_all(doc_ids, label_type=field, split='train')

    count = 0
    for doc_tokens, doc_labels in zip(tokens, labels):
        # Safety: skip mismatched docs
        if len(doc_tokens) != len(doc_labels):
            continue

        sentences = split_into_sentences(doc_tokens, doc_labels)
        for sent_tokens, sent_labels in sentences:
            # Only use sentences that actually contain a span for this field
            has_span = any(l in ['B', 'I'] for l in sent_labels)
            if has_span:
                all_sentences.append(" ".join(sent_tokens))
                all_field_labels.append(field_idx)
                count += 1

    print(f"    {count} PICO-containing sentences")

all_field_labels = np.array(all_field_labels)
print(f"\nTotal sentences: {len(all_sentences)}")
for i, field in enumerate(FIELDS):
    n = sum(all_field_labels == i)
    print(f"  {field}: {n}")


# ============================================================
# Step 3: TF-IDF vectorisation
# ============================================================
print("\nStep 3: Vectorising with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3
)
X = vectorizer.fit_transform(all_sentences)
print(f"  Feature matrix: {X.shape[0]} sentences x {X.shape[1]} features")


# ============================================================
# Step 4: PCA for 2D visualisation
# ============================================================
print("\nStep 4: PCA reduction to 2D...")
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X.toarray())
var = pca.explained_variance_ratio_
print(f"  Variance explained: PC1={var[0]:.1%}, PC2={var[1]:.1%}, Total={var.sum():.1%}")


# ============================================================
# Step 5: K-means (k=3)
# ============================================================
print("\nStep 5: Running K-means (k=3)...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
kmeans_labels = kmeans.fit_predict(X)

ari_km = adjusted_rand_score(all_field_labels, kmeans_labels)
hom_km = homogeneity_score(all_field_labels, kmeans_labels)
print(f"  Cluster sizes: {[int(sum(kmeans_labels==i)) for i in range(3)]}")
print(f"  Adjusted Rand Index: {ari_km:.4f}  (0=random, 1=perfect)")
print(f"  Homogeneity:         {hom_km:.4f}")

# Cross-tabulation
print("\n  Cluster composition (% of each field's sentences per cluster):")
print(f"  {'':20s} {'Cluster 0':>10} {'Cluster 1':>10} {'Cluster 2':>10}")
for i, field in enumerate(FIELDS):
    mask = all_field_labels == i
    total = mask.sum()
    pcts = [f"{100*int((kmeans_labels[mask]==c).sum())/total:.0f}%" for c in range(3)]
    print(f"  {field:20s} {pcts[0]:>10} {pcts[1]:>10} {pcts[2]:>10}")


# ============================================================
# Step 6: HAC (k=3, Ward linkage) on subsample
# ============================================================
print("\nStep 6: Running HAC (k=3, Ward linkage)...")
# HAC is memory-intensive — subsample for speed
n_sub = min(3000, len(all_sentences))
rng = np.random.default_rng(42)
idx = rng.choice(len(all_sentences), n_sub, replace=False)
X_sub = X[idx].toarray()
labels_sub = all_field_labels[idx]

hac = AgglomerativeClustering(n_clusters=3, linkage='ward')
hac_labels = hac.fit_predict(X_sub)

ari_hac = adjusted_rand_score(labels_sub, hac_labels)
hom_hac = homogeneity_score(labels_sub, hac_labels)
print(f"  Cluster sizes (subsample n={n_sub}): {[int(sum(hac_labels==i)) for i in range(3)]}")
print(f"  Adjusted Rand Index: {ari_hac:.4f}")
print(f"  Homogeneity:         {hom_hac:.4f}")


# ============================================================
# Step 7: Two-panel plot
# ============================================================
print("\nStep 7: Generating two-panel plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Sentence Clustering vs PICO Schema Fields (TF-IDF + PCA)\n"
    "Left: K-means clusters  |  Right: True PICO field labels",
    fontsize=12, fontweight='bold'
)

# Panel A — K-means clusters
ax = axes[0]
for c in range(3):
    mask = kmeans_labels == c
    ax.scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        c=CLUSTER_COLORS[c],
        label=f'Cluster {c}  (n={mask.sum()})',
        alpha=0.35, s=7, linewidths=0
    )
ax.set_title(
    f'K-means (k=3)\nARI={ari_km:.3f}  Homogeneity={hom_km:.3f}',
    fontsize=10
)
ax.set_xlabel(f'PC1 ({var[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({var[1]:.1%} variance)')
ax.legend(fontsize=8, markerscale=2)
ax.grid(True, alpha=0.25)

# Panel B — True field labels
ax = axes[1]
for i, field in enumerate(FIELDS):
    mask = all_field_labels == i
    ax.scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        c=FIELD_COLORS[field],
        label=f'{field.capitalize()}  (n={mask.sum()})',
        alpha=0.35, s=7, linewidths=0
    )
ax.set_title(
    'True PICO Field Labels\n(Ground truth for comparison)',
    fontsize=10
)
ax.set_xlabel(f'PC1 ({var[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({var[1]:.1%} variance)')
ax.legend(fontsize=8, markerscale=2)
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('clustering_plot.png', dpi=150, bbox_inches='tight')
print("  Saved: clustering_plot.png")
plt.show()


# ============================================================
# Step 8: Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Method':<12} {'ARI':>8} {'Homogeneity':>14}")
print(f"{'K-means':<12} {ari_km:>8.4f} {hom_km:>14.4f}")
print(f"{'HAC':<12} {ari_hac:>8.4f} {hom_hac:>14.4f}")
print()

if ari_km < 0.05:
    finding = (
        "Clusters do NOT correspond to PICO schema fields (ARI ≈ 0).\n"
        "  Sentence vocabulary overlaps heavily across P/I/O categories.\n"
        "  This motivates supervised extraction (BiLSTM/LLM) over clustering."
    )
elif ari_km < 0.2:
    finding = (
        "Clusters show WEAK correspondence to PICO fields.\n"
        "  Some structure exists but overlap is substantial.\n"
        "  Supervised models are needed for reliable extraction."
    )
else:
    finding = (
        "Clusters show MODERATE correspondence to PICO fields.\n"
        "  Some vocabulary distinction exists across fields."
    )

print(f"FINDING: {finding}")