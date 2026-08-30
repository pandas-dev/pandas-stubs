#!/usr/bin/env bash
set -e

# Reproducible NLP clustering pipeline for pandas-stubs architecture chronicles.
# Requires: uv (https://docs.astral.sh/uv/)
#
# Usage:
#   cd <repo-root>
#   bash scratch/run_clustering.sh
#
# Inputs:
#   scratch/raw_prs.jsonl          — one JSON object per merged PR
# Outputs:
#   scratch/semantic_clusters.json — cluster assignments and top terms

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Semantic PR Clustering Pipeline ==="
echo "Script dir : $SCRIPT_DIR"
echo "Repo root  : $REPO_ROOT"

VENV_DIR="$SCRIPT_DIR/.venv"
echo "Creating dedicated uv environment at $VENV_DIR ..."
uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install scikit-learn numpy

cat << 'PYEOF' > "$SCRIPT_DIR/semantic_pr_router.py"
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

script_dir = Path(__file__).resolve().parent
raw_prs = script_dir / "raw_prs.jsonl"

prs = []
with open(raw_prs) as f:
    for line in f:
        if line.strip():
            prs.append(json.loads(line))

corpus = []
pr_numbers = []
for p in prs:
    text = (p.get("title", "") + " " + (p.get("body", "") or "")).replace("\r", " ").replace("\n", " ")
    corpus.append(text)
    pr_numbers.append(p["number"])

print(f"Running semantic analysis on {len(corpus)} PRs...")

vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(corpus)

k = 10
model = KMeans(n_clusters=k, random_state=42, n_init=10)
model.fit(X)

terms = vectorizer.get_feature_names_out()
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

cluster_map = {}
for i in range(k):
    top_terms = [terms[ind] for ind in order_centroids[i, :7]]
    cluster_map[f"Cluster {i}"] = {"top_terms": top_terms, "prs": []}

for pr_idx, cluster_idx in enumerate(model.labels_):
    cluster_map[f"Cluster {cluster_idx}"]["prs"].append(pr_numbers[pr_idx])

out = script_dir / "semantic_clusters.json"
with open(out, "w") as f:
    json.dump(cluster_map, f, indent=2)

print(f"\nWrote {out}")
print("\n=== Semantic Clustering Results ===")
for cluster, data in cluster_map.items():
    print(f"{cluster} (Size: {len(data['prs'])}): {', '.join(data['top_terms'])}")
PYEOF

echo "Running NLP clustering model..."
python "$SCRIPT_DIR/semantic_pr_router.py"
echo "Done."
