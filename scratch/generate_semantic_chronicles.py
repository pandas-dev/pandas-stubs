#!/usr/bin/env python3
"""Generate the 10 semantic domain chronicles from the K-Means cluster output.

Inputs:
    scratch/semantic_clusters.json  — cluster assignments from run_clustering.sh
Outputs:
    docs/architecture/chronicles/*.md — one chronicle per cluster
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHRONICLES_DIR = REPO_ROOT / "docs" / "architecture" / "chronicles"
CLUSTERS_FILE = REPO_ROOT / "scratch" / "semantic_clusters.json"


def generate_semantic_chronicles() -> None:
    logger.info("Loading Semantic Clusters...")
    with CLUSTERS_FILE.open() as f:
        clusters = json.load(f)

    for f in CHRONICLES_DIR.glob("*.md"):
        f.unlink()

    cluster_titles = {
        "Cluster 0": "Temporal and Offset Algebra",
        "Cluster 1": "GitHub CI and Issue Tracking",
        "Cluster 2": "Typing Assertions and Improvements",
        "Cluster 3": "Multi-Checker Toolchains (Pyright & Pyrefly)",
        "Cluster 4": "General Bug Triage and Issue Closures",
        "Cluster 5": "Mypy and Series Core Fixes",
        "Cluster 6": "Nightly Build and Release Engineering",
        "Cluster 7": "Agentic Prompting and Development (Part 1)",
        "Cluster 8": "Return Type and Assertion Overloads",
        "Cluster 9": "Agentic Prompting and Development (Part 2)",
    }

    logger.info("Writing %s Semantic Chronicles...", len(clusters))

    for i, (cluster_key, data) in enumerate(clusters.items()):
        domain_name = cluster_titles.get(cluster_key, cluster_key)
        slug = (
            domain_name.lower()
            .replace(" ", "-")
            .replace("(", "")
            .replace(")", "")
            .replace("&", "and")
        )
        filename = f"{i:02d}-{slug}-chronicle.md"
        filepath = CHRONICLES_DIR / filename

        pr_numbers = data["prs"]
        top_terms = data["top_terms"]

        markdown = f"""\
# {domain_name} (Semantic NLP Chronicle)

> **Generated via Unsupervised NLP Clustering (TF-IDF + K-Means)**
> This domain was automatically synthesized based on the semantic similarity \
of PR bodies and titles.
> To reproduce this algorithmic classification locally, run:
> ```bash
> cd <repo-root>
> bash scratch/run_clustering.sh
> python scratch/generate_semantic_chronicles.py
> ```

## 1. Domain Signature
- **Defining Terms**: `{'`, `'.join(top_terms)}`
- **Total Historical PRs**: {len(pr_numbers)}

## 2. Synthesized Technical Overview
This domain captures the architectural evolution surrounding \
**{top_terms[0]}** and **{top_terms[1]}**. Rather than relying on fragile \
keyword regex routing, these PRs were mathematically clustered due to their \
shared discussion of `{top_terms[2]}` and related type paradigms.

## 3. Historical PR Index
| PR/Issue Reference |
| :--- |
"""
        for pr in sorted(pr_numbers, reverse=True):
            markdown += f"| pandas-dev/pandas-stubs#{pr} |\n"

        with filepath.open("w", encoding="utf-8") as f:
            f.write(markdown)

    logger.info("Done generating semantic chronicles.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_semantic_chronicles()
