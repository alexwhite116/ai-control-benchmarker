#!/usr/bin/env python3
"""
AI Control & Framework Benchmarking Tool
-----------------------------------------
Maps an organisation's own AI/tech control library against reference
frameworks (NIST AI RMF, EU AI Act high-risk obligations, ISO 42001-style
topic areas) and flags gaps where no good match exists.

Usage:
    python benchmark.py --controls data/sample_company_controls.csv --framework nist
    python benchmark.py --controls data/sample_company_controls.csv --framework all --engine cross-encoder
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from similarity_engine import EmbeddingSimilarityEngine, TfidfSimilarityEngine, CrossEncoderSimilarityEngine

FRAMEWORK_FILES = {
    "nist": "frameworks/nist_ai_rmf.json",
    "eu_ai_act": "frameworks/eu_ai_act.json",
    "iso42001": "frameworks/iso_42001_topics.json",
}

ENGINES = {
    "tfidf": TfidfSimilarityEngine,
    "embedding": EmbeddingSimilarityEngine,
    "cross-encoder": CrossEncoderSimilarityEngine
}

# Raw scores aren't comparable across engines (see docs/findings.md) - each
# needs its own calibrated default. cross-encoder's 0.36 is evidence-based
# (highest known non-match scored 0.351 during calibration); embedding's 0.42
# is a rough midpoint only - raw bi-encoder cosine similarity was found to be
# unreliable as a threshold signal regardless of where it's set.
DEFAULT_THRESHOLDS = {
    "tfidf": 0.15,
    "embedding": 0.42,
    "cross-encoder": 0.36,
}


class ControlBenchmarker:
    def __init__(self, engine=None):
        self.engine = engine or TfidfSimilarityEngine()

    def load_framework(self, name: str) -> pd.DataFrame:
        path = Path(FRAMEWORK_FILES[name])
        with open(path) as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def load_company_controls(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        required = {"control_id", "description"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required}")
        return df

    def benchmark(self, company_df: pd.DataFrame, framework_df: pd.DataFrame, threshold: float = 0.15) -> pd.DataFrame:
        sim_matrix = self.engine.similarity_matrix(
            framework_df["description"].tolist(),
            company_df["description"].tolist(),
        )

        results = []
        for i, frow in framework_df.iterrows():
            scores = sim_matrix[i]
            best_idx = scores.argmax()
            best_score = scores[best_idx]
            status = "COVERED" if best_score >= threshold else "NEEDS_REVIEW"
            results.append({
                "framework_id": frow["id"],
                "category": frow["category"],
                "framework_control": frow["description"],
                "best_match_control_id": company_df.iloc[best_idx]["control_id"] if best_score >= threshold else None,
                "best_match_description": company_df.iloc[best_idx]["description"] if best_score >= threshold else None,
                "match_score": round(float(best_score), 3),
                "status": status,
            })
        return pd.DataFrame(results).sort_values("match_score", ascending=True, ignore_index=True)

    def summarise(self, results_df: pd.DataFrame, framework_name: str):
        total = len(results_df)
        needs_review = (results_df["status"] == "NEEDS_REVIEW").sum()
        covered = total - needs_review
        print(f"\n=== {framework_name.upper()} ===")
        print(f"Coverage: {covered}/{total} controls matched  |  Needs review: {needs_review}")
        if needs_review > 0:
            print("\nFlagged for review (lowest-scoring first - not confirmed gaps, just unconfirmed):")
            for _, row in results_df[results_df["status"] == "NEEDS_REVIEW"].iterrows():
                print(f"  [{row['framework_id']}] ({row['category']}) score={row['match_score']}")
                print(f"      {row['framework_control']}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark company AI controls against reference frameworks.")
    parser.add_argument("--controls", required=True, help="Path to CSV of company controls (control_id, description).")
    parser.add_argument("--framework", choices=list(FRAMEWORK_FILES.keys()) + ["all"], default="all")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Minimum similarity score to count as covered (0-1). "
                              "Defaults to a value calibrated per-engine (see DEFAULT_THRESHOLDS) "
                              "since raw scores aren't comparable across engines.")
    parser.add_argument("--output", default="output/benchmark_report.csv", help="Where to write the combined CSV report.")
    parser.add_argument("--engine", choices=list(ENGINES.keys()), default="tfidf", help="Similarity backend to use.")
    args = parser.parse_args()

    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLDS[args.engine]

    benchmarker = ControlBenchmarker(engine=ENGINES[args.engine]())
    company_df = benchmarker.load_company_controls(args.controls)

    frameworks_to_run = list(FRAMEWORK_FILES.keys()) if args.framework == "all" else [args.framework]

    all_results = []
    for fw_name in frameworks_to_run:
        framework_df = benchmarker.load_framework(fw_name)
        results_df = benchmarker.benchmark(company_df, framework_df, threshold=threshold)
        results_df.insert(0, "framework", fw_name)
        benchmarker.summarise(results_df, fw_name)
        all_results.append(results_df)

    combined = pd.concat(all_results, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"\nFull report written to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
