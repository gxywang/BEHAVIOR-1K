"""Audit which fixed task categories have complete scene labels for precision grading.

Single-asset-category task synsets are covered by the exporter. Broad task synsets are not guaranteed complete:
unmatched predictions in those categories must remain ungraded, while annotated-instance recall stays usable.
Only labels.json is updated; input files, mask arrays and their ordering are preserved.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@functools.lru_cache(maxsize=1)
def _taxonomy():
    from bddl.object_taxonomy import DEFAULT_HIERARCHY_FILE, ObjectTaxonomy

    return ObjectTaxonomy(), sha256(DEFAULT_HIERARCHY_FILE)


def precision_coverage(case: dict) -> dict:
    taxonomy, taxonomy_hash = _taxonomy()
    task_file = ROOT / "bddl3/bddl/activity_definitions" / case["task"] / "problem0.bddl"
    objects = task_file.read_text().split("(:objects", 1)[1].split("(:init", 1)[0]
    synsets = set(re.findall(r"-\s+([^\s()]+\.n\.\d+)", objects))
    evidence = {}
    for category, phrase in sorted(case["category_prompts"].items()):
        matching = sorted(s for s in synsets if s.partition(".n.")[0] == category)
        if not matching:
            raise ValueError(f"{case['task']}: no task synset for prompt category {category}")
        assets = sorted({a for s in matching for a in taxonomy.get_subtree_categories(s)})
        row = evidence.setdefault(phrase, {"synsets": [], "asset_categories": []})
        row["synsets"] = sorted(set(row["synsets"]) | set(matching))
        row["asset_categories"] = sorted(set(row["asset_categories"]) | set(assets))
    for row in evidence.values():
        row["precision_graded"] = len(row["asset_categories"]) == 1
    complete = sorted(p for p, r in evidence.items() if r["precision_graded"])
    excluded = sorted(set(evidence) - set(complete))
    return {
        "precision_categories": complete,
        "precision_excluded_categories": excluded,
        "precision_coverage": {
            "rule": "only task synsets mapping to exactly one asset category are complete for precision",
            "broad_unmatched_predictions": "ungraded; never false positives",
            "recall_scope": "annotated visible instances; task targets and available same-category distractors",
            "taxonomy_sha256": taxonomy_hash, "task_definition_sha256": sha256(task_file),
            "categories": evidence,
        },
    }


def main(argv=None):
    from visionbench import read_cases

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("visionbench_cases.json"))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    cases = read_cases(args.manifest)
    rows, missing = [], []
    for case in cases:
        directory = args.corpus / case["id"]
        path = directory / "labels.json"
        if not path.exists():
            missing.append(case["id"])
            continue
        protected = [directory / f for f in ("input.json", "input.npz", "labels.npz")]
        hashes = {p.name: sha256(p) for p in protected}
        labels = json.loads(path.read_text())
        labels.update(precision_coverage(case))
        temp = directory / "labels.coverage.tmp"
        temp.write_text(json.dumps(labels, indent=2) + "\n")
        temp.replace(path)
        if hashes != {p.name: sha256(p) for p in protected}:
            raise RuntimeError(f"immutable capture files changed: {case['id']}")
        rows.append({"id": case["id"], "precision_categories": labels["precision_categories"],
                     "precision_excluded_categories": labels["precision_excluded_categories"],
                     "immutable_file_hashes": hashes})
    report = {"ok": not missing, "annotated": len(rows), "expected": len(cases), "missing": missing, "cases": rows}
    (args.corpus / "label_coverage_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)
    if missing and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
