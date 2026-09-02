"""Diagnostic eval for the TextToSQL agent.

Runs the agent over the gold sets and classifies every result — pass, SQL
error, wrong result, no-SQL — so you can see red-flag *patterns* (a whole
category failing, format mismatches, missing constraints) instead of a
single aggregate number.

Usage:  uv run python -m src.diagnose_sql
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

import config
from src.dspy_eval import configure_dspy  # sets up the LM
from src.sql_llm.sql_agent import TextToSQL
from src.sql_llm.sql_pokeapi import run_query  # your read-only executor

SIMPLE = config.ROOT_DIR / "eval" / "simple_eval_set.json"
COMPLEX = config.ROOT_DIR / "eval" / "complex_eval_set.json"


# ---- normalization (must match your metric exactly) ----
def _norm_cell(v) -> str:
    s = str(v).strip().lower()
    s = re.sub(r"[,\s]", "", s)
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    return s


def _result_set(rows) -> set:
    return {_norm_cell(c) for row in rows for c in row}


# ---- constraint sniffers: look at the generated SQL for common omissions ----
def sql_red_flags(sql: str) -> list[str]:
    flags = []
    low = (sql or "").lower()
    if not low.strip():
        return ["EMPTY_SQL"]
    if "is_default" not in low:
        flags.append("missing_is_default")
    # generation asked but not filtered
    return flags


def classify(example, sql, model_rows, gold_rows, error):
    """Return (verdict, detail) for one example."""
    if not sql or not sql.strip():
        return "NO_SQL", "agent produced no query"
    if error is not None:
        return "SQL_ERROR", error.split("\n")[0][:100]

    gold, model = _result_set(gold_rows), _result_set(model_rows)
    if gold == model:
        return "PASS", ""
    if gold and model and gold.issubset(model):
        return "WRONG_SUPERSET", "returned gold + extra rows (missing LIMIT/filter?)"
    if gold and model and model.issubset(gold):
        return "WRONG_SUBSET", "returned only part of the gold set"
    if not model:
        return "WRONG_EMPTY", "query ran but returned nothing"
    # right values present but wrapped differently? check overlap
    overlap = gold & model
    if overlap:
        return "WRONG_PARTIAL", f"overlap={sorted(overlap)[:3]} gold={sorted(gold)[:3]}"
    return "WRONG_DISJOINT", f"model={sorted(model)[:3]} gold={sorted(gold)[:3]}"


def run_diagnostic(agent, examples, label):
    verdicts = Counter()
    flags = Counter()
    detail = defaultdict(list)  # verdict -> list of (question, sql, note)

    for ex in tqdm(examples, desc=label, unit="q"):
        q = ex["question"]
        gold_rows = ex["gold_answer"]
        try:
            pred = agent(question=q)
            sql = getattr(pred, "sql", "") or ""
        except Exception as e:
            verdicts["AGENT_CRASH"] += 1
            detail["AGENT_CRASH"].append((q, "", str(e)[:120]))
            continue

        error, model_rows = None, []
        if sql.strip():
            try:
                _cols, model_rows = run_query(sql)
            except Exception as e:
                error = str(e)

        verdict, note = classify(ex, sql, model_rows, gold_rows, error)
        verdicts[verdict] += 1
        for f in sql_red_flags(sql):
            flags[f] += 1
        if verdict != "PASS":
            detail[verdict].append((q, sql, note))

    # ---- report ----
    total = len(examples)
    passed = verdicts["PASS"]
    print(
        f"\n{'=' * 70}\n{label}: {passed}/{total} pass ({passed / total:.0%})\n{'=' * 70}"
    )
    print("Verdict breakdown:")
    for v, n in verdicts.most_common():
        print(f"  {v:<16} {n:>3}  ({n / total:.0%})")
    print("\nSQL red flags (across all queries):")
    for f, n in flags.most_common():
        print(f"  {f:<20} {n:>3}")

    return verdicts, detail


def main():
    lm = configure_dspy()
    print(lm.model)
    agent = TextToSQL()

    simple = json.loads(Path(SIMPLE).read_text())
    complex_ = json.loads(Path(COMPLEX).read_text())

    _, ds = run_diagnostic(agent, simple, "SIMPLE")
    _, dc = run_diagnostic(agent, complex_, "COMPLEX")

    # ---- dump the actual failing cases so you can read them ----
    out = config.ROOT_DIR / "sql_failures_smaller_model.txt"
    with out.open("w") as f:
        for label, detail in [("SIMPLE", ds), ("COMPLEX", dc)]:
            for verdict, items in detail.items():
                f.write(
                    f"\n\n########## {label} / {verdict} ({len(items)}) ##########\n"
                )
                for q, sql, note in items:
                    f.write(f"\nQ: {q}\n  note: {note}\n  SQL: {sql}\n")
    print(f"\nFull failing cases written to {out}")


if __name__ == "__main__":
    main()
