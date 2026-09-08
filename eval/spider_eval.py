"""
Execution-accuracy evaluation of DataWizard's SQL-generation model against a
stratified sample of the Spider dev set (https://yale-lily.github.io/spider).

For each sampled question:
  1. Introspect the question's own SQLite database (same approach as
     tools/sql_tools.get_sql_schema — schema is read from the DB itself,
     not from Spider's tables.json).
  2. Ask the model (same model/temperature DataWizard's agent uses) to write
     a single SQLite SELECT answering the question, given that schema.
  3. Execute both the predicted SQL and Spider's gold SQL against the DB and
     compare result sets (execution accuracy: order-sensitive only if the
     gold query has an ORDER BY, matching Spider's own convention).

Usage:
    ./datawizard-env/bin/python eval/spider_eval.py --n-per-bucket 35
"""
import argparse
import csv
import json
import os
import random
import sqlite3
import sys
import time
import tomllib
from itertools import permutations
from pathlib import Path

from groq import Groq

from spider_hardness import eval_hardness

ROOT = Path(__file__).parent
SPIDER_DIR = ROOT / "spider_data"
RESULTS_DIR = ROOT / "results"

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a SQLite expert. Given a database schema and a question, \
write ONE valid SQLite SELECT query that answers the question.

Rules:
- Output ONLY the raw SQL. No explanation, no markdown fences, no commentary.
- Use only tables/columns that appear in the schema.
- Write a single SELECT statement (no semicolon-separated multi-statements).
"""


def get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    secrets_path = ROOT.parent / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with open(secrets_path, "rb") as f:
            return tomllib.load(f).get("GROQ_API_KEY", "")
    return ""


def get_schema_text(db_path: Path) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cur.fetchall()]
    parts = []
    for table in tables:
        cur.execute(f'PRAGMA table_info("{table}");')
        cols = cur.fetchall()
        col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        parts.append(f"Table: {table}\n  Columns: {col_defs}")
    con.close()
    return "\n\n".join(parts)


def strip_sql_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text.rstrip(";").strip()


def run_query(db_path: Path, sql: str):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.execute("PRAGMA busy_timeout = 5000;")
    cur = con.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        return rows, None
    except Exception as e:
        return None, str(e)
    finally:
        con.close()


def normalize_row(row):
    return tuple(round(v, 4) if isinstance(v, float) else v for v in row)


def _sequences_match(pred, gold, ordered: bool) -> bool:
    if ordered:
        return pred == gold
    return sorted(pred) == sorted(gold)


def results_match(pred_rows, gold_rows, ordered: bool) -> bool:
    """Column order isn't semantically meaningful in SQL SELECT results, so —
    matching Spider's own execution-accuracy convention — try every
    permutation of the predicted columns against gold before failing.
    """
    if pred_rows is None:
        return False
    pred = [normalize_row(r) for r in pred_rows]
    gold = [normalize_row(r) for r in gold_rows]

    n_cols = len(pred[0]) if pred else 0
    if n_cols == 0 or n_cols != (len(gold[0]) if gold else 0) or n_cols > 7:
        return _sequences_match(pred, gold, ordered)

    for perm in permutations(range(n_cols)):
        reordered = [tuple(row[i] for i in perm) for row in pred]
        if _sequences_match(reordered, gold, ordered):
            return True
    return False


def call_model(client: Groq, schema: str, question: str, retries: int = 3) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ]
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return ""


def stratified_sample(dev_data, n_per_bucket: int, seed: int):
    buckets = {"easy": [], "medium": [], "hard": [], "extra": []}
    for i, ex in enumerate(dev_data):
        buckets[eval_hardness(ex["sql"])].append(i)

    rng = random.Random(seed)
    sampled = []
    for level, idxs in buckets.items():
        rng.shuffle(idxs)
        take = idxs[:n_per_bucket]
        sampled.extend((i, level) for i in take)
    rng.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-bucket", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=0.2, help="seconds between API calls")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        sys.exit("No GROQ_API_KEY found (env var or .streamlit/secrets.toml).")
    client = Groq(api_key=api_key)

    dev_data = json.load(open(SPIDER_DIR / "dev.json"))
    sample = stratified_sample(dev_data, args.n_per_bucket, args.seed)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_csv = RESULTS_DIR / "spider_eval_results.csv"
    rows_out = []

    total = len(sample)
    correct = 0
    for n, (idx, level) in enumerate(sample, 1):
        ex = dev_data[idx]
        db_id = ex["db_id"]
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"
        gold_sql = ex["query"]
        question = ex["question"]
        ordered = "order by" in gold_sql.lower()

        row = {
            "idx": idx, "db_id": db_id, "difficulty": level,
            "question": question, "gold_sql": gold_sql,
            "pred_sql": "", "correct": False, "error": "",
        }

        try:
            schema = get_schema_text(db_path)
            raw = call_model(client, schema, question)
            pred_sql = strip_sql_fences(raw)
            row["pred_sql"] = pred_sql

            gold_rows, gold_err = run_query(db_path, gold_sql)
            pred_rows, pred_err = run_query(db_path, pred_sql)

            if pred_err:
                row["error"] = f"pred_exec_error: {pred_err}"
            elif gold_err:
                row["error"] = f"gold_exec_error: {gold_err}"
            else:
                row["correct"] = results_match(pred_rows, gold_rows, ordered)
        except Exception as e:
            row["error"] = f"harness_error: {e}"

        if row["correct"]:
            correct += 1
        rows_out.append(row)

        print(f"[{n}/{total}] {db_id:30s} {level:8s} "
              f"{'OK' if row['correct'] else 'FAIL':4s}  "
              f"running acc={correct/n:.3f}")
        time.sleep(args.sleep)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    by_level = {}
    for row in rows_out:
        d = by_level.setdefault(row["difficulty"], {"correct": 0, "total": 0})
        d["total"] += 1
        d["correct"] += int(row["correct"])

    summary_lines = [
        f"Model: {MODEL}",
        f"Sample size: {total} (stratified, {args.n_per_bucket}/bucket, seed={args.seed})",
        f"Overall execution accuracy: {correct}/{total} = {correct/total:.3%}",
        "",
        "By difficulty:",
    ]
    for level in ["easy", "medium", "hard", "extra"]:
        d = by_level.get(level, {"correct": 0, "total": 0})
        if d["total"]:
            summary_lines.append(f"  {level:8s}: {d['correct']:3d}/{d['total']:3d} = {d['correct']/d['total']:.3%}")

    summary = "\n".join(summary_lines)
    print("\n" + summary)
    (RESULTS_DIR / "spider_eval_summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
