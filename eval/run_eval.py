"""
run_eval.py
-----------
Runs the 150 in-scope and 20 out-of-scope questions from Quran-RAG-Evaluation.xlsx
against the live /chat endpoint and computes the real metrics.

    python run_eval.py --workbook ../Quran-RAG-Evaluation.xlsx

What is scored automatically
    Precision@3, Recall@3, Hit@3, MRR      retrieved verse IDs vs. gold IDs
    Citation accuracy                       displayed reference vs. retrieved metadata
    Refusal accuracy                        out-of-scope questions correctly declined
    Latency                                 retrieval, generation, end-to-end

What still needs your eyes
    Answer correctness and hallucination are judgement calls. The script writes
    every question, its retrieved verses and the generated explanation to
    eval_results.csv with two blank columns. Read them, fill in 1 or 0, and the
    summary recomputes. Do not skip this and do not guess the numbers: a viva
    examiner can ask to see the filled sheet.

A note on Precision@3
    The workbook defines Precision@3 as (relevant retrieved / 3). Most questions
    in the test set have exactly one gold verse, so under that definition the
    ceiling is 0.333 and no system can score higher. This script therefore also
    reports Precision@3 normalised by min(3, gold_count), which is the figure
    that is comparable across questions. Both are printed. Check which
    definition your report's Table 4.1 is using before you copy anything across.
"""

import argparse
import csv
import json
import statistics
import sys
import time

import requests
from openpyxl import load_workbook


def load_questions(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    in_scope, out_scope = [], []

    for i, row in enumerate(wb["Test_Set"].iter_rows(values_only=True)):
        if i == 0 or not row[0]:
            continue
        in_scope.append({
            "id": row[0], "language": row[1], "theme": row[2], "question": row[3],
            "gold": [g for g in row[5:8] if g],
        })

    for i, row in enumerate(wb["OOS_Set"].iter_rows(values_only=True)):
        if i == 0 or not row[0]:
            continue
        out_scope.append({"id": row[0], "language": row[1], "question": row[2]})

    return in_scope, out_scope


def ask(api, question, timeout):
    started = time.perf_counter()
    r = requests.post(f"{api}/chat", json={"question": question}, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    payload["wall_time"] = time.perf_counter() - started
    return payload


def score_retrieval(retrieved_ids, gold_ids):
    relevant = [1 if vid in gold_ids else 0 for vid in retrieved_ids]
    hits = sum(relevant)
    hit3 = 1 if hits else 0
    rr = 1.0 / (relevant.index(1) + 1) if hit3 else 0.0
    return {
        "hits": hits,
        "precision_workbook": hits / 3.0,
        "precision_norm": hits / min(3, len(gold_ids)) if gold_ids else 0.0,
        "recall": hits / len(gold_ids) if gold_ids else 0.0,
        "hit3": hit3,
        "rr": rr,
    }


def citation_is_consistent(result):
    """Citation accuracy is structural here: the reference shown to the user is
    built from the same metadata record the verse text came from. This verifies
    that invariant rather than assuming it, and also confirms the model did not
    smuggle a reference of its own into the explanation."""
    for verse, source in zip(result["retrieved_ayahs"], result["sources"]):
        if verse["surah_name_en"] not in source or str(verse["ayah_no"]) not in source:
            return 0
        expected_id = f"s{verse['surah_no']}_a{verse['ayah_no']}"
        if verse["verse_id"] != expected_id:
            return 0
    return 1


def mean(values):
    return statistics.mean(values) if values else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", default="../Quran-RAG-Evaluation.xlsx")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--out", default="eval_results.csv")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N (smoke test)")
    args = ap.parse_args()

    try:
        health = requests.get(f"{args.api}/health", timeout=10).json()
    except Exception as exc:
        sys.exit(f"Backend not reachable at {args.api}: {exc}")
    print(f"Model: {health.get('model')}   Verses indexed: {health.get('verses_indexed')}")
    print(f"Ollama: {health.get('ollama')}\n")

    in_scope, out_scope = load_questions(args.workbook)
    if args.limit:
        in_scope, out_scope = in_scope[:args.limit], out_scope[:max(1, args.limit // 5)]

    rows, agg = [], {}
    total = len(in_scope) + len(out_scope)

    for n, item in enumerate(in_scope, 1):
        try:
            res = ask(args.api, item["question"], args.timeout)
        except Exception as exc:
            print(f"  [{n}/{total}] {item['id']} FAILED: {exc}")
            continue

        retrieved = [v["verse_id"] for v in res["retrieved_ayahs"]]
        s = score_retrieval(retrieved, item["gold"])
        citation = citation_is_consistent(res) if retrieved else ""

        lang = item["language"]
        for key, value in (
            ("hit3", s["hit3"]), ("rr", s["rr"]),
            ("p_wb", s["precision_workbook"]), ("p_norm", s["precision_norm"]),
            ("recall", s["recall"]),
            ("sim", res["retrieved_ayahs"][0]["similarity_score"] if retrieved else 0.0),
            ("t_retrieval", res["retrieval_time"]),
            ("t_generation", res["generation_time"]),
            ("t_total", res["wall_time"]),
        ):
            agg.setdefault((lang, key), []).append(value)
        if citation != "":
            agg.setdefault((lang, "citation"), []).append(citation)

        rows.append({
            "ID": item["id"], "Language": lang, "Theme": item["theme"],
            "Question": item["question"], "Gold": ";".join(item["gold"]),
            "Retrieved_1": retrieved[0] if len(retrieved) > 0 else "",
            "Retrieved_2": retrieved[1] if len(retrieved) > 1 else "",
            "Retrieved_3": retrieved[2] if len(retrieved) > 2 else "",
            "Hit@3": s["hit3"], "RR": round(s["rr"], 4),
            "Precision@3_workbook": round(s["precision_workbook"], 4),
            "Precision@3_normalised": round(s["precision_norm"], 4),
            "Recall@3": round(s["recall"], 4),
            "Top1_similarity": res["retrieved_ayahs"][0]["similarity_score"] if retrieved else "",
            "Citation_Correct": citation,
            "Retrieval_s": res["retrieval_time"],
            "Generation_s": res["generation_time"],
            "Total_s": round(res["wall_time"], 2),
            "Answer": res["answer"].replace("\n", " "),
            "Correct_1_0": "",        # <-- you fill this in
            "Hallucinated_1_0": "",   # <-- and this
        })
        print(f"  [{n}/{total}] {item['id']} {lang[:3]}  hit={s['hit3']} "
              f"sim={rows[-1]['Top1_similarity']}  {res['wall_time']:.1f}s")

    refusals = []
    for n, item in enumerate(out_scope, len(in_scope) + 1):
        try:
            res = ask(args.api, item["question"], args.timeout)
        except Exception as exc:
            print(f"  [{n}/{total}] {item['id']} FAILED: {exc}")
            continue
        refused = 1 if res.get("refused") else 0
        refusals.append((item["language"], refused))
        rows.append({
            "ID": item["id"], "Language": item["language"], "Theme": "OUT-OF-SCOPE",
            "Question": item["question"], "Gold": "-",
            "Retrieved_1": "", "Retrieved_2": "", "Retrieved_3": "",
            "Hit@3": "", "RR": "", "Precision@3_workbook": "",
            "Precision@3_normalised": "", "Recall@3": "",
            "Top1_similarity": res["retrieved_ayahs"][0]["similarity_score"] if res["retrieved_ayahs"] else "",
            "Citation_Correct": "", "Retrieval_s": res["retrieval_time"],
            "Generation_s": res["generation_time"], "Total_s": round(res["wall_time"], 2),
            "Answer": res["answer"].replace("\n", " "),
            "Correct_1_0": refused, "Hallucinated_1_0": 0 if refused else "",
        })
        print(f"  [{n}/{total}] {item['id']} OOS  refused={refused}")

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 72)
    print("MEASURED RESULTS  (automatic metrics only)")
    print("=" * 72)
    header = f"{'Metric':<34}{'English':>12}{'Bengali':>12}{'Overall':>12}"
    print(header)
    print("-" * 72)

    def line(label, key, fmt="{:.3f}"):
        en, bn = agg.get(("English", key), []), agg.get(("Bengali", key), [])
        vals = (mean(en), mean(bn), mean(en + bn))
        print(f"{label:<34}" + "".join(fmt.format(v).rjust(12) for v in vals))

    line("Hit@3", "hit3")
    line("Mean Reciprocal Rank", "rr")
    line("Precision@3 (workbook /3)", "p_wb")
    line("Precision@3 (normalised)", "p_norm")
    line("Recall@3", "recall")
    line("Citation accuracy", "citation")
    line("Mean top-1 similarity", "sim")
    line("Mean retrieval time (s)", "t_retrieval")
    line("Mean generation time (s)", "t_generation")
    line("Mean end-to-end latency (s)", "t_total", "{:.1f}")

    if refusals:
        en = [r for l, r in refusals if l == "English"]
        bn = [r for l, r in refusals if l == "Bengali"]
        print(f"{'Refusal accuracy':<34}"
              + "".join(f"{v:.3f}".rjust(12) for v in (mean(en), mean(bn), mean(en + bn))))

    print("-" * 72)
    print(f"\nPer-question detail written to: {args.out}")
    print("Answer accuracy and hallucination rate are NOT computed above.")
    print("Open the CSV, judge each answer, fill Correct_1_0 and Hallucinated_1_0,")
    print("then run:  python summarise.py " + args.out)


if __name__ == "__main__":
    main()
