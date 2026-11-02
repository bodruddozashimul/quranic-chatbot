"""
summarise.py
------------
Recomputes the headline table after you have judged the answers by hand.

    python summarise.py eval_results.csv

Produces exactly the rows of Table 4.1 in the report, so the numbers can be
transcribed without arithmetic in between.
"""

import csv
import statistics
import sys


def mean(v):
    return statistics.mean(v) if v else 0.0


def main(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    scope = [r for r in rows if r["Theme"] != "OUT-OF-SCOPE"]
    oos = [r for r in rows if r["Theme"] == "OUT-OF-SCOPE"]

    unjudged = [r["ID"] for r in scope if r["Correct_1_0"] == ""]
    if unjudged:
        print(f"WARNING: {len(unjudged)} in-scope answers are still unjudged "
              f"(first few: {', '.join(unjudged[:5])}).")
        print("Those rows are excluded, so the accuracy below is over a partial set.\n")

    def by_lang(rows_, key, cast=float):
        out = {}
        for lang in ("English", "Bengali"):
            vals = [cast(r[key]) for r in rows_ if r["Language"] == lang and r[key] != ""]
            out[lang] = vals
        out["Overall"] = out["English"] + out["Bengali"]
        return out

    print(f"{'Metric':<34}{'English':>12}{'Bengali':>12}{'Overall':>12}")
    print("-" * 72)

    def row(label, data, pct=False, fmt="{:.3f}"):
        cells = []
        for k in ("English", "Bengali", "Overall"):
            v = mean(data[k])
            cells.append((f"{v * 100:.1f}" if pct else fmt.format(v)).rjust(12))
        print(f"{label:<34}" + "".join(cells))

    row("Answer accuracy (%)", by_lang(scope, "Correct_1_0"), pct=True)
    row("Citation accuracy (%)", by_lang(scope, "Citation_Correct"), pct=True)
    row("Retrieval Precision@3 (norm.)", by_lang(scope, "Precision@3_normalised"))
    row("Retrieval Precision@3 (/3)", by_lang(scope, "Precision@3_workbook"))
    row("Retrieval Recall@3", by_lang(scope, "Recall@3"))
    row("Hit@3", by_lang(scope, "Hit@3"))
    row("Mean Reciprocal Rank", by_lang(scope, "RR"))
    row("Hallucination rate (%)", by_lang(scope, "Hallucinated_1_0"), pct=True)
    row("Refusal accuracy (%)", by_lang(oos, "Correct_1_0"), pct=True)
    print("-" * 72)
    row("Mean top-1 cosine similarity", by_lang(scope, "Top1_similarity"))
    row("Mean retrieval time (s)", by_lang(scope, "Retrieval_s"))
    row("Mean end-to-end latency (s)", by_lang(scope, "Total_s"), fmt="{:.1f}")
    print(f"\nJudged {len([r for r in scope if r['Correct_1_0'] != ''])} "
          f"of {len(scope)} in-scope questions.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_results.csv")
