"""Stop the loop, then answer.

For each MATH-500 problem with a released first sample (Sparks of WSP, Experiment 8), find the
first point where the repetition detector fires inside the thinking. Cut the trace there,
append the end-of-thinking token and let the same model write its answer, with the
evaluation's sampling (temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5 counting the
sample's earlier tokens), within the original 32,768-token budget. A trace that hit the budget
without a flag is cut 2,048 tokens before it. Every other trace keeps its released outcome.

    python experiments/stop_and_answer.py --logs <lm_understand-logs>/exp8_math --model unedited
    python experiments/stop_and_answer.py --logs ... --model unedited --check-flags-only

Writes results/stop_and_answer_<model>.jsonl (one line per problem) and prints a summary.
"""
import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BUDGET, ANSWER_MAX = 32_768, 2_048
MATH500 = ("HuggingFaceH4/MATH-500", "test.jsonl", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be")


def released(logs, model):
    texts, grades = {}, {}
    for shard in range(10):
        part = logs / f"eval/{model}/parts/math500_s{shard}of10"
        with gzip.open(part / "math500_first_samples.jsonl.gz", "rt") as f:
            for line in f:
                r = json.loads(line)
                texts[r["id"]] = r["text"]
        for r in json.loads((part / "math500_per_problem.json").read_text()):
            grades[r["id"]] = r["correct"]
    return texts, grades


def first_flag(ids, detector):
    for end in range(detector.window, len(ids) + 1, detector.step):
        if detector.looping(ids[:end]):
            return end
    return None


def plan(tok, texts, grades, answer_max=ANSWER_MAX):
    """Which traces get the rule, and where they are cut."""
    from wspserve.engine import LoopDetector

    detector = LoopDetector()
    close_len = len(tok("\n</think>\n\n", add_special_tokens=False)["input_ids"])
    out = []
    for pid in sorted(texts):
        text = texts[pid]
        closed = "</think>" in text
        thinking = tok(text.split("</think>", 1)[0], add_special_tokens=False)["input_ids"]
        flag = first_flag(thinking, detector)
        if flag is not None:
            rule, cut = "loop_flag", flag
        elif not closed:
            rule, cut = "budget", BUDGET - answer_max - close_len
        else:
            rule, cut = "none", None
        out.append({"id": pid, "rule": rule, "cut": cut, "flag": flag, "closed": closed,
                    "original_correct": bool(grades[pid][0]), "thinking_ids": thinking})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True, help="exp8_math folder of RikiyaT/lm_understand-logs")
    parser.add_argument("--out", type=Path, help="default: results/stop_and_answer_<model>.jsonl")
    parser.add_argument("--model", default="unedited", choices=["unedited", "p10c", "p25c", "p50c"])
    parser.add_argument("--answer-max", type=int, default=ANSWER_MAX)
    parser.add_argument("--check-flags-only", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    from wspserve.engine import REPO as QWEN, REVISION

    tok = AutoTokenizer.from_pretrained(QWEN, revision=REVISION)
    texts, grades = released(args.logs, args.model)
    rows = plan(tok, texts, grades, args.answer_max)
    print(f"{args.model}: {len(rows)} problems, {sum(r['rule'] != 'none' for r in rows)} get the rule "
          f"({sum(r['rule'] == 'loop_flag' for r in rows)} at a loop flag)")
    if args.check_flags_only:
        print(json.dumps({r["id"]: r["flag"] for r in rows}))
        return

    from wspserve.engine import Engine, Row, Sampling
    from wspserve.server import SUFFIX, grade

    problems = {}
    for line in Path(hf_hub_download(*MATH500[:2], repo_type="dataset", revision=MATH500[2])).read_text().splitlines():
        r = json.loads(line)
        problems["test/" + r["unique_id"].split("test/", 1)[-1]] = r
    engine = Engine()
    version = "base"
    if args.model != "unedited":
        sys.path.insert(0, str(REPO / "experiments"))
        from released_edits import load_final_edit
        version = engine.register_float32(load_final_edit(engine, args.logs, args.model))

    close = tok("\n</think>\n\n", add_special_tokens=False)["input_ids"]
    todo = []
    for r in rows:
        if r["rule"] == "none":
            continue
        ids = r["thinking_ids"][:r["cut"]]
        prompt = tok.apply_chat_template([{"role": "user", "content": problems[r["id"]]["problem"] + SUFFIX}],
                                         tokenize=True, add_generation_prompt=True, enable_thinking=True)["input_ids"]
        seed = int.from_bytes(hashlib.sha256(f"force:{r['id']}".encode()).digest()[:4], "little")
        row = Row(version=version, prompt=list(prompt) + ids + close, seed=seed, sampling=Sampling(),
                  max_new=min(args.answer_max, BUDGET - len(ids) - len(close)), loop_stop=False)
        row.penalize, row.meta = ids, r
        todo.append(row)
    todo.sort(key=lambda row: len(row.prompt))
    i = 0
    while i < len(todo):
        size = max(1, min(16, 96_000 // len(todo[min(len(todo) - 1, i + 15)].prompt)))
        batch, started = todo[i:i + size], time.time()
        engine.generate(batch, chunk=64, prefill_chunk=1024)
        for row in batch:
            answer = engine.tok.decode(row.tokens)
            pred, ok = grade(answer, problems[row.meta["id"]]["answer"])
            row.meta.update(answer_tokens=len(row.tokens), finish=row.finish, pred=pred, correct=bool(ok),
                            answer_text=answer[:1500])
        i += len(batch)
        print(f"  {i}/{len(todo)} answers, {time.time() - started:.0f} s", flush=True)

    for r in rows:
        r.pop("thinking_ids")
        r.setdefault("correct", r["original_correct"])
    out = args.out or REPO / f"results/stop_and_answer_{args.model}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    forced = [r for r in rows if r["rule"] != "none"]
    print(json.dumps({"problems": len(rows), "original_correct": sum(r["original_correct"] for r in rows),
                      "with_rule_correct": sum(r["correct"] for r in rows), "rule_applied": len(forced),
                      "forced_with_boxed_answer": sum(r.get("pred") is not None for r in forced),
                      "broken_by_rule": sum(r["original_correct"] and not r["correct"] for r in rows)}))


if __name__ == "__main__":
    main()
