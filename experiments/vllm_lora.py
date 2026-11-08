"""The released p50c edit served by vLLM as a LoRA adapter, in one batch with the base model.

The adapter is the rank-one factorization that compile_released_edits.py recovers (two
down_proj matrices, 37 KB). vLLM applies it unmerged in BF16, so edited rows are close to, not
bit-identical with, the evaluated checkpoint. Accuracy on the same problems is reported for
both halves as a check that the adapter carries the edit.

    python experiments/compile_released_edits.py --runs p50c --logs <logs>/exp8_math
    python experiments/vllm_lora.py --practice <logs>/exp8_math/practice --adapter results/adapters/p50c

Writes results/vllm_lora.json. With --max-tokens 32768 --passes paired edited it runs a pass of
the size the released sessions ran (128 rows at the evaluation's token limit).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.vllm_reference import ENGINE, practice_prompts  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--practice", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--passes", nargs="+", default=["base", "edited", "paired"],
                        choices=["base", "edited", "paired"])
    parser.add_argument("--out", type=Path, default=REPO / "results/vllm_lora.json")
    args = parser.parse_args()

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from wspserve.server import grade

    problems, prompts = practice_prompts(args.practice, args.prompts)
    started = time.time()
    llm = LLM(max_model_len=args.max_tokens + 1024, enable_lora=True, max_lora_rank=8, max_loras=4, **ENGINE)
    report = {"vllm": vllm.__version__, "prompts": len(prompts), "max_tokens": args.max_tokens,
              "start_s": round(time.time() - started, 1)}

    # A new edit is a new adapter id: time its first request against a plain one-token request.
    one = SamplingParams(max_tokens=1, temperature=0)
    t = time.time()
    llm.generate(prompts[:1], one, use_tqdm=False)
    report["one_token_base_s"] = round(time.time() - t, 3)
    loads = []
    for k in range(2, 7):
        t = time.time()
        llm.generate(prompts[:1], one, lora_request=LoRARequest(f"edit-{k}", k, str(args.adapter)), use_tqdm=False)
        loads.append(time.time() - t)
    report["one_token_new_adapter_s"] = [round(x, 3) for x in loads]

    edit = LoRARequest("p50c", 1, str(args.adapter))

    def sampling(seed):
        return SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5,
                              max_tokens=args.max_tokens, seed=seed)

    def run(halves):
        batch = [(i, half) for half in halves for i in range(len(prompts))]
        t = time.time()
        outputs = llm.generate([prompts[i] for i, _ in batch], [sampling(i) for i, _ in batch],
                               lora_request=[edit if half == "edited" else None for _, half in batch])
        elapsed = time.time() - t
        result = {"s": round(elapsed, 1), "rows": len(batch)}
        texts = {}
        for (i, half), o in zip(batch, outputs):
            texts.setdefault(half, []).append((o.outputs[0].text, len(o.outputs[0].token_ids),
                                               o.outputs[0].finish_reason, grade(o.outputs[0].text, problems[i]["answer"])[1]))
        tokens = sum(n for rows in texts.values() for _, n, _, _ in rows)
        result.update(generated_tokens=tokens, tokens_per_s=round(tokens / elapsed, 1))
        for half, rows in texts.items():
            result[half] = {"correct": sum(c for *_, c in rows), "hit_cap": sum(f == "length" for _, _, f, _ in rows),
                            "median_tokens": statistics.median(n for _, n, _, _ in rows)}
        if len(texts) == 2:
            result["rows_that_differ"] = sum(a[0] != b[0] for a, b in zip(texts["base"], texts["edited"]))
        print(json.dumps(result), flush=True)
        return result

    report["passes"] = {}
    for name in args.passes:
        report["passes"][name] = run(["base", "edited"] if name == "paired" else [name])
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
