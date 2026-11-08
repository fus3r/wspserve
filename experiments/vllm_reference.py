"""Reference numbers from vLLM on the same GPU.

1. Engine start time, each in a fresh process (the first one with an empty compile cache):
   what the Experiment 8 harness paid for every generation pass.
2. Generation throughput of a resident engine on a pass shaped like the replay's: the given
   number of practice problems, the evaluation's chat template and sampling, --max-tokens each.

    python experiments/vllm_reference.py --practice <logs>/exp8_math/practice --prompts 64

Writes results/vllm_reference.json.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QWEN, REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
ENGINE = dict(model=QWEN, revision=REVISION, dtype="bfloat16", gpu_memory_utilization=0.85, seed=0,
              limit_mm_per_prompt={"image": 0, "video": 0})


def practice_prompts(practice, count):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN, revision=REVISION)
    problems = []
    for name in ("gsm8k_train.jsonl", "math_train.jsonl"):
        lines = [l for l in (practice / name).read_text().split("\n") if l.strip()]
        problems += [json.loads(l) for l in lines[: count // 2]]
    prompts = [tok.apply_chat_template([{"role": "user", "content": p["problem"] + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) for p in problems]
    return problems, prompts


def start_only(max_len):
    started = time.time()
    from vllm import LLM

    LLM(max_model_len=max_len, **ENGINE)
    print(json.dumps({"start_s": round(time.time() - started, 1)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--practice", type=Path)
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--start-only", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO / "results/vllm_reference.json")
    args = parser.parse_args()
    max_len = args.max_tokens + 1024
    if args.start_only:
        return start_only(max_len)

    import vllm

    report = {"vllm": vllm.__version__, "prompts": args.prompts, "max_tokens": args.max_tokens, "starts_s": []}
    for attempt in range(args.starts):
        t = time.time()
        out = subprocess.run([sys.executable, __file__, "--start-only", "--max-tokens", str(args.max_tokens)],
                             capture_output=True, text=True, check=True).stdout
        inner = json.loads([l for l in out.splitlines() if l.startswith("{")][-1])["start_s"]
        report["starts_s"].append({"engine_s": inner, "process_s": round(time.time() - t, 1)})
        print("engine start", attempt, report["starts_s"][-1], flush=True)

    from vllm import LLM, SamplingParams

    problems, prompts = practice_prompts(args.practice, args.prompts)
    llm = LLM(max_model_len=max_len, **ENGINE)
    params = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5,
                             max_tokens=args.max_tokens, seed=i) for i in range(len(prompts))]
    started = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - started
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    report.update(generate_s=round(elapsed, 1), generated_tokens=tokens, tokens_per_s=round(tokens / elapsed, 1),
                  hit_cap=sum(o.outputs[0].finish_reason == "length" for o in outputs))
    print(json.dumps({k: report[k] for k in ("starts_s", "generate_s", "generated_tokens", "tokens_per_s")}))
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
