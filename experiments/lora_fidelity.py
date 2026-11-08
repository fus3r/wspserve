"""Does the LoRA adapter reproduce the p50c edit? Per-token log-probabilities on the base
model's own reasoning traces, from vLLM (base and adapter) and from wspserve (base and the
exact edit), compared position by position.

    # in the vLLM environment: sample traces, score them with and without the adapter
    python experiments/lora_fidelity.py vllm --practice <logs>/exp8_math/practice --adapter results/adapters/p50c
    # in the wspserve environment: score the same token ids with the exact edit, compare
    python experiments/lora_fidelity.py exact --logs <logs>/exp8_math

Writes results/lora_fidelity_vllm.json, then results/lora_fidelity.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RAW = REPO / "results/lora_fidelity_vllm.json"


def run_vllm(args):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    from experiments.vllm_reference import ENGINE, practice_prompts

    _, prompts = practice_prompts(args.practice, args.traces)
    llm = LLM(max_model_len=4096, enable_lora=True, max_lora_rank=8, max_loras=2, **ENGINE)
    edit = LoRARequest("p50c", 1, str(args.adapter))
    sampled = llm.generate(prompts, [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5,
                                                    max_tokens=args.trace_tokens, seed=i) for i in range(len(prompts))])
    traces = [list(o.prompt_token_ids) + list(o.outputs[0].token_ids) for o in sampled]
    starts = [len(o.prompt_token_ids) for o in sampled]
    score = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    scored = {}
    for name, lora in (("base", None), ("adapter", edit)):
        outs = llm.generate([TokensPrompt(prompt_token_ids=t) for t in traces], score, lora_request=lora)
        scored[name] = [[o.prompt_logprobs[j][t[j]].logprob for j in range(1, len(t))] for o, t in zip(outs, traces)]
    RAW.write_text(json.dumps({"traces": traces, "starts": starts, "vllm": scored}) + "\n")
    print("traces", len(traces), "tokens", sum(len(t) for t in traces))


def run_exact(args):
    from experiments.replay import p50c_float32
    from wspserve.engine import Engine

    raw = json.loads(RAW.read_text())
    engine = Engine()
    vid = engine.register_float32(p50c_float32(engine, args.logs))
    exact = {}
    for name, version in (("base", "base"), ("edit", vid)):
        out = engine.teacher_forced([(version, t) for t in raw["traces"]])
        exact[name] = [list(map(float, o["token_logprobs"])) for o in out]

    def generated(scores):  # positions after the prompt only: the model's own reasoning
        return np.concatenate([np.asarray(s)[start - 1:] for s, start in zip(scores, raw["starts"])])

    vb, va = generated(raw["vllm"]["base"]), generated(raw["vllm"]["adapter"])
    eb, ee = generated(exact["base"]), generated(exact["edit"])
    shift_adapter, shift_exact = va - vb, ee - eb
    report = {"traces": len(raw["traces"]), "scored_tokens": int(len(vb)),
              "edit_shift_mean_abs": float(np.abs(shift_exact).mean()),
              "adapter_shift_mean_abs": float(np.abs(shift_adapter).mean()),
              "shift_correlation": float(np.corrcoef(shift_adapter, shift_exact)[0, 1]),
              "adapter_minus_edit_shift_mean_abs": float(np.abs(shift_adapter - shift_exact).mean()),
              "engines_base_mean_abs": float(np.abs(vb - eb).mean()),
              "large_shifts": int((np.abs(shift_exact) > 0.5).sum()),
              "large_shifts_same_sign": int(((np.abs(shift_exact) > 0.5) & (np.sign(shift_adapter) == np.sign(shift_exact))).sum())}
    (REPO / "results/lora_fidelity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["vllm", "exact"])
    parser.add_argument("--practice", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--traces", type=int, default=16)
    parser.add_argument("--trace-tokens", type=int, default=1536)
    args = parser.parse_args()
    run_vllm(args) if args.mode == "vllm" else run_exact(args)


if __name__ == "__main__":
    main()
