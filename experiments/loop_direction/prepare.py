"""Select released traces and prepare the two loop directions."""
import argparse
import os
import sys
import json
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/loop_direction"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
sys.path.insert(0, str(ROOT))

import numpy as np

def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def checkpoint():
    from huggingface_hub import hf_hub_download
    from wspdelta.safetensors_raw import SafeTensors
    path = hf_hub_download("Qwen/Qwen3.5-0.8B", "model.safetensors-00001-of-00001.safetensors",
                           revision=REVISION, local_files_only=True)
    return SafeTensors(path)


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", revision=REVISION,
                                          local_files_only=True, trust_remote_code=False)


def unit(v):
    v = np.asarray(v, np.float64).reshape(-1)
    return v / np.linalg.norm(v)



def bootstrap(values, seed=20261108, repetitions=10000):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean": None, "ci95": [None, None]}
    if len(x) == 1:
        return {"n": 1, "mean": float(x[0]), "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(repetitions, len(x)))].mean(axis=1)
    return {"n": len(x), "mean": float(x.mean()), "ci95": np.quantile(means, [.025, .975]).tolist()}


def ngram_hashes(tokens, n):
    if len(tokens) < n:
        return np.zeros(0, dtype=np.uint64)
    rng = np.random.default_rng(0)
    weights = rng.integers(1, 2**63, size=n, dtype=np.uint64) | np.uint64(1)
    view = np.lib.stride_tricks.sliding_window_view(np.asarray(tokens, dtype=np.uint64), n)
    with np.errstate(over="ignore"):
        return (view * weights).sum(axis=1, dtype=np.uint64)


def first_stop(hashes, n, window, step, threshold):
    """First token position at which the trailing window is mostly repeated n-grams."""
    for end in range(window, len(hashes) + n, step):
        grams = hashes[max(0, end - window):end - n + 1]
        if len(grams) < window // 2:
            continue
        repeated = 1.0 - len(np.unique(grams)) / len(grams)
        if repeated >= threshold:
            return end
    return None


import gzip
import hashlib
import urllib.request
from wspserve.engine import LoopDetector

DATA_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
DATA_URL = f"https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/{DATA_REVISION}/test.jsonl"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."



def directions(results, logs):
    from wspdelta import bf16
    from wspdelta.structure import analyze
    from experiments.compile_released_edits import rebuild_from_bundle

    st = checkpoint()
    manifest = json.loads((logs / "runs/p50c/edit/manifest.json").read_text())
    edits, bases = {}, {}
    with np.load(logs / "runs/p50c/edit/edit.npz", allow_pickle=False) as bundle:
        for i, tensor in enumerate(manifest["tensors"]):
            name = tensor["name"]
            base = st.bf16(name)
            assert bf16.sha256(base) == tensor["base_sha256"]
            edited = rebuild_from_bundle(base, tensor, bundle, i)
            assert bf16.sha256(edited) == tensor["edited_sha256"]
            best, _, _ = analyze(base, edited)
            assert bf16.sha256(best.materialize(base)) == tensor["edited_sha256"]
            edits[name], bases[name] = best, base
    ref_name = list(edits)[1]
    ref = unit(edits[ref_name].b[:, 0])
    cols = edits[ref_name].support
    if (ref @ bf16.to_f64(bases[ref_name][:, cols])).mean() < 0:
        ref = -ref
    names, vectors = [], []
    for name, delta in edits.items():
        u = unit(delta.b[:, 0])
        if u @ ref < 0:
            u = -u
        names.append(f"p50c_L{name.split('.layers.')[1].split('.')[0]}_mlp")
        vectors.append(u)
    np.savez_compressed(results / "directions.npz", names=np.asarray(names), vectors=np.asarray(vectors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    results, logs = args.out, args.logs
    results.mkdir(parents=True, exist_ok=True)
    path = results / "math500_test.jsonl"
    if not path.exists():
        with urllib.request.urlopen(DATA_URL, timeout=45) as r:
            data = r.read(5_000_001)
        assert len(data) < 5_000_000
        path.write_bytes(data)
    data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print("dataset schema",list(data[0]),flush=True)
    # HF uses unique_id, match every available released trace by its exact original id.
    problems={r.get("unique_id",r.get("id")):r for r in data}
    tok=tokenizer()
    rows=[]
    source_paths=sorted((logs/"eval/unedited/parts").glob("*/math500_first_samples.jsonl.gz"))
    for file in source_paths:
        correctness={r["id"]:int(r["correct"][0]) for r in json.loads((file.parent/"math500_per_problem.json").read_text())}
        for line in gzip.open(file,"rt"):
            r=json.loads(line)
            assert r["id"] in problems,r["id"]
            sample=r["text"]
            tokens=tok(sample,add_special_tokens=False)["input_ids"]
            hashes=ngram_hashes(tokens,32)
            flag=first_stop(hashes,n=32,window=1024,step=128,threshold=.75)
            if flag is not None:
                assert LoopDetector().looping(tokens[:flag])
            prompt=tok.apply_chat_template([{"role":"user","content":problems[r["id"]]["problem"]+SUFFIX}],
                                            tokenize=False,add_generation_prompt=True,enable_thinking=True)
            prompt_n=len(tok(prompt,add_special_tokens=False)["input_ids"])
            rows.append({"id":r["id"],"text":sample,"tokens":tokens,"source":"<logs>/" + str(file.relative_to(logs)),"correct_first":correctness[r["id"]],
                         "sample_tokens":len(tokens),"first_loop_flag_generated":flag,"has_think_close":"</think>" in sample,
                         "prompt":prompt,"prompt_tokens":prompt_n,"prefix_flag_visible":flag is not None and flag+prompt_n<2872})
        print("scanned",file.parent.name,"n",len(rows),flush=True)
    loops=sorted([r for r in rows if r["first_loop_flag_generated"] is not None],key=lambda r:(not r["prefix_flag_visible"],r["first_loop_flag_generated"],r["id"]))[:24]
    finishes=sorted([r for r in rows if r["has_think_close"] and r["first_loop_flag_generated"] is None],key=lambda r:(r["sample_tokens"],r["id"]))[:8]
    selected=[]
    for group,group_rows in [("loop",loops),("finish",finishes)]:
        for r in group_rows:
            encoded=tok(r["prompt"]+r["text"],add_special_tokens=False,return_offsets_mapping=True)
            ids=encoded["input_ids"][:3000]
            # A decoded prefix must round-trip exactly, or choose the nearest earlier boundary.
            while True:
                text=tok.decode(ids,skip_special_tokens=False)
                if tok(text,add_special_tokens=False)["input_ids"]==ids:
                    break
                ids=ids[:-1]
            enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
            offsets=enc["offset_mapping"]
            prompt_end=len(r["prompt"])
            gen_start=next((i for i,(_,b) in enumerate(offsets) if b>prompt_end),len(ids))
            generated_ids=ids[gen_start:]
            # Detect against generated-only tokens in precisely the teacher-forced tokenization.
            flag=first_stop(ngram_hashes(generated_ids,32),n=32,window=1024,step=128,threshold=.75)
            selected.append({"index":len(selected),"id":r["id"],"group":group,"text":text,"input_ids":ids,
                             "offsets":offsets,"generated_start":gen_start,"first_loop_flag_visible":flag,
                             "first_loop_flag_generated":r["first_loop_flag_generated"],"sample_tokens":r["sample_tokens"],
                             "correct_first":r["correct_first"],"has_think_close_visible":"</think>" in text[prompt_end:],
                             "source":r["source"],"sample_sha256":hashlib.sha256(r["text"].encode()).hexdigest()})
    # Strict prefixes rarely contain a detector flag. A separate, explicitly
    # context-truncated panel centers the original first flag (or thought closure).
    # It cannot recover the original full-history state, but gives paired readouts
    # around these late events within the same per-text token cap.
    focused=[]
    for group,group_rows in [("loop",loops),("finish",finishes)]:
        for r in group_rows:
            if group=="loop":
                center=r["first_loop_flag_generated"]
            else:
                close_char=r["text"].index("</think>")+len("</think>")
                center=len(tok(r["text"][:close_char],add_special_tokens=False)["input_ids"])
            capacity=3000-r["prompt_tokens"]-4
            end=min(len(r["tokens"]),center+512)
            start=max(0,end-capacity)
            generated=tok.decode(r["tokens"][start:end],skip_special_tokens=False)
            text=r["prompt"]+generated
            enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
            assert len(enc["input_ids"])<=3000
            gen_start=next((i for i,(_,b) in enumerate(enc["offset_mapping"]) if b>len(r["prompt"])),len(enc["input_ids"]))
            gen_ids=enc["input_ids"][gen_start:]
            event_char=len(r["prompt"])+len(tok.decode(r["tokens"][start:center],skip_special_tokens=False))
            event_position=next((i for i,(_,b) in enumerate(enc["offset_mapping"]) if b>event_char),len(enc["input_ids"]))
            focused.append({"index":len(focused),"id":r["id"],"group":group,"text":text,
                            "input_ids":enc["input_ids"],"offsets":enc["offset_mapping"],"generated_start":gen_start,
                            "source_generated_start":start,"source_generated_end":end,
                            "reference_event_generated":center,"reference_event_position":event_position,
                            "window_boundary_ids_exact":gen_ids==r["tokens"][start:end],
                            "first_loop_flag_generated":r["first_loop_flag_generated"],"first_loop_flag_visible":None,
                            "correct_first":r["correct_first"],"has_think_close_visible":"</think>" in generated,
                            "sample_tokens":r["sample_tokens"],"source":r["source"]})
    dump(results / "probes.json", selected)
    dump(results / "focused_probes.json", focused)
    directions(results, logs)
    print(len(selected), "prefix probes,", len(focused), "focused probes", flush=True)


if __name__ == "__main__":
    main()
