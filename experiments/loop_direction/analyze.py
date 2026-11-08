"""Paired readouts with whole-trace bootstrap intervals."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.loop_direction.prepare import OUT, bootstrap, dump, tokenizer
import numpy as np

import re
import csv

BINS=[(-2048,-1536),(-1536,-1024),(-1024,-768),(-768,-512),(-512,-256),(-256,-128),(-128,0),(0,128),(128,256),(256,512)]


def masks_for(p,tok):
    ids=np.asarray(p["input_ids"])
    n=len(ids)-1
    # Array position i predicts target i+1. A detector flags after consuming N tokens,
    # its zero-aligned prediction is therefore made at input index event_position-1.
    target=np.arange(1,n+1)
    generated=target>=p["generated_start"]
    close_ids=tok("</think>",add_special_tokens=False)["input_ids"]
    assert len(close_ids)==1
    close=close_ids[0]
    close_pos=np.flatnonzero(ids==close)
    thought=generated & (target<=(close_pos[0] if len(close_pos) else len(ids)))
    input_pieces=[p["text"][a:b] for a,b in p["offsets"][:-1]]
    target_pieces=[p["text"][a:b] for a,b in p["offsets"][1:]]
    line_end=np.asarray(["\n" in s for s in input_pieces])
    target_newline=np.asarray(["\n" in s for s in target_pieces])
    recheck=np.asarray([bool(re.fullmatch(r"\s*(?:wait|however|alternatively|again|check|recheck)",s,re.I)) for s in target_pieces])
    wait=np.asarray([s.strip().lower()=="wait" for s in target_pieces])
    digits=np.asarray([bool(re.fullmatch(r"\s*\d+(?:\.\d+)?",s)) for s in target_pieces])
    indent=np.asarray([len(s)>=2 and s.isspace() and "\n" not in s for s in target_pieces])
    result={"all_thought":thought,"early512":thought & (target<p["generated_start"]+512),
            "line_end_decisions":thought & line_end,"newline_targets":thought & target_newline,
            "recheck_targets":thought & recheck,"wait_targets":thought & wait,
            "numeric_targets":thought & digits,"indent_targets":thought & indent,
            "end_thought_targets":thought & (ids[1:]==close)}
    if p["group"]=="loop":
        if "reference_event_position" in p:
            event=p["reference_event_position"]
        elif p["first_loop_flag_visible"] is not None:
            event=p["generated_start"]+p["first_loop_flag_visible"]
        else:
            event=None
        if event is not None:
            relative=target-event
            result.update({"pre_flag_all":thought & (relative<0),"post_flag_all":thought & (relative>=0),
                           "pre_flag_512":thought & (relative>=-512) & (relative<0),
                           "post_flag_512":thought & (relative>=0) & (relative<512)})
    return result,close


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--probes", type=Path, default=OUT, help="prepared inputs and compact readouts")
    args = parser.parse_args()
    results = args.probes
    args.out.mkdir(parents=True, exist_ok=True)
    tok=tokenizer()
    lp_rows=[]
    projection_rows=[]
    bounds_rows=[]
    timeseries_rows=[]
    controls=[]
    for panel in ("prefix","focused"):
        probes=json.loads((results/("probes.json" if panel=="prefix" else "focused_probes.json")).read_text())
        data={}
        for variant in ("original","p50c","p25c"):
            with np.load(results/f"compact_{variant}_{panel}.npz",allow_pickle=False) as archive:
                data[variant]={key:archive[key] for key in archive.files}
        base=data["original"]
        layers=base["layers"].tolist()
        directions=base["directions"].tolist()
        for variant in ("p50c","p25c"):
            equal=base["pre_layer15_hashes"]==data[variant]["pre_layer15_hashes"]
            controls.append({"panel":panel,"variant":variant,"sequences":len(equal),"pre_layer15_identical":int(equal.sum()),
                             "max_abs_pre_layer15_projection_difference":float(np.max(np.abs(data[variant]["projection"][:,0]-base["projection"][:,0])))})
        for si,p in enumerate(probes):
            masks,close=masks_for(p,tok)
            n=len(p["input_ids"])-1
            original=base["token_logprobs"][si,:n]
            for variant in ("p50c","p25c"):
                edited=data[variant]["token_logprobs"][si,:n]
                delta=edited-original
                for category,mask in masks.items():
                    if not np.any(mask):continue
                    lp_rows.append({"panel":panel,"group":p["group"],"id":p["id"],"variant":variant,
                                    "category":category,"positions":int(mask.sum()),"original_logprob":float(original[mask].mean()),
                                    "edited_logprob":float(edited[mask].mean()),"delta_logprob":float(delta[mask].mean()),
                                    "absolute_delta_logprob":float(np.abs(delta[mask]).mean())})
                # With only top-20 output, unreturned </think> probabilities are censored.
                for category in ("all_thought","line_end_decisions","pre_flag_512","post_flag_512"):
                    if category not in masks or not np.any(masks[category]):continue
                    mask=masks[category]
                    bounds={}
                    for label,d in [("original",base),(variant,data[variant])]:
                        ids=d["topk_ids"][si,:n]
                        probs=np.exp(d["topk_logprobs"][si,:n].astype(np.float64))
                        present=np.any(ids==close,axis=1)
                        known=np.sum(probs*(ids==close),axis=1)
                        lower=known
                        upper=np.where(present,known,probs[:,-1])
                        bounds[label]=(lower,upper)
                    lo=bounds[variant][0]-bounds["original"][1]
                    hi=bounds[variant][1]-bounds["original"][0]
                    bounds_rows.append({"panel":panel,"group":p["group"],"id":p["id"],"variant":variant,
                                        "category":category,"positions":int(mask.sum()),
                                        "original_probability_lower":float(bounds["original"][0][mask].mean()),
                                        "original_probability_upper":float(bounds["original"][1][mask].mean()),
                                        "edited_probability_lower":float(bounds[variant][0][mask].mean()),
                                        "edited_probability_upper":float(bounds[variant][1][mask].mean()),
                                        "delta_probability_lower":float(lo[mask].mean()),"delta_probability_upper":float(hi[mask].mean())})
            for category in ("all_thought","pre_flag_512","post_flag_512"):
                if category not in masks or not np.any(masks[category]):continue
                mask=masks[category]
                for li,layer in enumerate(layers):
                    for di,direction in enumerate(directions):
                        for quantity in ("projection","cosine"):
                            original=base[quantity][si,li,di,:n]
                            for variant in ("p50c","p25c"):
                                edited=data[variant][quantity][si,li,di,:n]
                                projection_rows.append({"panel":panel,"group":p["group"],"id":p["id"],"variant":variant,
                                                        "category":category,"layer_hidden_index":layer,"direction":direction,"quantity":quantity,
                                                        "positions":int(mask.sum()),"original":float(original[mask].mean()),
                                                        "edited":float(edited[mask].mean()),"delta":float((edited-original)[mask].mean())})
            if panel=="focused":
                event=p["reference_event_position"]
                relative=np.arange(1,n+1)-event
                for a,b in BINS:
                    mask=masks["all_thought"] & (relative>=a) & (relative<b)
                    if not np.any(mask):continue
                    for variant,d in data.items():
                        for li,layer in enumerate(layers):
                            for di,direction in enumerate(directions):
                                timeseries_rows.append({"group":p["group"],"id":p["id"],"variant":variant,
                                                        "bin_start":a,"bin_end":b,"positions":int(mask.sum()),
                                                        "layer_hidden_index":layer,"direction":direction,
                                                        "projection":float(d["projection"][si,li,di,:n][mask].mean()),
                                                        "cosine":float(d["cosine"][si,li,di,:n][mask].mean())})
    with (args.out / "loop_direction_timeseries.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(timeseries_rows[0]))
        writer.writeheader()
        writer.writerows(timeseries_rows)
    lp_summary=[]
    keys=sorted({tuple(r[k] for k in ("panel","group","variant","category")) for r in lp_rows})
    for key in keys:
        rs=[r for r in lp_rows if tuple(r[k] for k in ("panel","group","variant","category"))==key]
        lp_summary.append({**dict(zip(("panel","group","variant","category"),key)),"positions":sum(r["positions"] for r in rs),
                           **{m:bootstrap([r[m] for r in rs]) for m in ("delta_logprob","absolute_delta_logprob","original_logprob","edited_logprob")}})
    contrasts=[]
    for panel in ("prefix","focused"):
        for variant in ("p50c","p25c"):
            for metric in ("delta_logprob","absolute_delta_logprob"):
                pre={r["id"]:r[metric] for r in lp_rows if r["panel"]==panel and r["variant"]==variant and r["category"]=="pre_flag_512"}
                post={r["id"]:r[metric] for r in lp_rows if r["panel"]==panel and r["variant"]==variant and r["category"]=="post_flag_512"}
                ids=sorted(set(pre)&set(post))
                contrasts.append({"panel":panel,"variant":variant,"metric":metric,"contrast":"post minus pre, matched 512-token windows",
                                  **bootstrap([post[i]-pre[i] for i in ids])})
    projection_summary=[]
    names=("panel","group","variant","category","layer_hidden_index","direction","quantity")
    for key in sorted({tuple(r[k] for k in names) for r in projection_rows}):
        rs=[r for r in projection_rows if tuple(r[k] for k in names)==key]
        projection_summary.append({**dict(zip(names,key)),**{m:bootstrap([r[m] for r in rs]) for m in ("original","edited","delta")}})
    baseline_contrasts=[]
    for layer in (15,16,19,20,23):
        for direction in ("p50c_L15_mlp","p50c_L19_mlp"):
            for quantity in ("projection","cosine"):
                pre={r["id"]:r["original"] for r in projection_rows if r["panel"]=="focused" and r["group"]=="loop" and r["variant"]=="p50c" and r["category"]=="pre_flag_512" and r["layer_hidden_index"]==layer and r["direction"]==direction and r["quantity"]==quantity}
                post={r["id"]:r["original"] for r in projection_rows if r["panel"]=="focused" and r["group"]=="loop" and r["variant"]=="p50c" and r["category"]=="post_flag_512" and r["layer_hidden_index"]==layer and r["direction"]==direction and r["quantity"]==quantity}
                baseline_contrasts.append({"layer_hidden_index":layer,"direction":direction,"quantity":quantity,
                                           **bootstrap([post[i]-pre[i] for i in sorted(set(pre)&set(post))])})
    bound_summary=[]
    for key in sorted({tuple(r[k] for k in ("panel","group","variant","category")) for r in bounds_rows}):
        rs=[r for r in bounds_rows if tuple(r[k] for k in ("panel","group","variant","category"))==key]
        metrics=("original_probability_lower","original_probability_upper","edited_probability_lower","edited_probability_upper","delta_probability_lower","delta_probability_upper")
        bound_summary.append({**dict(zip(("panel","group","variant","category"),key)),"traces":len(rs),
                              **{m:bootstrap([r[m] for r in rs]) for m in metrics}})
    result={"uncertainty":"Seeded percentile bootstrap of whole-trace means, 10000 resamples; all intervals exploratory, no multiplicity adjustment.",
            "position_convention":"Logprob i predicts token i+1; hidden state i has consumed i+1 tokens. Hidden k is the residual entering zero-indexed block k, so 16/20 are outputs of edited blocks 15/19; 23 is before block 23, not final-normalized.",
            "metric":"Edited minus original observed-token natural log-probability; not KL, accuracy, or a generated completion rate.",
            "controls":controls,"logprob":lp_summary,"within_trace_post_pre":contrasts,
            "projection":projection_summary,"baseline_post_pre_projection":baseline_contrasts,"thought_close_probability_bounds":bound_summary}
    dump(args.out / "loop_direction_summary.json", result)
    print("CAUSAL ANALYSIS COMPLETE",json.dumps(contrasts),flush=True)


if __name__=="__main__":main()
