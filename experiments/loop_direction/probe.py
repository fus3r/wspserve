"""Probe the selected traces through a running service, then save compact readouts."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.loop_direction.prepare import OUT, checkpoint, dump
import numpy as np

import time
import urllib.request
from datetime import datetime, timezone
from wspdelta import bf16
from experiments.compile_released_edits import rebuild_from_bundle

import wspserve.client as W


def status():
    with urllib.request.urlopen(W.SERVER+"/status",timeout=10) as r:
        return json.loads(r.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    results, logs = args.out, args.logs
    W.WORK = results / "work"
    probes=json.loads((results/"probes.json").read_text())
    focused=json.loads((results/"focused_probes.json").read_text())
    assert len(probes)<=32
    assert all(len(p["input_ids"])<=3000 for p in probes)
    ledger_path=results/"request_ledger.json"
    if ledger_path.exists():
        raise RuntimeError("A request ledger already exists; inspect it before issuing any additional requests.")
    st=checkpoint()
    W._st=st  # Use the verified local cache, save_edit need not access the network.
    before=status()
    dump(results/"server_before.json",before)
    assert W.SESSION not in before["sessions"], "Do not overwrite an existing editor session."
    ledger=[]
    edits=[]
    try:
        for variant in ("original","p50c","p25c"):
            for name in list(edits):
                response=W.drop_edit(name)
                dump(results/f"drop_{variant}_{name}.json",response)
                edits.remove(name)
            if variant!="original":
                manifest=json.loads((logs/f"runs/{variant}/edit/manifest.json").read_text())
                with np.load(logs/f"runs/{variant}/edit/edit.npz",allow_pickle=False) as bundle:
                    for i,t in enumerate(manifest["tensors"]):
                        base=st.bf16(t["name"])
                        assert bf16.sha256(base)==t["base_sha256"]
                        edited=rebuild_from_bundle(base,t,bundle,i)
                        assert bf16.sha256(edited)==t["edited_sha256"]
                        response=W.save_edit(t["name"],bf16.to_f32(edited))
                        edits.append(t["name"])
                        dump(results/f"register_{variant}_{i}.json",{"base_hash":t["base_sha256"],"edited_hash":t["edited_sha256"],"response":response})
            panels=[("prefix",probes),("focused",focused)]
            batches=[(panel,first,items[first:first+16]) for panel,items in panels for first in range(0,len(items),16)]
            for panel,first,batch in batches:
                request={"type":"teacher_forced","model":"original" if variant=="original" else "edited",
                         "texts":[p["text"] for p in batch],
                         "return":{"token_logprobs":True,"topk":20,"hidden_states":[15,16,19,20,23]}}
                label=f"{variant}_{panel}_{first//16}"
                dump(results/f"request_{label}.json",request)
                entry={"label":label,"variant":variant,"panel":panel,"indices":[p["index"] for p in batch],
                       "texts":len(batch),"max_tokens":max(len(p["input_ids"]) for p in batch),
                       "start_utc":datetime.now(timezone.utc).isoformat(),"state":"submitted"}
                ledger.append(entry)
                assert len(ledger)<=12
                dump(ledger_path,ledger)
                print("SUBMIT",label,len(batch),"texts",sum(len(p["input_ids"]) for p in batch),"tokens",flush=True)
                started=time.monotonic()
                response=W.forward(request)
                dump(results/f"response_{label}.json",response)
                entry.update({"response":response,"seconds":time.monotonic()-started,"state":"done"})
                dump(ledger_path,ledger)
                assert response.get("ok"),response
                assert response["lengths"]==[len(p["input_ids"]) for p in batch]
                # The service owns its output directory. Keep an explicit inventory here,
                # later numeric summaries and compact projections are written in our folder.
                entry["server_output_dir"]=response["dir"]
                dump(ledger_path,ledger)
                print("DONE",label,round(entry["seconds"],2),"seconds",response["dir"],flush=True)
    finally:
        for name in list(edits):
            response=W.drop_edit(name)
            dump(results/f"cleanup_{name}.json",response)
        dump(results/"server_after.json",status())
    compact(results)
    print("PROBES COMPLETE", len(ledger), flush=True)


LAYERS=[15,16,19,20,23]
DIRECTIONS=["p50c_L15_mlp","p50c_L19_mlp"]


def compact(results):
    ledger=json.loads((results/"request_ledger.json").read_text())
    assert len(ledger)==12 and all(r["state"]=="done" for r in ledger)
    assert max(r["texts"] for r in ledger)<=16 and max(r["max_tokens"] for r in ledger)<=3000
    with np.load(results/"directions.npz",allow_pickle=False) as z:
        names=z["names"].tolist()
        U=z["vectors"][[names.index(k) for k in DIRECTIONS]].astype(np.float32)
    for panel in ("prefix","focused"):
        probes=json.loads((results/("probes.json" if panel=="prefix" else "focused_probes.json")).read_text())
        width=max(len(p["input_ids"]) for p in probes)
        for variant in ("original","p50c","p25c"):
            destination=results/f"compact_{variant}_{panel}.npz"
            if destination.exists():
                print("already compacted",destination.name,flush=True)
                continue
            items=[r for r in ledger if r["variant"]==variant and r["panel"]==panel]
            arrays={"lengths":np.asarray([len(p["input_ids"]) for p in probes]),
                    "layers":np.asarray(LAYERS),"directions":np.asarray(DIRECTIONS),
                    "token_logprobs":np.zeros((len(probes),width-1),np.float32),
                    "topk_ids":np.zeros((len(probes),width,20),np.int32),
                    "topk_logprobs":np.zeros((len(probes),width,20),np.float32),
                    "projection":np.zeros((len(probes),len(LAYERS),2,width),np.float32),
                    "cosine":np.zeros((len(probes),len(LAYERS),2,width),np.float32),
                    "residual_norm":np.zeros((len(probes),len(LAYERS),width),np.float32)}
            hashes={}
            for item in items:
                source=Path(item["response"]["dir"])
                for key in ("token_logprobs","topk_ids","topk_logprobs"):
                    arr=np.load(source/f"{key}.npy",mmap_mode="r",allow_pickle=False)
                    for row,index in enumerate(item["indices"]):
                        n=len(probes[index]["input_ids"])-(key=="token_logprobs")
                        arrays[key][index,:n]=arr[row,:n]
                for li,layer in enumerate(LAYERS):
                    arr=np.load(source/f"hidden_{layer}.npy",mmap_mode="r",allow_pickle=False)
                    for row,index in enumerate(item["indices"]):
                        n=len(probes[index]["input_ids"])
                        h=np.asarray(arr[row,:n],np.float32)
                        assert np.isfinite(h).all()
                        dot=h@U.T
                        norm=np.linalg.norm(h,axis=1)
                        arrays["projection"][index,li,:,:n]=dot.T
                        arrays["cosine"][index,li,:,:n]=(dot/np.maximum(norm,1e-30)[:,None]).T
                        arrays["residual_norm"][index,li,:n]=norm
                        if layer==15:
                            hashes[str(index)]=hashlib.sha256(np.asarray(arr[row,:n]).tobytes()).hexdigest()
                print("reduced",item["label"],flush=True)
            arrays["pre_layer15_hashes"]=np.asarray([hashes[str(i)] for i in range(len(probes))])
            np.savez_compressed(destination,**arrays)
    print("REDUCTION COMPLETE",flush=True)


if __name__=="__main__":main()
