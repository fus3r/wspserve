"""wspserve: a shared forward-pass service for weight-space programming sessions.

The request format is the Sparks of WSP Experiment 8 forward-pass tool ("generate" with
model original | edited | both, or "teacher_forced"), so an editor written for that harness
can use it. What differs is the execution:

- one resident model, each session's edits are compiled to typed deltas and applied per row,
- a paired pass runs both halves in one batch, generation requests waiting at the same time
  from different sessions are merged into the same batch,
- probes (teacher-forced passes) run between decode chunks instead of queueing behind a
  generation, base-model samples are memoized (seeds are fixed per problem and sample, so a
  memoized sample is the one the service would draw again), rows caught looping stop early
  and are reported separately from those cut at max_tokens.

Run: python -m wspserve.server --logs <exp8_math> --port 8765
"""
import argparse
import hashlib
import json
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .engine import Engine, Row

ROOT = Path(__file__).resolve().parents[1]
PRACTICE = Path(os.environ.get("WSP_PRACTICE",   # exp8_math/practice of RikiyaT/lm_understand-logs
                               ROOT / "results/practice"))
SESSIONS = ROOT / "results/sessions"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 0, i + len("\\boxed{")
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            if depth == 0:
                return text[j:k]
            depth -= 1
    return None


def grade(text, answer):
    from math_verify import parse, verify

    pred = last_boxed(text)
    if pred is None:
        return None, False
    try:
        return pred, bool(verify(parse(f"${answer}$"), parse(f"${pred}$")))
    except Exception:
        return pred, pred.strip() == str(answer).strip()


@dataclass
class Job:
    session: "Session"
    request: dict
    done: threading.Event = field(default_factory=threading.Event)
    result: dict = None
    submitted: float = field(default_factory=time.time)


@dataclass
class Session:
    sid: str
    path: Path
    edits: dict = field(default_factory=dict)      # tensor name -> float32 array
    version: str = "base"
    passes: int = 0


@dataclass
class _Generation:
    job: Job
    remaining: int
    halves: dict = field(default_factory=lambda: {"original": [], "edited": []})
    rows: list = field(default_factory=list)
    memo_hits: int = 0
    started: float = None
    joined: bool = False


class Service:
    def __init__(self, chunk=32, loop_stop=True, memo=True, gather_s=3.0, active_s=1800.0, *,
                 engine=None, join=True, max_rows=128):
        """gather_s: before a generation batch starts, wait up to this long for sessions that
        were active in the last active_s seconds but have nothing queued, so that sessions
        alternating with each other end up in one batch instead of taking turns."""
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        self.engine = engine if engine is not None else Engine()
        self.chunk, self.loop_stop, self.use_memo = chunk, loop_stop, memo
        self.gather_s, self.active_s = gather_s, active_s
        self.join, self.max_rows = join, max_rows
        self.last_seen = {}
        self.problems = {}
        for name in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
            for line in (PRACTICE / name).read_text().split("\n"):   # texts contain U+2028
                if not line.strip():
                    continue
                r = json.loads(line)
                self.problems[r["id"]] = r
        self.sessions, self.memo = {}, {}
        self.probes, self.gens = queue.Queue(), queue.Queue()
        self.wakeup = threading.Event()
        self.lock = threading.Lock()
        self.log = []
        self._batch_id = 0
        self._generating = False
        self.log_path = SESSIONS / "service-log.jsonl"
        threading.Thread(target=self._serve, daemon=True).start()

    def _dump_log(self):
        SESSIONS.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("".join(json.dumps(e) + "\n" for e in self.log))

    # ------------------------------------------------------------------ sessions and edits

    def session(self, sid):
        with self.lock:
            if sid not in self.sessions:
                path = SESSIONS / sid
                path.mkdir(parents=True, exist_ok=True)
                self.sessions[sid] = Session(sid, path)
            return self.sessions[sid]

    def edit(self, sid, op, name=None, npy=None):
        s = self.session(sid)
        with self.lock:
            if op == "save":
                s.edits[name] = np.load(npy).astype(np.float32)
            elif op == "drop":
                s.edits.pop(name, None)
            elif op == "clear":
                s.edits.clear()
            s.version = self.engine.register_float32(s.edits)
            deltas = self.engine.versions[s.version]
            return {"version": s.version, "edited": sorted(deltas),
                    "forms": {n: d.describe() for n, d in deltas.items()}}

    # ------------------------------------------------------------------ requests

    def forward(self, sid, request):
        self.last_seen[sid] = time.time()
        job = Job(self.session(sid), request)
        (self.probes if request.get("type") == "teacher_forced" else self.gens).put(job)
        self.wakeup.set()
        job.done.wait()
        return job.result

    def _serve(self):
        while True:
            self.wakeup.wait(timeout=0.5)
            self.wakeup.clear()
            self._run_probes()
            jobs = []
            while True:
                try:
                    jobs.append(self.gens.get_nowait())
                except queue.Empty:
                    break
            if jobs and self.gather_s > 0:
                now = time.time()
                waiting = {sid for sid, t in self.last_seen.items() if now - t < self.active_s}
                deadline = now + self.gather_s
                while time.time() < deadline and waiting - {j.session.sid for j in jobs}:
                    self._run_probes()
                    try:
                        jobs.append(self.gens.get(timeout=0.1))
                    except queue.Empty:
                        pass
            if jobs:
                try:
                    self._run_generations(jobs)
                except Exception as exc:  # report the failure to every waiting session
                    for job in jobs:
                        if not job.done.is_set():
                            job.result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                            job.done.set()

    def _run_probes(self):
        while True:
            try:
                job = self.probes.get_nowait()
            except queue.Empty:
                return
            started = time.time()
            try:
                job.result = self._teacher_forced(job)
            except Exception as exc:
                job.result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            job.result["s"] = round(time.time() - started, 2)
            job.result["wait_s"] = round(started - job.submitted, 2)
            self.log.append({"kind": "probe", "session": job.session.sid, "wait_s": job.result["wait_s"],
                             "s": job.result["s"], "t0": round(started, 2), "t1": round(time.time(), 2),
                             "during_generation": self._generating})
            job.done.set()
            self._dump_log()

    def _prompt_ids(self, text_or_problem):
        tok = self.engine.tok
        return tok.apply_chat_template([{"role": "user", "content": text_or_problem + SUFFIX}], tokenize=True,
                                       add_generation_prompt=True, enable_thinking=True)["input_ids"]

    def _teacher_forced(self, job):
        req, s = job.request, job.session
        tok = self.engine.tok
        if req.get("chats"):
            texts = [tok.apply_chat_template(c, tokenize=False, enable_thinking=bool(req.get("enable_thinking")))
                     for c in req["chats"]]
        else:
            texts = req["texts"]
        ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        version = s.version if req.get("model") == "edited" else "base"
        ret = req.get("return", {})
        results = self.engine.teacher_forced([(version, i) for i in ids], hidden_layers=ret.get("hidden_states", []),
                                             mlp_layers=ret.get("mlp_act", []), topk=int(ret.get("topk", 0)))
        s.passes += 1
        out = s.path / f"fwd_{s.passes}"
        out.mkdir(exist_ok=True)
        width = max(map(len, ids))
        files = {}
        for key in results[0]:
            first = results[0][key]
            shape = (len(ids), width - (1 if key == "token_logprobs" else 0), *first.shape[1:])
            arr = np.zeros(shape, dtype=np.float16 if key.startswith(("hidden", "mlp")) else np.float32)
            for r, res in enumerate(results):
                arr[r, :len(res[key])] = res[key]
            np.save(out / f"{key}.npy", arr)
            files[key] = list(arr.shape)
        (out / "meta.json").write_text(json.dumps({"lengths": [len(i) for i in ids], "version": version}))
        mean_lp = [float(np.mean(r["token_logprobs"])) if len(r["token_logprobs"]) else 0.0 for r in results]
        return {"ok": True, "type": "teacher_forced", "dir": str(out), "files": files,
                "lengths": [len(i) for i in ids], "mean_token_logprob": mean_lp,
                "edited_tensors": len(self.engine.versions[version])}

    def _run_generations(self, jobs):
        # Rows waiting for a free slot stay here, outside the engine's active batch.
        ready, running, owners, pending = deque(), {}, {}, {}
        self._batch_id += 1
        batch = self._batch_id

        def admitted(state, joined):
            if state.started is None:
                state.started, state.joined = time.time(), joined

        def complete(state):
            if state.remaining or state.job.done.is_set():
                return
            job = state.job
            finished = time.time()
            elapsed = finished - state.started
            job.result = self._summarize(job, state.halves, elapsed, state.memo_hits)
            self.log.append({"kind": "generate", "sessions": [job.session.sid], "rows": len(state.rows),
                             "memo_hits": state.memo_hits, "s": round(elapsed, 1),
                             "waits_s": [round(state.started - job.submitted, 1)],
                             "t0": round(state.started, 2), "t1": round(finished, 2),
                             "versions": sorted({r.version for r in state.rows}),
                             "tokens": sum(len(r.tokens) for r in state.rows),
                             "finish": {k: sum(r.finish == k for r in state.rows)
                                        for k in ("stop", "length", "loop")},
                             "joined": state.joined, "batch": batch})
            self._dump_log()
            job.done.set()

        def prepare(job, joined):
            req, s = job.request, job.session
            models = {"original": ["base"], "edited": [s.version], "both": ["base", s.version]}[req.get("model", "both")]
            items = req.get("problems", [])
            n = int(req.get("n", 1))
            max_new = int(req.get("max_tokens", 4096))
            state = _Generation(job, len(models) * len(items) * n)
            for model_name, version in zip(["original", "edited"] if len(models) == 2 else [req.get("model")], models):
                for pid in items:
                    problem = self.problems[pid]
                    for k in range(n):
                        seed = int.from_bytes(hashlib.sha256(f"{pid}:{k}".encode()).digest()[:4], "little")
                        key = (pid, k, max_new, self.loop_stop)
                        if version == "base" and self.use_memo and key in self.memo:
                            admitted(state, joined)
                            state.halves[model_name or "original"].append((pid, k, self.memo[key]))
                            state.memo_hits += 1
                            state.remaining -= 1
                            continue
                        if version == "base" and self.use_memo and key in pending:
                            row = pending[key]
                            owners[id(row)].append((state, model_name, pid, k, key))
                            state.memo_hits += 1
                            if id(row) in running:
                                admitted(state, joined)
                            continue
                        row = Row(version=version, prompt=self._prompt_ids(problem["problem"]), seed=seed,
                                  max_new=max_new, loop_stop=self.loop_stop)
                        if version == "base" and self.use_memo:
                            pending[key] = row
                        ready.append(row)
                        state.rows.append(row)
                        owners[id(row)] = [(state, model_name, pid, k, key)]
            if state.remaining == 0:
                admitted(state, joined)
                complete(state)

        def take_rows(joined):
            rows = []
            while ready and len(running) < self.max_rows:
                row = ready.popleft()
                running[id(row)] = row
                rows.append(row)
                for state, *_ in owners[id(row)]:
                    admitted(state, joined)
            return rows

        def finish(row):
            running.pop(id(row))
            listeners = owners.pop(id(row))
            record = {"tokens": len(row.tokens), "finish": row.finish, "text": self.engine.tok.decode(row.tokens)}
            if row.version == "base" and self.use_memo:
                key = listeners[0][-1]
                self.memo[key] = record
                del pending[key]
            for state, model_name, pid, k, _ in listeners:
                state.halves[model_name or "original"].append((pid, k, record))
                state.remaining -= 1
                complete(state)

        def admit():
            joined = bool(running)
            if self.join and joined:
                while True:
                    try:
                        job = self.gens.get_nowait()
                    except queue.Empty:
                        break
                    jobs.append(job)  # also report engine failures to jobs admitted later
                    prepare(job, joined=True)
            return take_rows(joined)

        for job in jobs:
            prepare(job, joined=False)
        self._generating = True
        try:
            rows = take_rows(joined=False)
            while rows:
                self.engine.generate(rows, chunk=self.chunk, between_chunks=self._run_probes,
                                     admit=admit if self.join or ready else None,
                                     on_finish=finish, max_rows=self.max_rows)
                # A prefill whose rows all finish immediately can leave more ready rows.
                rows = take_rows(joined=False)
        finally:
            self._generating = False

    def _summarize(self, job, halves, elapsed, memo_hits):
        s, req = job.session, job.request
        s.passes += 1
        out = {"ok": True, "type": "generate", "model": req.get("model", "both"), "s": round(elapsed, 1),
               "memo_hits": memo_hits}
        graded = {}
        for half, records in halves.items():
            if not records:
                continue
            path = s.path / (f"gen_{s.passes}_{half}.jsonl" if req.get("model") == "both" else f"gen_{s.passes}.jsonl")
            rows = []
            for pid, k, rec in sorted(records, key=lambda x: (x[0], x[1])):
                problem = self.problems[pid]
                pred, correct = grade(rec["text"], problem["answer"])
                rows.append({"id": pid, "sample": k, "source": problem.get("source"), "pred": pred,
                             "correct": correct, "tokens": rec["tokens"], "finish": rec["finish"], "text": rec["text"]})
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            n = len(rows)
            graded[half] = rows
            out[half] = {"file": str(path), "generations": n, "accuracy": sum(r["correct"] for r in rows) / n,
                         "mean_tokens": sum(r["tokens"] for r in rows) / n,
                         "cut_share": sum(r["finish"] in ("length", "loop") for r in rows) / n,
                         "failures": {"wrong_answer": sum(r["finish"] == "stop" and r["pred"] is not None and not r["correct"] for r in rows),
                                      "no_answer_finished": sum(r["finish"] == "stop" and r["pred"] is None for r in rows),
                                      "cut_at_max_tokens": sum(r["finish"] == "length" for r in rows),
                                      "stopped_looping": sum(r["finish"] == "loop" for r in rows)}}
        if len(graded) == 2:
            a = {(r["id"], r["sample"]): r["correct"] for r in graded["original"]}
            b = {(r["id"], r["sample"]): r["correct"] for r in graded["edited"]}
            keys = sorted(a)
            by_item = {}
            for key in keys:
                by_item.setdefault(key[0], []).append(b[key] - a[key])
            diffs = np.array([np.mean(v) for v in by_item.values()])
            out["paired"] = {"difference": float(diffs.mean()),
                             "se": float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else None,
                             "wrong_to_right": sum(b[k] and not a[k] for k in keys),
                             "right_to_wrong": sum(a[k] and not b[k] for k in keys)}
        return out


class Handler(BaseHTTPRequestHandler):
    service: Service = None

    def log_message(self, *args):
        pass

    def _send(self, payload, code=200):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status":
            svc = self.service
            self._send({"stats": svc.engine.stats, "versions": len(svc.engine.versions), "memo": len(svc.memo),
                        "sessions": {k: {"version": v.version, "passes": v.passes} for k, v in svc.sessions.items()},
                        "log": svc.log[-50:]})
        else:
            self._send({"error": "unknown path"}, 404)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            if self.path == "/edit":
                self._send(self.service.edit(data["session"], data["op"], data.get("name"), data.get("npy")))
            elif self.path == "/forward":
                self._send(self.service.forward(data["session"], data["request"]))
            else:
                self._send({"error": "unknown path"}, 404)
        except Exception as exc:
            self._send({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)


def main():
    global PRACTICE, SESSIONS
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, help="exp8_math folder, or set WSP_PRACTICE")
    parser.add_argument("--out", type=Path, default=SESSIONS)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--no-loop-stop", action="store_true")
    parser.add_argument("--no-memo", action="store_true")
    parser.add_argument("--no-join", action="store_true")
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--gather", type=float, default=3.0)
    args = parser.parse_args()
    if args.logs is not None:
        PRACTICE = args.logs / "practice"
    SESSIONS = args.out
    Handler.service = Service(chunk=args.chunk, loop_stop=not args.no_loop_stop, memo=not args.no_memo,
                              gather_s=args.gather, join=not args.no_join, max_rows=args.max_rows)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"wspserve listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
