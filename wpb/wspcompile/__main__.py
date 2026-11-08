from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import PolicyError, compile_program, edits_block, load
from .bench import protocol, score


def main():
    parser = argparse.ArgumentParser(description="Compile exact integer policies to WPB edits")
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("policy", type=Path)
    compile_parser.add_argument("-o", "--output", required=True, type=Path)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("policy", type=Path)
    score_parser.add_argument("--game", required=True, choices=tuple(protocol().CAPS))
    score_parser.add_argument("--episodes", type=int, default=1000)
    score_parser.add_argument("--offset", type=int, default=0)
    score_parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    try:
        program = load(args.policy)
        weights = compile_program(program)
        if args.command == "compile":
            args.output.write_text(edits_block(weights), encoding="utf-8")
            print(f"{args.output}: {len(program.atoms)} threshold atoms, {len(program.expanded)} expanded rules")
        else:
            result = score(args.game, weights, args.episodes, args.offset, args.workers)
            print(json.dumps({"policy": str(args.policy), **{k: v for k, v in result.items() if k not in ("scores", "lengths")}}, indent=2))
    except (PolicyError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
