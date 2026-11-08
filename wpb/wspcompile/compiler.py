"""Compile integer predicates and first-match priorities to WPB float32 tensors."""
from __future__ import annotations

import ast
import re

import numpy as np

from .language import Equal, PolicyError, Program, load, parse

SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,),
          "W3": (32, 6), "b3": (6,)}


def compile_program(program: Program):
    w = {name: np.zeros(shape, dtype=np.float32) for name, shape in SHAPES.items()}
    units = {atom: i for i, atom in enumerate(program.atoms)}
    for atom, i in units.items():
        coefficients, k = atom.coefficients, atom.threshold
        # All preactivation summands are exact multiples of 16. This bound
        # preserves their integer lattice under any float32 summation order.
        if 2 * sum(abs(a) for a in coefficients) + abs(1 - 2 * k) > 2**24:
            raise PolicyError("atom exceeds the exact float32 accumulation budget")
        if max((32 * abs(a) for a in coefficients), default=0) > 1_000_000 or abs(16 * (1 - 2*k)) > 1_000_000:
            raise PolicyError("atom needs a weight above 1e6 at the certified saturation gain")
        w["W1"][:, i] = [32 * a for a in coefficients]
        w["b1"][i] = 16 * (1 - 2*k)
    n = len(program.expanded)
    for j, (clause, action) in enumerate(program.expanded):
        # Every literal contributes +-1 (times 16), the unit fires when all of them are +1.
        literals, offset = 0, 0
        for atom, positive in clause:
            if isinstance(atom, Equal):
                (lo, slo), (up, sup) = atom.lower, atom.upper
                slo, sup = (1 if slo else -1), (1 if sup else -1)
                if positive:   # [x >= k] and not [x >= k + 1]: two literals
                    w["W2"][units[lo], j] += 16 * slo
                    w["W2"][units[up], j] -= 16 * sup
                    literals += 2
                else:          # x != k: upper - lower + 1, one literal
                    w["W2"][units[lo], j] -= 16 * slo
                    w["W2"][units[up], j] += 16 * sup
                    offset += 16
                    literals += 1
            else:
                w["W2"][units[atom], j] += 16 if positive else -16
                literals += 1
        w["b2"][j] = 16 * (1 - literals) + offset
        priority = 48 * 2**(n - 1 - j)
        w["W3"][j, action] = priority // 2
        w["b3"][action] += priority // 2
    w["b3"][program.default] += 24
    validate_weights(w)
    return w


def compile_source(source: str):
    return compile_program(parse(source))


def compile_file(path):
    return compile_program(load(path))


def validate_weights(weights):
    if set(weights) != set(SHAPES):
        raise PolicyError("network must have exactly W1, b1, W2, b2, W3, b3")
    for name, shape in SHAPES.items():
        a = weights[name]
        if a.shape != shape or a.dtype != np.float32:
            raise PolicyError(f"{name} must have shape {shape} and dtype float32")
        if not np.all(np.isfinite(a)) or np.max(np.abs(a), initial=0) > 1e6:
            raise PolicyError("weights must be finite and have magnitude <= 1e6")


def logits(weights, x):
    x = np.asarray(x, dtype=np.float32)
    h1 = np.tanh(x @ weights["W1"] + weights["b1"])
    h2 = np.tanh(h1 @ weights["W2"] + weights["b2"])
    return h2 @ weights["W3"] + weights["b3"]


def probabilities(action_logits):
    shifted = np.asarray(action_logits, dtype=np.float64)
    shifted = shifted - shifted.max(axis=-1, keepdims=True)
    p = np.exp(shifted)
    return p / p.sum(axis=-1, keepdims=True)


def edits_block(weights):
    validate_weights(weights)
    lines = []
    for name in SHAPES:
        a = weights[name]
        for index in zip(*np.nonzero(a)):
            target = ", ".join(str(int(i)) for i in index)
            lines.append(f"{name}[{target}] = {float(a[index])!r}")
    return "```edits\n" + "\n".join(lines) + "\n```\n"


def read_edits(answer):
    """Independent AST round-trip check for the scalar assignments we emit.

    Matches the documented WPB scalar-assignment subset without importing or
    executing any benchmark snapshot file.
    """
    matches = re.findall(r"```edits\s*\n(.*?)```", answer, re.I | re.S)
    if not matches:
        raise PolicyError("answer has no edits block")
    w = {name: np.zeros(shape, dtype=np.float32) for name, shape in SHAPES.items()}
    try:
        for statement in ast.parse(matches[-1]).body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                raise PolicyError("expected one scalar assignment per edit")
            target = statement.targets[0]
            if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name) or target.value.id not in SHAPES:
                raise PolicyError("invalid edit target")
            indices = target.slice.elts if isinstance(target.slice, ast.Tuple) else [target.slice]
            index = tuple(ast.literal_eval(v) for v in indices)
            shape = SHAPES[target.value.id]
            if len(index) != len(shape) or any(type(v) is not int or not 0 <= v < size for v, size in zip(index, shape)):
                raise PolicyError("edit index out of bounds")
            value = ast.literal_eval(statement.value)
            if type(value) not in (float, int) or not np.isfinite(value) or abs(value) > 1e6:
                raise PolicyError("invalid edit value")
            w[target.value.id][index] = value
    except (SyntaxError, TypeError, ValueError, OverflowError) as exc:
        raise PolicyError(f"invalid edits: {exc}") from exc
    validate_weights(w)
    return w
