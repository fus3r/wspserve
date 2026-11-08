"""Restricted AST reader and exact integer interpreter. Policy text is never run."""
from __future__ import annotations

import ast
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

ACTION_NAMES = ("NOOP", "LEFT", "UP", "RIGHT", "DOWN", "FIRE")
ACTIONS = {name: i for i, name in enumerate(ACTION_NAMES)}
SIZE = 600


class PolicyError(ValueError):
    """An unsupported program or one that cannot fit the fixed network."""


@dataclass(frozen=True)
class Linear:
    coefficients: tuple[int, ...] = (0,) * SIZE
    constant: int = 0

    def add(self, other: Linear, sign: int = 1) -> Linear:
        return Linear(tuple(a + sign * b for a, b in zip(self.coefficients, other.coefficients)),
                      self.constant + sign * other.constant)

    def scale(self, value: int) -> Linear:
        return Linear(tuple(value * a for a in self.coefficients), value * self.constant)


@dataclass(frozen=True)
class Atom:
    """Canonical predicate: dot(coefficients, x) >= threshold."""
    coefficients: tuple[int, ...]
    threshold: int

    def holds(self, x) -> bool:
        # Python integers retain exact semantics even for large source constants.
        return sum(a * int(x[i]) for i, a in enumerate(self.coefficients) if a) >= self.threshold


@dataclass(frozen=True)
class Equal:
    """expr == k as one literal: lower = [expr >= k], upper = [expr >= k + 1].

    Each is a (Atom, positive) pair. upper implies lower, so in +-1 unit encoding the
    negation (expr != k) is upper - lower + 1, which is again +-1: a single clause literal.
    The compiler therefore never expands NOT(expr == k) into two rules (v2, v1 did)."""
    lower: tuple
    upper: tuple

    def holds(self, x) -> bool:
        (a, pa), (b, pb) = self.lower, self.upper
        return (a.holds(x) == pa) and not (b.holds(x) == pb)


# Conditions are bool, ('lit', Atom | Equal, positive), or ('and'/'or', children).
def junction(kind: str, children):
    identity = kind == "and"
    flat = []
    for child in children:
        if isinstance(child, bool):
            if child != identity:
                return child
        elif child[0] == kind:
            flat.extend(child[1])
        else:
            flat.append(child)
    flat = list(dict.fromkeys(flat))
    if not flat:
        return identity
    return flat[0] if len(flat) == 1 else (kind, tuple(flat))


def negate(condition):
    if isinstance(condition, bool):
        return not condition
    if condition[0] == "lit":
        return ("lit", condition[1], not condition[2])
    return junction("or" if condition[0] == "and" else "and",
                    [negate(child) for child in condition[1]])


def threshold(expression: Linear, value: int):
    coefficients = expression.coefficients
    value -= expression.constant
    divisor = math.gcd(*coefficients)
    if divisor:
        coefficients = tuple(a // divisor for a in coefficients)
        value = -(-value // divisor)  # ceil, valid on the integer lattice
    minimum = sum(min(0, a) for a in coefficients)
    maximum = sum(max(0, a) for a in coefficients)
    if value <= minimum:
        return True
    if value > maximum:
        return False
    # Share a unit with the complementary inequality.
    positive = next(a for a in coefficients if a) > 0
    if not positive:
        coefficients = tuple(-a for a in coefficients)
        value = 1 - value
    return ("lit", Atom(coefficients, value), positive)


def condition_holds(condition, x):
    if isinstance(condition, bool):
        return condition
    if condition[0] == "lit":
        return condition[1].holds(x) == condition[2]
    values = (condition_holds(child, x) for child in condition[1])
    return all(values) if condition[0] == "and" else any(values)


def _simplify_clauses(clauses):
    result = []
    for clause in clauses:
        clause = frozenset(clause)
        if any((atom, not sign) in clause for atom, sign in clause):
            continue
        if any(old <= clause for old in result):
            continue
        result = [old for old in result if not clause < old]
        result.append(clause)
    return result


def expand(condition):
    """DNF clauses, preserving the priority of the source rule containing them."""
    if isinstance(condition, bool):
        return [frozenset()] if condition else []
    if condition[0] == "lit":
        return [frozenset([(condition[1], condition[2])])]
    groups = [expand(child) for child in condition[1]]
    if condition[0] == "or":
        result = _simplify_clauses(itertools.chain.from_iterable(groups))
    else:
        result = [frozenset()]
        for group in groups:
            result = _simplify_clauses(a | b for a in result for b in group)
            if len(result) > 15:
                raise PolicyError("condition expands to more than 15 rules")
    if len(result) > 15:
        raise PolicyError("condition expands to more than 15 rules")
    return result


@dataclass(frozen=True)
class Program:
    rules: tuple
    default: int
    expanded: tuple
    atoms: tuple[Atom, ...]
    source_atoms: int
    filename: str

    def action(self, observation) -> int:
        # Avoid converting through float32: the language has integer semantics.
        x = observation.reshape(-1) if hasattr(observation, "reshape") else observation
        if len(x) != SIZE or any(v not in (0, 1) for v in x):
            raise PolicyError("observation must contain exactly 600 binary cells")
        for condition, action in self.rules:
            if condition_holds(condition, x):
                return action
        return self.default

    def actions(self, observations):
        """Exact batched interpreter, independent of the compiled network."""
        import numpy as np
        x = np.asarray(observations)
        if x.ndim != 2 or x.shape[1] != SIZE or not np.all((x == 0) | (x == 1)):
            raise PolicyError("observations must have shape (n, 600) and binary cells")
        # The compiler accepts a bounded integer range, object arithmetic is the
        # fallback for interpreting a parsed but not yet compiled large program.
        maximum = max((sum(abs(a) for a in atom.coefficients) for atom in self.atoms), default=0)
        dtype = np.int64 if maximum < 2**62 else object
        truth = {atom: x.astype(dtype) @ np.array(atom.coefficients, dtype=dtype) >= atom.threshold
                 for atom in self.atoms}

        def check(condition):
            if isinstance(condition, bool):
                return np.full(len(x), condition)
            if condition[0] == "lit":
                atom = condition[1]
                if isinstance(atom, Equal):
                    lower = truth[atom.lower[0]] == atom.lower[1]
                    upper = truth[atom.upper[0]] == atom.upper[1]
                    value = lower & ~upper
                else:
                    value = truth[atom]
                return value if condition[2] else ~value
            values = [check(child) for child in condition[1]]
            return (np.logical_and.reduce(values) if condition[0] == "and"
                    else np.logical_or.reduce(values))

        answer = np.full(len(x), self.default, dtype=np.int64)
        for condition, action in reversed(self.rules):
            answer[check(condition)] = action
        return answer


class Reader:
    def __init__(self, filename):
        self.filename = filename
        self.source_atoms = set()

    def fail(self, node, message):
        raise PolicyError(f"{self.filename}:{getattr(node, 'lineno', 1)}: {message}")

    def integer(self, node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return self.integer(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        self.fail(node, "expected an integer literal")

    def index(self, node, limit):
        value = self.integer(node)
        if not 0 <= value < limit:
            self.fail(node, f"index must be in 0..{limit - 1}")
        return value

    def selection(self, node):
        if isinstance(node, ast.List):
            return sorted({self.index(item, 10) for item in node.elts})
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "range" and not node.keywords and 1 <= len(node.args) <= 3):
            args = [self.integer(arg) for arg in node.args]
            if len(args) == 3 and args[2] == 0:
                self.fail(node, "range step cannot be zero")
            value = range(*args)
            try:
                length = len(value)
            except OverflowError:
                self.fail(node, "range selects cells outside 0..9")
            if length and (min(value[0], value[-1]) < 0 or max(value[0], value[-1]) > 9):
                self.fail(node, "range selects cells outside 0..9")
            return list(value)
        self.fail(node, "expected a list of indices or range(...)")

    def expression(self, node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Linear(constant=node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return self.expression(node.operand).scale(-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp):
            left, right = self.expression(node.left), self.expression(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return left.add(right, -1 if isinstance(node.op, ast.Sub) else 1)
            if isinstance(node.op, ast.Mult):
                if not any(left.coefficients):
                    return right.scale(left.constant)
                if not any(right.coefficients):
                    return left.scale(right.constant)
            self.fail(node, "only +, -, and integer * expression are linear")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name, args = node.func.id, node.args
            coefficients = [0] * SIZE
            if name == "C" and len(args) == 3:
                ch, row, col = self.index(args[0], 6), self.index(args[1], 10), self.index(args[2], 10)
                coefficients[(row * 10 + col) * 6 + ch] = 1
            elif name == "COUNT" and len(args) == 3:
                ch = self.index(args[0], 6)
                for row in self.selection(args[1]):
                    for col in self.selection(args[2]):
                        coefficients[(row * 10 + col) * 6 + ch] = 1
            elif name in ("X", "Y") and len(args) == 1:
                ch = self.index(args[0], 6)
                for row in range(10):
                    for col in range(10):
                        coefficients[(row * 10 + col) * 6 + ch] = col if name == "X" else row
            else:
                self.fail(node, "expected C(ch,r,c), COUNT(ch,rows,cols), X(ch), or Y(ch)")
            return Linear(tuple(coefficients))
        self.fail(node, "unsupported linear expression")

    def condition(self, node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name in ("AND", "OR") and node.args:
                return junction(name.lower(), [self.condition(arg) for arg in node.args])
            if name == "NOT" and len(node.args) == 1:
                return negate(self.condition(node.args[0]))
            if name == "ANY" and len(node.args) == 3:
                count = ast.Call(func=ast.Name(id="COUNT"), args=node.args, keywords=[])
                result = threshold(self.expression(count), 1)
                self.source_atoms.add(result)
                return result
            self.fail(node, "expected ANY(...), AND(...), OR(...), or NOT(condition)")
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
            expression, value = self.expression(node.left), self.integer(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.GtE):
                result = threshold(expression, value)
            elif isinstance(op, ast.Gt):
                result = threshold(expression, value + 1)
            elif isinstance(op, ast.LtE):
                result = negate(threshold(expression, value + 1))
            elif isinstance(op, ast.Lt):
                result = negate(threshold(expression, value))
            elif isinstance(op, ast.Eq):
                lower, upper = threshold(expression, value), threshold(expression, value + 1)
                if isinstance(lower, bool) or isinstance(upper, bool):
                    result = junction("and", [lower, negate(upper)])
                elif lower == upper:
                    result = False
                else:
                    result = ("lit", Equal(lower[1:], upper[1:]), True)
            else:
                self.fail(node, "comparison must be >=, <=, >, <, or ==")
            self.source_atoms.add(result)
            return result
        self.fail(node, "expected a comparison or condition call")

    def action(self, node):
        if isinstance(node, ast.Name) and node.id in ACTIONS:
            return ACTIONS[node.id]
        self.fail(node, "action must be NOOP, LEFT, UP, RIGHT, DOWN, or FIRE")


def parse(source: str, filename: str = "<policy>") -> Program:
    reader = Reader(filename)
    try:
        tree = ast.parse(source, filename=filename)
    except (SyntaxError, ValueError) as exc:
        raise PolicyError(f"{filename}: invalid Python syntax: {exc}") from exc
    definitions = {}
    for statement in tree.body:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id in ("rules", "default")):
            name = statement.targets[0].id
            if name in definitions:
                reader.fail(statement, f"{name} is defined twice")
            definitions[name] = statement.value
        else:
            reader.fail(statement, "program may only assign rules and default")
    if set(definitions) != {"rules", "default"}:
        raise PolicyError(f"{filename}: define both rules and default exactly once")
    rule_list = definitions["rules"]
    if not isinstance(rule_list, ast.List):
        reader.fail(rule_list, "rules must be a list of (condition, ACTION) tuples")
    rules, expanded = [], []
    for entry in rule_list.elts:
        if not isinstance(entry, ast.Tuple) or len(entry.elts) != 2:
            reader.fail(entry, "each rule must be a (condition, ACTION) tuple")
        condition, action = reader.condition(entry.elts[0]), reader.action(entry.elts[1])
        rules.append((condition, action))
        expanded.extend((clause, action) for clause in expand(condition))
        if len(expanded) > 15:
            reader.fail(entry, "program expands to more than 15 rules")
    if len(reader.source_atoms) > 32:
        raise PolicyError(f"{filename}: more than 32 distinct source atoms")

    def atoms_in(condition):
        if isinstance(condition, bool):
            return []
        if condition[0] == "lit":
            atom = condition[1]
            return [atom.lower[0], atom.upper[0]] if isinstance(atom, Equal) else [atom]
        return itertools.chain.from_iterable(atoms_in(c) for c in condition[1])

    atoms = tuple(dict.fromkeys(itertools.chain.from_iterable(atoms_in(c) for c, _ in rules)))
    if len(atoms) > 32:
        raise PolicyError(f"{filename}: needs {len(atoms)} distinct threshold atoms; maximum 32 "
                          "(equality can require two thresholds)")
    return Program(tuple(rules), reader.action(definitions["default"]), tuple(expanded),
                   atoms, len(reader.source_atoms), filename)


def load(path) -> Program:
    path = Path(path)
    return parse(path.read_text(encoding="utf-8"), str(path))
