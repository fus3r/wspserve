"""Recover a compact, bit-exact representation of a BF16 weight edit from (base, edited) bits."""
import numpy as np

from . import bf16
from .delta import ChannelScale, LowRank, SparseOverlay


def changed(base_bits, edited_bits):
    diff = np.asarray(base_bits) != np.asarray(edited_bits)
    return np.flatnonzero(diff.any(axis=1)), np.flatnonzero(diff.any(axis=0)), int(diff.sum())


def _scale_candidates(base32, target_bits, estimate, search):
    """float32 scales near ``estimate`` that reproduce one column's bits exactly."""
    center = np.float32(estimate)
    if center == 0:
        center = np.float32(np.copysign(np.finfo(np.float32).tiny, estimate or 1.0))
    raw = center.view(np.int32) + np.arange(-search, search + 1, dtype=np.int64)
    raw = raw[(raw > np.iinfo(np.int32).min) & (raw < np.iinfo(np.int32).max)].astype(np.int32)
    values = raw.view(np.float32)
    values = values[np.isfinite(values)]
    bits = bf16.from_f32(base32[None, :] * values[:, None])
    return values[np.all(bits == target_bits[None, :], axis=1)]


def _snap(values):
    """Prefer the shortest decimal (e.g. 0.5, 1.1) among exact float32 candidates."""
    for digits in range(1, 8):
        rounded = np.unique(np.round(values.astype(np.float64), digits).astype(np.float32))
        hits = [v for v in rounded if np.any(values == v)]
        if hits:
            return np.float32(hits[0])
    return np.float32(values[len(values) // 2])


def _shortest_decimal(low, high):
    """Shortest decimal number in [low, high] (e.g. 1.1 rather than 1.10000002)."""
    for digits in range(0, 10):
        q = 10.0 ** digits
        c = np.ceil(low * q) / q
        if c <= high:
            return c
    return 0.5 * (low + high)


def _scale_interval(base32, target_bits, shrink=0.5):
    """Range of c for which c * b rounds to the target bits in every nonzero row (single rounding)."""
    nonzero = base32 != 0
    low, high = bf16.rounding_interval(target_bits[nonzero], shrink=shrink)
    b = base32[nonzero].astype(np.float64)
    r1, r2 = low / b, high / b
    return float(np.max(np.minimum(r1, r2))), float(np.min(np.maximum(r1, r2)))


def fit_channel_scale(base_bits, edited_bits, columns, *, search=96):
    """ChannelScale reproducing every edited entry of ``columns`` exactly, or None."""
    columns = np.asarray(columns, dtype=np.int64)
    base32 = bf16.to_f32(base_bits[:, columns])
    target = np.asarray(edited_bits)[:, columns]
    candidate_sets, assigned = [], []
    for j in range(len(columns)):
        t, b = target[:, j], base32[:, j]
        if not np.any(t & 0x7FFF):
            if np.all(t == 0):
                candidate_sets.append(np.array([0.0], dtype=np.float32))
                assigned.append(True)
                continue
            if np.array_equal(bf16.from_f32(b * np.float32(0.0)), t):
                candidate_sets.append(np.array([0.0], dtype=np.float32))
                assigned.append(False)
                continue
            return None
        nonzero = b != 0
        if not nonzero.any() or np.any(t[~nonzero] & 0x7FFF):
            return None
        # The exact scale lies (up to fp32 double rounding at ties) in the intersection of every
        # row's rounding interval. Try short decimals first (the editor wrote 0.5, 1.1, 1.5...),
        # then the float32 grid around the interval midpoint and the median ratio.
        low, high = _scale_interval(b, t)
        median = float(np.median(bf16.to_f64(t[nonzero]) / b[nonzero].astype(np.float64)))
        decimals = [round(median, d) for d in range(1, 7)]
        if low <= high:
            decimals.insert(0, _shortest_decimal(low, high))
        exact = np.zeros(0, dtype=np.float32)
        for c in dict.fromkeys(np.float32(d) for d in decimals):
            if np.array_equal(bf16.from_f32(b * c), t):
                exact = np.array([c], dtype=np.float32)
                break
        for estimate in ([0.5 * (low + high)] if low <= high else []) + [median]:
            if len(exact):
                break
            exact = _scale_candidates(b, t, estimate, search)
        if not len(exact):
            return None
        candidate_sets.append(exact)
        assigned.append(False)
    # Shortest-decimal snapping makes columns edited with the same scale report it identically.
    scales = np.array([_snap(c) for c in candidate_sets], dtype=np.float32)
    return ChannelScale(columns, scales, np.asarray(assigned, dtype=bool))


def _top(matrix, rank, start=None, iterations=None):
    """Top singular triplets by subspace iteration (cheap for rank << min(shape))."""
    if min(matrix.shape) <= 4 * rank + 8:
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        return u[:, :rank] * s[:rank], vt[:rank], s
    rng = np.random.default_rng(0)
    if iterations is None:
        iterations = 4 if start is not None else 40
    q = start if start is not None else matrix @ rng.standard_normal((matrix.shape[1], rank))
    q = np.linalg.qr(q)[0]
    for _ in range(iterations):
        q = np.linalg.qr(matrix @ (matrix.T @ q))[0]
    u, s, vt = np.linalg.svd(q.T @ matrix, full_matrices=False)
    return (q @ u) * s, vt, s


def spectral_rank(base_bits, edited_bits, columns, *, max_rank=4, gap=30.0):
    """Smallest r <= max_rank with sigma_r / sigma_{r+1} >= gap, or 0 if the delta has no such gap.

    BF16 rounding adds a full-rank noise floor to any delta, so a fixed residual threshold
    cannot pick the rank, a large drop between consecutive singular values can.
    """
    columns = np.asarray(columns, dtype=np.int64)
    delta = bf16.to_f64(np.asarray(edited_bits)[:, columns]) - bf16.to_f64(np.asarray(base_bits)[:, columns])
    _, _, spectrum = _top(delta, min(max_rank + 1, min(delta.shape)))
    for r in range(1, min(max_rank, len(spectrum) - 1) + 1):
        if spectrum[r] == 0 or spectrum[r - 1] / spectrum[r] >= gap:
            return r, [float(v) for v in spectrum]
    return 0, [float(v) for v in spectrum]


def fit_low_rank(base_bits, edited_bits, columns, rank, *, refine=None, patience=6):
    """Rank-r factors on ``columns`` plus the fixups needed for a bit-exact rebuild.

    Plain SVD of the rounded delta gives factors whose re-rounding misses many entries.
    Refinement alternates between the rank-r matrices and the box of pre-rounding values
    that round to the published bits, each iterate is verified exactly and the one with
    the fewest mismatches is kept.
    """
    columns = np.asarray(columns, dtype=np.int64)
    base64 = bf16.to_f64(base_bits[:, columns])
    target = np.asarray(edited_bits)[:, columns]
    if refine is None:  # each step costs one exact re-rounding of the support
        refine = 300 if target.size <= 200_000 else 60
    low, high = bf16.rounding_interval(target)
    delta = bf16.to_f64(target) - base64
    b, a, spectrum = _top(delta, rank)

    def misses(b, a):
        return int(np.count_nonzero(bf16.from_f64(base64 + b @ a) != target))

    best = (misses(b, a), b, a, 0)
    plain = best[0]
    for step in range(1, refine + 1):
        if best[0] == 0 or step - best[3] > patience:
            break
        projected = np.clip(base64 + b @ a, low, high) - base64
        b, a, _ = _top(projected, rank, start=b)
        m = misses(b, a)
        if m < best[0]:
            best = (m, b, a, step)
    _, b, a, steps = best
    full_a = np.zeros((rank, base_bits.shape[1]))
    full_a[:, columns] = a
    rebuilt = np.array(base_bits, copy=True)
    rebuilt[:, columns] = bf16.from_f64(base64 + b @ a)
    wrong = np.flatnonzero(rebuilt.reshape(-1) != np.asarray(edited_bits).reshape(-1))
    delta_norm = np.linalg.norm(delta)
    stats = {"rank": rank, "plain_svd_mismatches": plain, "refined_mismatches": int(len(wrong)),
             "refinement_steps_used": steps,
             "relative_residual": float(np.linalg.norm(delta - b @ a) / delta_norm) if delta_norm else 0.0,
             "leading_singular_values": [float(v) for v in spectrum[: min(len(spectrum), 8)]]}
    return LowRank(b, full_a, wrong.astype(np.int64), np.asarray(edited_bits).reshape(-1)[wrong]), stats


def analyze(base_bits, edited_bits, *, max_rank=4):
    """All exact representations found for one tensor, with the cheapest marked."""
    base_bits, edited_bits = np.asarray(base_bits), np.asarray(edited_bits)
    rows, cols, entries = changed(base_bits, edited_bits)
    flat = np.flatnonzero(base_bits.reshape(-1) != edited_bits.reshape(-1))
    options = [SparseOverlay(flat.astype(np.int64), edited_bits.reshape(-1)[flat])]
    details = {"changed_entries": entries, "changed_rows": int(len(rows)), "changed_columns": int(len(cols))}
    scale = fit_channel_scale(base_bits, edited_bits, cols)
    if scale is not None:
        options.append(scale)
    else:
        rank, spectrum = spectral_rank(base_bits, edited_bits, cols, max_rank=max_rank)
        details["spectral_rank"] = {"rank": rank, "leading_singular_values": spectrum}
        if rank:
            low_rank, stats = fit_low_rank(base_bits, edited_bits, cols, rank)
            details[f"low_rank_{rank}"] = stats
            options.append(low_rank)
    for option in options:
        if not np.array_equal(option.materialize(base_bits), edited_bits):
            raise AssertionError(f"{option.kind} does not rebuild the edited bits")
    best = min(options, key=lambda o: o.nbytes())
    details["representations"] = [o.describe() for o in options]
    details["smallest_exact"] = best.kind
    return best, options, details
