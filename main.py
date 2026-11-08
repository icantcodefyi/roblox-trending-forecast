#!/usr/bin/env python3
"""Hourly chart transition model: a forecast of Roblox's Top Trending top 10 for every hour of a 24-hour window.

Reads $MAKEFUN_INPUT_DIR (else /input): history.ndjson, one read of the chart per line, and task.json, the window.
Writes $MAKEFUN_OUTPUT_DIR (else /output)/forecast.json: per place 1-10, a probability for every game it names.

The model, per hourly slot of the window, then averaged over the slots (the best single forecast under a log score
when one forecast is scored against every slot):
  1. Each game on the last read's chart gets a distribution over its future rank: its log rank drifts with its
     recent momentum (a damped trend) and spreads with the horizon (a two-scale normal mixture, heavy tailed).
  2. Each game may leave the chart (a hazard by rank band); new games may enter (a mass per place).
  3. The games-by-places matrix is balanced (Sinkhorn) so each place holds one game and each game one place.
  4. Every game of the last chart is named at every place, so the unnamed mass of a place is shared only by games
     that are new to the chart: that is the mass left for them.
Standard library only, no network, deterministic: the same history always gives the same bytes.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

PLACES = 10

# Parameters (log-rank units, hours), tuned on rolling-origin backtests of the public archive's reads (every read as
# the last of a history, scored on the reads of the next 24 hours), then hedged toward wider tails: a forecast too
# sharp loses far more in a day the chart moves than one too wide loses in a quiet day.
DEFAULTS = {
    "momentum": 0.148,  # share of the measured log-rank velocity carried forward
    "momentum_tau": 17.5,  # hours: the velocity decays with this time constant
    "vel_window": 65.0,  # hours of history the velocity is measured over
    "vel_prior_n": 0.135,  # shrinkage of the velocity toward 0 (pseudo-reads)
    "sigma0": 0.0067,  # log-rank spread at horizon 0
    "sigma1": 0.065,  # log-rank spread growth per sqrt(hour)
    "tail_w": 0.136,  # weight of the wide component
    "tail_mult": 3.0,  # the wide component's spread, as a multiple
    "exit_top": 0.003,  # per-hour hazard of leaving the chart, ranks 1-10
    "exit_mid": 0.048,  # ranks 11-40
    "exit_low": 0.044,  # ranks 41+
    "entrant_rate": 0.0015,  # per-hour growth of a top-10 place's mass for games new to the chart
    "entrant_cap": 0.12,  # most of a place's mass left to new games
    "entrant_slope": 0.0,  # places further down get more: mass * (k / 5.5) ** slope
    "surge_coef": 0.547,  # log-rank climb per unit of player surge (log ratio), reached over surge_tau
    "surge_tau": 16.2,  # hours for a surge to show in the rank
    "surge_from": 26.5,  # the surge's reference: reads this many hours ago (about the same hour yesterday) ...
    "surge_to": 17.8,  # ... to this many
    "surge_clip": 2.5,  # largest surge counted, either way
    "matrix_places": 40,  # places modelled one by one; the rest of the chart and off it share one column
    "sinkhorn_iters": 20,
}


def parse_time(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()


def load_reads(path):
    """Every usable read of the chart, oldest first: {'t', 'order', 'playing'}."""
    reads = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                t = parse_time(d["at"])
            except (ValueError, KeyError, TypeError):
                continue
            if d.get("ok") is False:
                continue
            rows = d.get("top_trending")
            if not isinstance(rows, list):
                continue
            clean = []
            for i, r in enumerate(rows):
                if not isinstance(r, dict):
                    continue
                u = r.get("universe_id")
                if not isinstance(u, int) or isinstance(u, bool) or u < 1:
                    continue
                rank = r.get("rank")
                rank = rank if isinstance(rank, (int, float)) and not isinstance(rank, bool) else i + 1
                clean.append((rank, i, u, r.get("playing")))
            clean.sort()
            order, seen, playing = [], set(), {}
            for _, _, u, p in clean:
                if u in seen:
                    continue
                seen.add(u)
                order.append(u)
                if isinstance(p, (int, float)) and not isinstance(p, bool):
                    playing[u] = float(p)
            if order:
                reads.append({"t": t, "order": order, "playing": playing})
    reads.sort(key=lambda r: r["t"])
    return reads


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def positions(reads, since_t):
    """Each read at or after since_t as (hours relative to the last read, {universe_id: rank}, chart length)."""
    last_t = reads[-1]["t"]
    out = []
    for r in reads:
        if r["t"] >= since_t:
            out.append(((r["t"] - last_t) / 3600.0, {u: i + 1 for i, u in enumerate(r["order"])}, len(r["order"])))
    return out


def velocity(pos_list, uid, prm):
    """Least-squares slope of log rank per hour over the recent reads (absent from a read: just below its chart)."""
    pts = []
    for x, pos, n in pos_list:
        y = math.log(pos[uid]) if uid in pos else math.log(n + 1)
        pts.append((x, y))
    if len(pts) < 2:
        return 0.0
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    if sxx <= 1e-9:
        return 0.0
    slope = sxy / sxx
    n = len(pts) - 1
    return slope * n / (n + prm["vel_prior_n"])


def player_surge(reads, prm):
    """Each game's players now against its own level surge_from..surge_to hours ago, both relative to the chart's
    median that read (which takes out the day's common rise and fall). Positive: the game is pulling players in."""
    last = reads[-1]
    rel = {}
    for r in reads:
        age = (last["t"] - r["t"]) / 3600.0
        if r is not last and not (prm["surge_to"] <= age <= prm["surge_from"]):
            continue
        vals = sorted(v for v in r["playing"].values() if v > 0)
        if not vals:
            continue
        med = vals[len(vals) // 2]
        for u, p in r["playing"].items():
            if p > 0:
                rel.setdefault(u, []).append((r is last, math.log(p / med)))
    out = {}
    for u in last["order"]:
        pts = rel.get(u, [])
        now = [y for is_last, y in pts if is_last]
        past = [y for is_last, y in pts if not is_last]
        out[u] = (now[0] - sum(past) / len(past)) if now and past else 0.0
    return out


def exit_hazard(rank, prm):
    if rank <= 10:
        return prm["exit_top"]
    if rank <= 40:
        return prm["exit_mid"]
    return prm["exit_low"]


def slot_matrix(chart, vel, surge, h, prm):
    """Games x places 1..C probabilities at horizon h hours (C = matrix_places), balanced by Sinkhorn.

    Column C+1 holds everything else: a rank below C or off the chart. Each game's row sums to 1; place k's column
    sums to 1 minus its mass for games new to the chart; the last column takes the balance.
    """
    n = len(chart)
    cols = min(n, int(prm["matrix_places"]))
    bounds = [math.log(k + 0.5) for k in range(1, cols + 1)]
    sig = math.sqrt(prm["sigma0"] ** 2 + prm["sigma1"] ** 2 * h)
    sig_w = sig * prm["tail_mult"]
    tw = prm["tail_w"]
    tau = prm["momentum_tau"]
    drift_h = tau * (1.0 - math.exp(-h / tau)) if tau > 0 else h
    surge_h = 1.0 - math.exp(-h / prm["surge_tau"])
    rows = []
    for i, u in enumerate(chart):
        rank = i + 1
        s = max(-prm["surge_clip"], min(prm["surge_clip"], surge[u]))
        mu = math.log(rank) + prm["momentum"] * vel[u] * drift_h - prm["surge_coef"] * s * surge_h
        stay = math.exp(-exit_hazard(rank, prm) * h)
        prev = 0.0
        row = []
        for b in bounds:
            c = (1 - tw) * norm_cdf((b - mu) / sig) + tw * norm_cdf((b - mu) / sig_w)
            row.append(max(c - prev, 0.0) * stay)
            prev = c
        row.append(max(1.0 - sum(row), 1e-12))
        rows.append(row)
    # Each place's mass for games new to the chart grows with the horizon and with the place.
    base = min(prm["entrant_rate"] * h, prm["entrant_cap"])
    ent = [min(base * ((k + 1) / 5.5) ** prm["entrant_slope"], 0.6) for k in range(cols)]
    col_t = [1.0 - e for e in ent]
    col_t.append(max(n - sum(col_t), 1e-9))
    width = cols + 1
    for _ in range(int(prm["sinkhorn_iters"])):
        sums = [0.0] * width
        for row in rows:
            for k in range(width):
                sums[k] += row[k]
        f = [col_t[k] / sums[k] if sums[k] > 0 else 0.0 for k in range(width)]
        for row in rows:
            s = 0.0
            for k in range(width):
                row[k] *= f[k]
                s += row[k]
            if s > 0:
                inv = 1.0 / s
                for k in range(width):
                    row[k] *= inv
    # Finish on the columns so each scored place sums to its target exactly.
    top = min(PLACES, cols)
    sums = [sum(row[k] for row in rows) for k in range(top)]
    out = [[row[k] * col_t[k] / sums[k] if sums[k] > 0 else 0.0 for k in range(top)] for row in rows]
    return out, ent[:top]


def forecast(reads, task, prm=None):
    prm = dict(DEFAULTS, **(prm or {}))
    places = int(task.get("places", PLACES))
    win = task.get("window", {})
    t0 = parse_time(win.get("from")) if win.get("from") else reads[-1]["t"]
    slots = int(win.get("slots", 24))
    slot_s = float(win.get("slot_minutes", 60)) * 60.0
    last = reads[-1]
    chart = last["order"]
    pos_list = positions(reads, last["t"] - prm["vel_window"] * 3600.0)
    vel = {u: velocity(pos_list, u, prm) for u in chart}
    surge = player_surge(reads, prm)
    acc = [[0.0] * places for _ in chart]
    ent_acc = [0.0] * places
    for s in range(slots):
        h = max((t0 + s * slot_s + 120.0 - last["t"]) / 3600.0, 0.05)
        m, ent = slot_matrix(chart, vel, surge, h, prm)
        for g in range(len(chart)):
            for k in range(min(places, len(m[g]))):
                acc[g][k] += m[g][k] / slots
        for k in range(len(ent)):
            ent_acc[k] += ent[k] / slots
    spare = [u for r in reversed(reads[:-1]) for u in r["order"]]
    return build_doc(chart, acc, places, spare)


def build_doc(chart, acc, places, spare=()):
    rows = []
    for k in range(places):
        # Every game of the chart is named (at least 1e-7), so the place's unnamed rest goes only to new games.
        cand = sorted(((acc[g][k], chart[g]) for g in range(len(chart)) if k < len(acc[g])), key=lambda x: (-x[0], x[1]))[:100]
        total = sum(p for p, _ in cand)
        room = 0.999999 - 1e-7 * len(cand)
        scale = room / total if total > room else 1.0
        for p, u in cand:
            q = math.floor(p * scale * 1e7) / 1e7 + 1e-7
            rows.append({"universe_id": u, "place": k + 1, "p": round(q, 7)})
    # Point guess: assign places greedily by the largest remaining probability.
    cells = sorted(((acc[g][k], g, k) for g in range(len(chart)) for k in range(min(places, len(acc[g])))), key=lambda x: (-x[0], x[1], x[2]))
    used_g, top = set(), [None] * places
    for p, g, k in cells:
        if top[k] is None and g not in used_g:
            top[k] = chart[g]
            used_g.add(g)
    # Pad a short chart's guess: the rest of the chart, then games of earlier reads, then placeholder ids.
    pool = [u for u in list(chart) + list(spare) if u not in top]
    filler = 1
    for k in range(places):
        if top[k] is None:
            while pool and pool[0] in top:
                pool.pop(0)
            if pool:
                top[k] = pool.pop(0)
            else:
                while filler in top:
                    filler += 1
                top[k] = filler
    return {"probabilities": rows, "top10": top}


def persistence_doc(reads, places):
    """Last resort: the last chart's order at 0.5 on each place's own game (the unnamed rest spread over the chart)."""
    chart = reads[-1]["order"] if reads else []
    rows = [{"universe_id": u, "place": k + 1, "p": 0.5} for k, u in enumerate(chart[:places])]
    top = list(chart[:places])
    spare = [u for r in reversed(reads[:-1]) for u in r["order"]] if reads else []
    filler = 1
    for u in spare:
        if len(top) >= places:
            break
        if u not in top:
            top.append(u)
    while len(top) < places:
        if filler not in top:
            top.append(filler)
        filler += 1
    return {"probabilities": rows, "top10": top}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    try:
        with open(os.path.join(in_dir, "task.json"), encoding="utf-8") as fh:
            task = json.load(fh)
    except (OSError, ValueError):
        task = {}
    if not isinstance(task, dict):
        task = {}
    try:
        reads = load_reads(os.path.join(in_dir, "history.ndjson"))
    except (OSError, UnicodeDecodeError):
        reads = []
    places = task.get("places") if isinstance(task.get("places"), int) and 1 <= task.get("places") <= PLACES else PLACES
    try:
        if not reads:
            raise ValueError("no usable read in the history")
        doc = forecast(reads, task)
    except Exception as err:  # noqa: BLE001 - any failure still writes a valid forecast
        sys.stderr.write(f"model failed ({err!r}); writing the persistence forecast\n")
        doc = persistence_doc(reads, places)
    name = task.get("output") if isinstance(task.get("output"), str) else "forecast.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, os.path.basename(name) or "forecast.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
