"""
Compare EvoX (this run) vs advisor-refresh (both epochs).
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── EvoX data (parsed from run log) ───────────────────────────────────────────
LOG = "grayscale/skydiscover_runs/run1.log"

metrics_re = re.compile(
    r"Metrics: combined_score=[0-9.]+, geomean_us=([0-9.]+), pass_rate=([0-9.]+)"
)
start_re = re.compile(
    r"Evaluated program [a-f0-9-]+ in [0-9.]+s: combined_score=[0-9.]+, geomean_us=([0-9.]+), pass_rate=1"
)

raw_times = []
with open(LOG) as f:
    for line in f:
        if not raw_times:
            m = start_re.search(line)
            if m:
                raw_times.append(float(m.group(1)))
                continue
        m = metrics_re.search(line)
        if m:
            raw_times.append(float(m.group(1)))

evox_iters, evox_times, evox_kinds = [], [], []
best_so_far = float("inf")
for i, t in enumerate(raw_times):
    evox_iters.append(i)
    evox_times.append(t)
    if t < best_so_far:
        best_so_far = t
        evox_kinds.append("keep")
    else:
        evox_kinds.append("discard")

# ── Advisor-refresh data (epoch 1 + epoch 2) ──────────────────────────────────
REFRESH_ITER = 15

epoch1_rows = [
    (0,  102.48, "keep"),
    (1,  102.69, "discard"),
    (2,  412.47, "discard"),
    (3,   60.27, "keep"),
    (4,   65.26, "discard"),
    (5,   63.32, "discard"),
    (6,   63.92, "discard"),
    (7,   65.35, "discard"),
    (8,    0.00, "crash"),
    (9,    0.00, "crash"),
    (10,  62.17, "discard"),
    (11,  62.69, "discard"),
    (12,   0.00, "crash"),
    (13,  62.42, "discard"),
    (14,  62.74, "discard"),
    (15,  62.24, "discard"),
]
epoch2_rows = [
    (15,  62.90, "keep"),
    (16,  68.65, "discard"),
    (17,  93.36, "discard"),
    (18, 408.38, "discard"),
    (19,  62.80, "keep"),
    (20,   0.00, "crash"),
    (21, 138.18, "discard"),
    (22,   0.00, "crash"),
    (23,  68.81, "discard"),
    (24,  61.74, "keep"),
    (25,  64.81, "discard"),
]

refresh_iters, refresh_times, refresh_kinds = [], [], []
for it, t, k in epoch1_rows + epoch2_rows:
    refresh_iters.append(it)
    refresh_times.append(t)
    refresh_kinds.append(k)

# ── Best-over-time step lines ─────────────────────────────────────────────────
def best_step(iters, times, kinds):
    bx, by = [], []
    best = float("inf")
    for it, t, k in sorted(zip(iters, times, kinds)):
        if k == "keep" and t > 0:
            best = t
        if best < float("inf"):
            bx.append(it)
            by.append(best)
    return bx, by

ref_bx, ref_by = best_step(refresh_iters, refresh_times, refresh_kinds)
evox_bx, evox_by = best_step(evox_iters, evox_times, evox_kinds)

ref_best  = min(t for t, k in zip(refresh_times, refresh_kinds) if k == "keep" and t > 0)
evox_best = min(evox_by) if evox_by else float("inf")

# ── Y-axis (negative latency, clip outliers) ──────────────────────────────────
CLIP_US = 450.0
all_valid = [t for t in refresh_times + evox_times if 0 < t <= CLIP_US]
y_hi = -(min(all_valid) * 0.82)
y_lo = -(CLIP_US * 1.08)

def ny(t):
    return max(-t, y_lo) if t > 0 else y_lo

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
fig.subplots_adjust(top=0.75)

# EvoX — blue
evox_kx = [it for it, k in zip(evox_iters, evox_kinds) if k == "keep"]
evox_ky = [ny(evox_times[i]) for i, k in enumerate(evox_kinds) if k == "keep"]
evox_dx = [it for it, k in zip(evox_iters, evox_kinds) if k == "discard"]
evox_dy = [ny(evox_times[i]) for i, k in enumerate(evox_kinds) if k == "discard"]
if evox_kx:
    ax.scatter(evox_kx, evox_ky, c="#3b82f6", s=70, zorder=5, edgecolors="white", linewidths=0.5, label="EvoX keep")
if evox_dx:
    ax.scatter(evox_dx, evox_dy, c="#93c5fd", s=40, zorder=4, edgecolors="white", linewidths=0.3, alpha=0.8, label="EvoX discard")
if evox_bx:
    ax.step(evox_bx, [-t for t in evox_by], where="post", color="#3b82f6", linewidth=2, label="EvoX best", zorder=6)

# Advisor-refresh — purple
ref_kx = [refresh_iters[i] for i, k in enumerate(refresh_kinds) if k == "keep" and refresh_times[i] > 0]
ref_ky = [ny(refresh_times[i]) for i, k in enumerate(refresh_kinds) if k == "keep" and refresh_times[i] > 0]
ref_dx = [refresh_iters[i] for i, k in enumerate(refresh_kinds) if k == "discard"]
ref_dy = [ny(refresh_times[i]) for i, k in enumerate(refresh_kinds) if k == "discard"]
ref_cx = [refresh_iters[i] for i, k in enumerate(refresh_kinds) if k == "crash"]
if ref_kx:
    ax.scatter(ref_kx, ref_ky, c="#a855f7", s=70, zorder=5, edgecolors="white", linewidths=0.5, label="advisor-refresh keep")
if ref_dx:
    ax.scatter(ref_dx, ref_dy, c="#d8b4fe", s=40, zorder=4, edgecolors="white", linewidths=0.3, alpha=0.7, label="advisor-refresh discard")
if ref_bx:
    ax.step(ref_bx, [-t for t in ref_by], where="post", color="#a855f7", linewidth=2, label="advisor-refresh best", zorder=6)

# Crashes
all_cx = ref_cx
if all_cx:
    ax.scatter(all_cx, [y_lo] * len(all_cx), c="#fbbf24", s=40, zorder=3,
               marker="x", linewidths=1.5, label=f"crash ({len(all_cx)})", alpha=0.8)

# Epoch refresh marker
ax.axvline(x=REFRESH_ITER, color="#a855f7", linewidth=1.5, linestyle="--", alpha=0.7, zorder=2)
ax.annotate("← epoch refresh", xy=(REFRESH_ITER + 0.2, y_hi * 0.97),
            fontsize=9, color="#7c3aed", va="top")

ax.set_ylim(y_lo * 1.05, y_hi)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax.set_xlabel("Iteration #", fontsize=12)
ax.set_ylabel("Negative Latency (-μs)", fontsize=12)
ax.grid(True, alpha=0.3)

ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
          framealpha=0.9, fontsize=10, borderaxespad=0)

fig.text(0.5, 0.92,
         f"EvoX best: {evox_best:.2f} μs    |    advisor-refresh best: {ref_best:.2f} μs",
         ha="center", va="top", fontsize=11, fontweight="bold", color="#1e3a5f",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#a855f7", alpha=0.9))

fig.text(0.5, 0.995, "EvoX vs advisor-refresh — grayscale",
         ha="center", va="top", fontsize=14, fontweight="bold")

ax.annotate(
    f"(outliers > {CLIP_US:.0f} μs shown at floor)",
    xy=(0.01, 0.02), xycoords="axes fraction",
    ha="left", va="bottom", fontsize=9, color="#6b7280",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d1d5db", alpha=0.8),
)

out = "grayscale/skydiscover_runs/comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
print(f"EvoX best: {evox_best:.2f} μs  |  advisor-refresh best: {ref_best:.2f} μs")
