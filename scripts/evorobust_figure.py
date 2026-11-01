"""Reproduce Figure 1 in percentages and percentage points from saved summaries."""
import json, os, shutil
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "tables")
aud = json.load(open(os.path.join(V, "replay_audit.json")))
ins = {r["run"]: r for r in json.load(open(os.path.join(V, "instruments.json")))}
# execution = hardened harness, read from paired_analyses.json (instruments.json holds the study's cached scorer)
_pa = json.load(open(os.path.join(ROOT, "tables", "paired_analyses.json")))["pass_at_k_with_prompt_bootstrap"]
for _run in ins:
    for _k in (1, 32, 128):
        ins[_run]["exec@%d" % _k] = _pa[_run.split()[0]]["k=%d" % _k]["exec_hardened"]["mean"]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight"})
CLEAN, BROKEN, EXEC, GATE = "#4c72b0", "#c44e52", "#55a868", "#8172b2"

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65),
                         gridspec_kw={"width_ratios": [1, 1.15]})

# ---------- Panel A: validity, archived vs replayed
ax = axes[0]
labels = ["base Qwen-2.5-7B", "CE-trained", "SLJ-trained"]
arch = [100*aud["runs"][l]["archived"]["rate"] for l in labels]
rep = [100*aud["runs"][l]["replay"]["rate"] for l in labels]
x = np.arange(3); w = 0.34
ax.bar(x-w/2, arch, w, color=CLEAN, label="first run")
ax.bar(x+w/2, rep, w, color=BROKEN, label="second run (our replay)")
for xi, v in list(zip(x-w/2, arch)) + list(zip(x+w/2, rep)):
    ax.text(xi, v*1.4, ("%.2f" % v) if v < 1 else ("%.1f" % v),
            ha="center", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(["base", "CE\ntrained", "SLJ\ntrained"])
ax.set_ylabel("% of samples that do not parse")
ax.set_yscale("log")
ax.set_ylim(0.15, 600)
ax.set_yticks([0.1, 1, 10, 100])
ax.set_yticklabels(["0.1", "1", "10", "100"])
ax.set_title("Same checkpoints, two passes")
ax.legend(frameon=False, loc="upper center", ncol=1, handlelength=1.2,
          borderpad=0.2, labelspacing=0.25)
ax.grid(axis="y", alpha=0.2, lw=0.5); ax.set_axisbelow(True)

# ---------- Panel B: separation vs sampling budget
ax = axes[1]
b, t = ins["base Qwen-2.5-7B"], ins["SLJ-trained"]
ks = [1, 32, 128]
series = [("code execution", EXEC, "o", "-"),
          ("judge + syntax gate", GATE, "s", "-"),
          ("LLM judge", BROKEN, "^", "-")]
for name, col, mk, ls in series:
    key = {"code execution": "exec", "judge + syntax gate": "gated",
           "LLM judge": "judge"}[name]
    y = [100 * (b["%s@%d" % (key, k)] - t["%s@%d" % (key, k)]) for k in ks]
    ax.plot(ks, y, ls, marker=mk, ms=4.5, lw=1.6, color=col, label=name)
    dy = 6 if y[-1] < 5 else -1   # keep the near-zero label off the axis line
    ax.annotate("%.2f" % y[-1], (ks[-1], y[-1]), textcoords="offset points",
                xytext=(6, dy), fontsize=6.6, color=col, va="center")
ax.set_xscale("log", base=2); ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.set_xlim(0.8, 260)
ax.set_ylim(-3, 63)
ax.axhline(0, color="#999999", lw=0.8, ls=":")
ax.set_xlabel("attempts per problem $k$")
ax.set_ylabel("base $-$ SLJ-trained (percentage points)")
ax.set_title("Where the judge loses the signal")
ax.legend(frameon=False, loc="upper right", handlelength=1.6,
          borderpad=0.2, labelspacing=0.3)
ax.grid(alpha=0.2, lw=0.5); ax.set_axisbelow(True)

fig.subplots_adjust(wspace=0.40)
out = os.path.join(ROOT, "figures",
                   "evorobust_fig1_replay.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, metadata={"CreationDate": None})
png = out.replace(".pdf", ".png")
fig.savefig(png, dpi=200)
site_png = os.path.join(ROOT, "docs", "assets", "fig1_replay.png")
os.makedirs(os.path.dirname(site_png), exist_ok=True)
shutil.copyfile(png, site_png)
print("wrote", out)
print("wrote", png)
print("updated", site_png)
for name, key in (("judge","judge"), ("gate","gated"), ("exec","exec")):
    print("  %-6s separation (pp): %s" % (name, ["%.2f" % (100 * (b["%s@%d"%(key,k)]-t["%s@%d"%(key,k)])) for k in ks]))
