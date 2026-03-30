import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sympy import symbols, expand, diff

df = pd.read_csv("output.csv")

with open("points.csv") as f:
    lines = f.read().strip().splitlines()

n    = int(lines[0].split(",")[1])
xs   = np.array([float(l.split(",")[0]) for l in lines[1:n+1]])
ys   = np.array([float(l.split(",")[1]) for l in lines[1:n+1]])
ks   = [float(v) for v in lines[n+1].split(",")[1:]]

x_next = float(n)
x = symbols("x")

def lagrange_sym(xs, ys):
    poly = 0
    for i in range(len(xs)):
        basis = 1
        for j in range(len(xs)):
            if j != i:
                basis *= (x - xs[j]) / (xs[i] - xs[j])
        poly += ys[i] * basis
    return expand(poly)

def perturbation_sym(xs):
    p = 1
    for xi in xs:
        p *= (x - xi)
    return expand(p)

P      = lagrange_sym(xs, ys)
W      = perturbation_sym(xs)
Polys  = [expand(P + k_val * W) for k_val in ks]
Derivs = [diff(pk, x) for pk in Polys]

def poly_str(expr):
    p      = expr.as_poly(x)
    coeffs = [float(c) for c in p.all_coeffs()]
    degree = p.degree()
    terms  = []
    for i, c in enumerate(coeffs):
        c   = round(c, 4)
        deg = degree - i
        if abs(c) < 1e-6:
            continue
        sign       = "+" if c > 0 else "-"
        ac         = abs(c)
        coeff_str  = "" if (abs(ac - 1.0) < 1e-6 and deg > 0) else f"{ac:g}"
        if deg == 0:
            terms.append(f"{sign} {coeff_str}")
        elif deg == 1:
            terms.append(f"{sign} {coeff_str}x")
        else:
            terms.append(f"{sign} {coeff_str}x^{{{deg}}}")
    s = " ".join(terms).lstrip("+ ").replace("+ -", "- ")
    return f"$P(x) = {s}$"

rng       = np.random.default_rng(42)
k_samples = rng.uniform(-5, 5, 1024)

def eval_poly_np(xs, ys, k_val, x_val):
    p = 0.0
    for i in range(len(xs)):
        basis = 1.0
        for j in range(len(xs)):
            if j != i:
                basis *= (x_val - xs[j]) / (xs[i] - xs[j])
        p += ys[i] * basis
    w = 1.0
    for xi in xs:
        w *= (x_val - xi)
    return p + k_val * w

dist_values = np.array([eval_poly_np(xs, ys, k, x_next) for k in k_samples])
y_min_next  = dist_values.min()
y_max_next  = dist_values.max()
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, height_ratios=[2.2, 1], hspace=0.45, wspace=0.35)
ax_main = fig.add_subplot(gs[0, :])
ax_hist = fig.add_subplot(gs[1, 0])
ax_derv = fig.add_subplot(gs[1, 1])
colors   = ["crimson", "forestgreen", "darkorange"]
labels_k = ["k = −3", "k = 0  (Lagrange puro)", "k = +3"]
x_vals   = df["x"].values
formula_xpos = [xs[1]] * 3
formula_yoff = [90, -20, -90]
ax_main.fill_between(df["x"], df["y_min"], df["y_max"], alpha=0.22, color="steelblue", label="Spazio delle soluzioni")

for pk, color, lbl, xpos, yoff in zip(Polys, colors, labels_k, formula_xpos, formula_yoff):
    y_vals  = np.array([float(pk.subs(x, xi)) for xi in x_vals])
    formula = poly_str(pk)
    ax_main.plot(x_vals, y_vals, color=color, linewidth=1.8, label=lbl)
    # formula annotata direttamente sulla curva
    idx = np.argmin(np.abs(x_vals - xpos))
    ax_main.annotate(formula,
                     xy=(x_vals[idx], y_vals[idx]),
                     xytext=(0, yoff), textcoords="offset points",
                     fontsize=7, color=color,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white",
                               alpha=0.8, ec=color, lw=0.8))

ax_main.scatter(xs, ys, color="black", zorder=5, s=65, label="Punti noti")

# linea verticale a x=n
ax_main.axvline(x=x_next, color="purple", linestyle="--", linewidth=1.4,
                label=f"$x = {int(x_next)}$ (termine successivo)")

# freccia doppia sul range a x=n
ax_main.annotate("", xy=(x_next, y_max_next), xytext=(x_next, y_min_next),
                 arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5))

# annotazione range
ax_main.annotate(f"Risposte valide a $x={int(x_next)}$:\n"
                 f"  min = {y_min_next:.1f}\n  max = {y_max_next:.1f}",
                 xy=(x_next, (y_min_next + y_max_next) / 2),
                 xytext=(x_next + 0.15, (y_min_next + y_max_next) / 2),
                 fontsize=8.5, color="purple",
                 bbox=dict(boxstyle="round,pad=0.3", fc="lavender", alpha=0.8))

y_center = ys.mean()
y_range  = max(ys) - min(ys)
margin   = y_range * 3
ax_main.set_ylim(y_center - margin, y_center + margin)
ax_main.set_xlim(df["x"].min(), df["x"].max())
ax_main.set_title("Interpolazione di Lagrange — infinite soluzioni compatibili con la sequenza",
                  fontsize=12)
ax_main.set_xlabel("x")
ax_main.set_ylabel("y")
ax_main.legend(fontsize=8, loc="upper left", handlelength=1.5)
ax_main.grid(True, alpha=0.3)

# ── istogramma ────────────────────────────────────────────────────────────────
ax_hist.hist(dist_values, bins=40, color="steelblue", alpha=0.75, edgecolor="white")
ax_hist.set_title(f"Distribuzione di $P_k({int(x_next)})$ — 1024 soluzioni", fontsize=9)
ax_hist.set_xlabel(f"Valore a $x = {int(x_next)}$")
ax_hist.set_ylabel("Frequenza")
ax_hist.grid(True, alpha=0.3)

# ── derivate ──────────────────────────────────────────────────────────────────
for dk, color, lbl in zip(Derivs, colors, labels_k):
    dy_vals = np.array([float(dk.subs(x, xi)) for xi in x_vals])
    ax_derv.plot(x_vals, dy_vals, color=color, linewidth=1.6, label=lbl)

ax_derv.scatter(xs, np.zeros(len(xs)), color="black", zorder=5, s=25,
                label="Posizione punti noti")
ax_derv.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax_derv.set_title("Derivate — divergenza del trend tra i punti noti", fontsize=9)
ax_derv.set_xlabel("x")
ax_derv.set_ylabel("$P'(x)$")
ax_derv.legend(fontsize=7)
ax_derv.grid(True, alpha=0.3)

dy_ref  = np.array([float(Derivs[1].subs(x, xi)) for xi in x_vals])
d_range = max(abs(dy_ref)) * 4
ax_derv.set_ylim(-d_range, d_range)
ax_derv.set_xlim(df["x"].min(), df["x"].max())

plt.savefig("lagrange_plot.png", dpi=150)
print("Salvato lagrange_plot.png")
print(f"\nRisposte valide a x={int(x_next)}: [{y_min_next:.2f}, {y_max_next:.2f}]")