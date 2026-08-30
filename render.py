"""render.py -- LaTeX rendering of the Section 5 tables, separated from the computation
so the tables can be rebuilt from summary.json without rerunning the experiments.

Column budget: the single-column tables must fit ~3.3 in.  Numbers are therefore given to
two significant figures, constants that do not vary down a column are moved to the caption,
and machine-precision entries are marked with a short symbol rather than a phrase.
"""
import json, os
import numpy as np

MACH = r"$\epsmach$"           # was $O(\epsmach)$, which is ~2x wider


def f(v, sig=2):
    """`sig` significant figures; plain decimal for moderate values, scientific outside."""
    if v is None:
        return "---"
    v = float(v)
    if abs(v) < 1e-13:
        return MACH
    e = int(np.floor(np.log10(abs(v))))
    if round(abs(v)/10**e, sig-1) >= 10.0:      # rounding carried into the next decade
        e += 1
    if -3 < e < 3:
        return f"${v:.{max(0, sig-1-e)}f}$"
    return f"${v/10**e:.{sig-1}f}\\times10^{{{e}}}$"


def write(name, body):
    os.makedirs("tables", exist_ok=True)
    open(f"tables/{name}.tex", "w").write(body + "\n")
    print(f"\n===== tables/{name}.tex =====\n{body}")


def tab(cols, header, rows, name):
    b = [r"\begin{tabular}{" + cols + "}", r"\toprule", header + r" \\", r"\midrule"]
    b += [r for r in rows]
    b += [r"\bottomrule", r"\end{tabular}"]
    write(name, "\n".join(b))


# ------------------------------------------------------------------ tables
def degree_accuracy(S):
    rows = [f"${r['d']}$ & ${r['mult']:.2f}$ & ${r['gamma']:.2f}$ & "
            f"${r['Q']:.2f}$ & {f(r['eps'],3)} & {f(r['t'],3)} & "
            f"{f(r['tbar'],3)} \\\\" for r in S["degree"]["rows"]]
    tab("rrrrrrr", r"$d$ & $d/d_{\rm ref}$ & $\gamma$ & $\mathcal{Q}/\mathcal{Q}_0$ "
        r"& $\varepsilon_0(d)$ & $\varepsilon_K(d)$ & $\bar\varepsilon$",
        rows, "degree_accuracy")


def isoguarantee(S):
    lbl = {"1D": "1D", "Weyl3": "Weyl 3", "2D": "2D"}
    rows = [f"{lbl[r['tag']]} & ${r['K_eff']}$ & ${r['d']}$ & ${r['mult']:.2f}$ & "
            f"${r['gamma']:.2f}$ & ${r['mult']*r['gamma']:.2f}$ & {f(r['tbar'],3)} \\\\"
            for r in S["iso"]]
    tab("lrrrrrr",
        r"spectrum & $K_{\rm eff}$ & $d$ & $d/d_{\rm ref}$ & $\gamma$ "
        r"& $\mathcal{Q}/\mathcal{Q}_0$ & $\bar\varepsilon$",
        rows, "isoguarantee")


def qsvt_1d(S):
    order = [r"$p_{SC}$, iso-guarantee", r"$p_0$, same guarantee", r"$p_0$, same $e_x$"]
    rows = []
    for load, title in (("uniform", "Uniform load"), ("point", "Point load at midpoint")):
        rows += [r"\midrule", r"\multicolumn{7}{l}{\textit{" + title + r"}} \\",
                 r"\midrule"]
        for tag in order:
            m = S["qsvt_1d"]["rows"][f"{load}|{tag}"]
            rows.append(f"{tag} & ${m['d']}$ & ${m['F']:.6f}$ & {f(m['eC'],3)} & "
                        f"{f(m['ex'],3)} & ${m['P']:.3f}$ & ${m['Q']:.0f}$ \\\\")
    b = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Method & $d$ & $F$ & $e_C$ & $e_x$ & $P_{\rm succ}$ & $\mathcal{Q}$ \\"]
    b += rows + [r"\bottomrule", r"\end{tabular}"]
    write("qsvt_1d", "\n".join(b))


def perturbation(S):
    rows = [f"${r['eta']:g}$ & {f(r['tbar'],3)} & {f(r['eC'])} & "
            f"{f(r['ex'])} & ${r['P']:.3f}$ \\\\" for r in S["perturb"]["rows"]]
    tab("rrrrr",
        r"$\eta$ & $\bar\varepsilon$ & $e_C$ & $e_x$ & $P_{\rm succ}$",
        rows, "perturbation")


def qsvt_2d(S):
    d_ref = S["qsvt_2d"]["d_ref"]
    tau0 = [r for r in S["qsvt_2d"]["rows"] if r["K"] == 0][0]["tau"]
    rows = []
    for r in S["qsvt_2d"]["rows"]:
        if r["K"] == 0:
            rows.append(f"0 (base) & --- & {d_ref} & 1.00 & 1.00 & 1.00 & "
                        f"{f(S['qsvt_2d']['eps_ref'],3)} & ${r['F']:.6f}$ & "
                        f"{f(r['eC'])} & {f(r['ex'])} & ${r['P']:.3f}$ \\\\")
        else:
            g = r["tau"]/tau0
            rows.append(f"${r['K']}$ & ${r['K_eff']}$ & ${r['d']}$ & ${r['mult']:.2f}$ & "
                        f"${g:.2f}$ & ${r['mult']*g:.2f}$ & {f(r['tbar'],3)} & "
                        f"${r['F']:.6f}$ & {f(r['eC'])} & {f(r['ex'])} & "
                        f"${r['P']:.3f}$ \\\\")
    tab("rrrrrrrrrrr",
        r"$K$ & $K_{\rm eff}$ & $d$ & $d/d_{\rm ref}$ & $\gamma$ "
        r"& $\mathcal{Q}/\mathcal{Q}_0$ & $\bar\varepsilon$ & $F$ & $e_C$ & $e_x$ "
        r"& $P_{\rm succ}$", rows, "qsvt_2d")


def multiplier(S):
    eps_ref = S["multiplier"]["eps_ref"]
    rows, prev = [], None
    for r in S["multiplier"]["rows"]:
        tag = {"minmax": "LP", "minnorm": "min-norm"}[r["method"]]
        if prev is not None and r["method"] != prev:
            rows.append(r"\midrule")
        lead = tag if r["method"] != prev else ""
        dag = r"$^{\dagger}$" if r["tbar"] <= eps_ref else ""
        rows.append(f"{lead} & ${r['mult']:.2f}${dag} & ${r['gamma']:.2f}$ & "
                    f"{f(r['tbar'],3)} & ${r['g_uniform']:.0f}$ & "
                    f"${r['g_point']:.1f}$ & ${r['g_random']:.1f}$ \\\\")
        prev = r["method"]
    tab("lrrrrrr",
        r"method & $d/d_{\rm ref}$ & $\gamma$ & $\bar\varepsilon$ & unif. & point "
        r"& rand.", rows, "multiplier")


def all_tables(S):
    degree_accuracy(S); isoguarantee(S); qsvt_1d(S)
    perturbation(S); qsvt_2d(S); multiplier(S)


if __name__ == "__main__":
    all_tables(json.load(open("summary.json")))
