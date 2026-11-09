"""
graficos.py: gera os gráficos (graficos/*.png) e o resumo das conclusões
(dados/resumo_conclusoes.csv) a partir dos CSVs em dados/.
Faixa de referência em todos os gráficos: valor esperado para dados aleatórios ± 3 sigma.
"""
import math, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.special import gammaincc

plt.rcParams.update({"font.size": 9, "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})
SIG_HD = math.sqrt(128) / 2          # 5.657 bits: desvio de Binomial(128, 1/2)
SIG_BYTES = math.sqrt(16 * (255 / 256) * (1 / 256))
conc = []                              # linhas do resumo


def faixa(ax, centro, meia, x0, x1, rotulo="aleatório ±3σ"):
    ax.axhspan(centro - meia, centro + meia, color="tab:green", alpha=0.18, label=rotulo)
    ax.axhline(centro, color="tab:green", lw=0.8)


def menor_a_partir_de(xs, ok):
    """Menor x tal que ok vale para ele e para todos os maiores."""
    xs, ok = list(xs), list(ok)
    for i in range(len(xs)):
        if all(ok[i:]):
            return xs[i]
    return None


# ------------------------------------------------------------------ G1 AES
def g1_aes():
    r = pd.read_csv("dados/g1_aes_resumo_padrao.csv")
    n = r["n"].iloc[0]
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    for tipo, cor in (("texto", "tab:blue"), ("chave", "tab:red")):
        d = r[r.tipo == tipo]
        ax[0, 0].plot(d.R, d.media_hd, "o-", color=cor, label=tipo)
        ax[0, 1].plot(d.R, d.media_hd - 64, "o-", color=cor, label=tipo)
        ax[1, 0].plot(d.R, d.desvio_hd, "o-", color=cor, label=tipo)
        ax[1, 1].plot(d.R, d.media_bytes, "o-", color=cor, label=tipo)
    ax[0, 0].axhline(64, color="tab:green"); ax[0, 0].set_ylabel("média de bits alterados (de 128)")
    ax[0, 0].set_title("(a) avalanche: média")
    faixa(ax[0, 1], 0, 3 * SIG_HD / math.sqrt(n), 1, 10); ax[0, 1].set_ylim(-1, 1)
    ax[0, 1].set_title("(b) média − 64 (zoom, R≥3)")
    faixa(ax[1, 0], SIG_HD, 3 * SIG_HD / math.sqrt(2 * n), 1, 10)
    ax[1, 0].set_title("(c) desvio padrão (esperado 5,657)"); ax[1, 0].set_ylabel("desvio padrão (bits)")
    faixa(ax[1, 1], 16 * 255 / 256, 3 * SIG_BYTES / math.sqrt(n), 1, 10)
    ax[1, 1].set_title("(d) bytes alterados (de 16)")
    for a in ax.ravel():
        a.set_xlabel("R (rodadas)"); a.set_xticks(range(1, 11))
    ax[0, 0].legend(); ax[0, 1].legend()
    fig.tight_layout(); fig.savefig("graficos/g1_aes_avalanche.png"); plt.close(fig)

    h = pd.read_csv("dados/g1_aes_hist_bits_padrao.csv")
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    x = np.arange(129)
    for a, R in zip(ax, (1, 2, 3, 10)):
        d = h[(h.R == R) & (h.tipo == "texto")]
        a.bar(d.hd, d.contagem / n, width=1, color="tab:blue", alpha=0.7, label="medido")
        a.plot(x, binom.pmf(x, 128, 0.5), "k-", lw=1, label="Binomial(128; 0,5)")
        a.set_title(f"R={R}"); a.set_xlabel("bits alterados"); a.set_xlim(0, 128)
    ax[0].set_ylabel("frequência"); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig("graficos/g1_aes_histogramas.png"); plt.close(fig)

    for tipo in ("texto", "chave"):
        d = r[r.tipo == tipo]
        ok = [(abs(m - 64) < 3 * SIG_HD / math.sqrt(n)) and (abs(s - SIG_HD) < 3 * SIG_HD / math.sqrt(2 * n))
              for m, s in zip(d.media_hd, d.desvio_hd)]
        conc.append(["G1 AES avalanche (média e desvio)", f"flip no {tipo}", "menor R indistinguível (±3σ)",
                     menor_a_partir_de(d.R, ok)])
    s = pd.read_csv("dados/g1_aes_sac_padrao.csv")
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for tipo, cor in (("texto", "tab:blue"), ("chave", "tab:red")):
        d = s[s.tipo == tipo]
        ax.semilogy(d.R, d.frac_fora_3sigma.clip(lower=1e-5), "o-", color=cor, label=tipo)
    ax.axhline(0.0027, color="tab:green", label="aleatório (~0,27%)")
    ax.set_xlabel("R"); ax.set_ylabel("fração de células SAC fora de ±3σ"); ax.legend(); ax.set_xticks(range(1, 11))
    fig.tight_layout(); fig.savefig("graficos/g1_aes_sac.png"); plt.close(fig)
    for tipo in ("texto", "chave"):
        d = s[s.tipo == tipo]
        conc.append(["G1 AES SAC", f"flip no {tipo}", "menor R com fração fora de 3σ ≤ 1% (aleatório ≈ 0,27%)",
                     menor_a_partir_de(d.R, list(d.frac_fora_3sigma <= 0.01))])


# ------------------------------------------------------------------ G1 Trivium
def g1_trivium():
    c = pd.read_csv("dados/g1_trivium_curva.csv")
    n = c["n"].iloc[0]
    sm = SIG_HD / math.sqrt(n)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for tipo, cor in (("iv", "tab:blue"), ("chave", "tab:red")):
        d = c[c.tipo == tipo]
        z = (d.media_hd - 64) / sm
        ax[0].plot(d.N, d.media_hd, color=cor, label=tipo)
        ax[1].plot(d.N, z, color=cor, lw=0.8, label=tipo)
        ax[2].plot(d.N, d.desvio_hd, color=cor, lw=0.8, label=tipo)
        for lim in (3, 4):
            ult = d.N[(z.abs() > lim).values].max()
            conc.append(["G1 Trivium avalanche (média)", f"flip no {tipo}", f"último N com |z|>{lim}", int(ult)])
        sdev = (d.desvio_hd - SIG_HD) / (SIG_HD / math.sqrt(2 * n))
        conc.append(["G1 Trivium avalanche (desvio)", f"flip no {tipo}", "último N com |z|>4", int(d.N[(sdev.abs() > 4).values].max())])
    ax[0].axhline(64, color="tab:green"); ax[0].set_ylabel("média de bits alterados (de 128)")
    ax[0].set_title("(a) média"); ax[0].legend()
    ax[1].axhspan(-3, 3, color="tab:green", alpha=0.18); ax[1].set_ylabel("z = (média − 64)/σ_média")
    ax[1].set_yscale("symlog", linthresh=10); ax[1].set_title("(b) z-score da média (verde: ±3σ)")
    faixa(ax[2], SIG_HD, 3 * SIG_HD / math.sqrt(2 * n), 0, 1152); ax[2].set_title("(c) desvio padrão")
    for a in ax:
        a.set_xlabel("N (passos de pré-aquecimento)"); a.set_xscale("symlog", linthresh=16); a.set_xlim(0, 1152)
    fig.tight_layout(); fig.savefig("graficos/g1_trivium_avalanche.png"); plt.close(fig)

    h = pd.read_csv("dados/g1_trivium_hist.csv")
    Ns = [0, 100, 200, 300, 1152]
    fig, ax = plt.subplots(1, len(Ns), figsize=(11, 2.6), sharey=True)
    x = np.arange(129)
    for a, N in zip(ax, Ns):
        d = h[(h.N == N) & (h.tipo == "iv")]
        a.bar(d.hd, d.contagem / d.contagem.sum(), width=1, color="tab:blue", alpha=0.7)
        a.plot(x, binom.pmf(x, 128, 0.5), "k-", lw=1)
        a.set_title(f"N={N}"); a.set_xlim(0, 128); a.set_xlabel("bits alterados")
    ax[0].set_ylabel("frequência (flip no IV)")
    fig.tight_layout(); fig.savefig("graficos/g1_trivium_histogramas.png"); plt.close(fig)


# ------------------------------------------------------------------ G2
def g2(cifra, arq, rot_param, ctrl, nome_param, band_m=100):
    p = pd.read_csv(arq)
    p.columns = ["param", "seq", "teste", "p"]
    p["ok"] = p.p >= 0.01
    prop = p.groupby(["param", "teste"]).ok.mean().unstack("teste")
    m = p.seq.nunique()
    lim_inf = 0.99 - 3 * math.sqrt(0.99 * 0.01 / m)
    # uniformidade dos p-valores (NIST: chi2 em 10 bins; aprovado se P-valor_T >= 0.0001)
    unif = {}
    for (par, t), g in p.groupby(["param", "teste"]):
        cont = np.histogram(g.p, bins=10, range=(0, 1))[0]
        chi = ((cont - m / 10) ** 2 / (m / 10)).sum()
        unif[(par, t)] = float(gammaincc(4.5, chi / 2))
    unif = pd.Series(unif).unstack()
    linhas = []
    for par in prop.index:
        for t in prop.columns:
            linhas.append([par, t, m, round(prop.loc[par, t], 4), round(unif.loc[par, t], 6)])
    pd.DataFrame(linhas, columns=[nome_param, "teste", "sequencias", "proporcao_aprovada", "p_uniformidade"]).to_csv(
        f"dados/g2_{cifra}_resumo.csv", index=False)

    params = [x for x in prop.index if x != ctrl]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for t in prop.columns:
        ax[0].plot(params, prop.loc[params, t], "o-", ms=3, lw=0.9, label=t)
    ax[0].axhspan(lim_inf, 1.0, color="tab:green", alpha=0.18, label=f"aleatório (α=0,01; m={m})")
    ax[0].set_ylabel("proporção de sequências aprovadas"); ax[0].set_xlabel(rot_param)
    ax[0].set_title("(a) proporção aprovada por teste"); ax[0].legend(fontsize=6, ncol=2, loc="center left")
    nfalha = (1 - prop.loc[params]).mean(axis=1) * len(prop.columns)
    ax[1].plot(params, nfalha, "ko-", ms=3)
    nt = len(prop.columns)
    taxa = np.array([0.0199 if t == "serial" else 0.01 for t in prop.columns])   # serial = min(p1,p2)
    esp = float(taxa.sum())
    dp = math.sqrt(m * float((taxa * (1 - taxa)).sum())) / m
    ax[1].axhspan(max(esp - 3 * dp, 0), esp + 3 * dp, color="tab:green", alpha=0.18, label="aleatório (0,01×nº testes ±3σ)")
    ax[1].axhline(esp, color="tab:green", lw=0.8)
    if ctrl is not None:
        ax[1].axhline((1 - prop.loc[ctrl]).sum(), color="k", ls=":", label="controle (os.urandom)")
    ax[1].set_ylabel("nº médio de testes reprovados por sequência"); ax[1].set_xlabel(rot_param)
    ax[1].set_title("(b) testes reprovados por sequência"); ax[1].legend(fontsize=7)
    if cifra.startswith("trivium"):
        for a in ax:
            a.set_xscale("symlog", linthresh=16); a.set_xlim(0, 1152)
    else:
        for a in ax:
            a.set_xticks(params)
    fig.tight_layout(); fig.savefig(f"graficos/g2_{cifra}.png"); plt.close(fig)

    # critério agregado: total de reprovações (teste x sequência) dentro de 3σ do esperado (0,01 por teste)
    falhas = {x: (1 - prop.loc[x]).sum() * m for x in params}
    lim = m * esp + 4 * m * dp   # 4σ: margem p/ correlação entre testes (cusum, monobit, runs)
    ok_prop = [bool(falhas[x] <= lim) for x in params]
    ok_unif = [bool((unif.loc[x].drop("serial") >= 0.0001).all()) for x in params]   # serial: min de 2 p-valores, não uniforme sob H0
    pior = {x: float(prop.loc[x].min()) for x in params}
    conc.append([f"G2 {cifra} (NIST subconjunto)", "reprovações totais", f"menor {nome_param} com nº de reprovações ≤ esperado+4σ (≤{lim:.0f} de {m*nt})",
                 menor_a_partir_de(params, ok_prop)])
    conc.append([f"G2 {cifra} (NIST subconjunto)", "uniformidade dos p-valores", f"menor {nome_param} com uniformidade ≥ 0,0001 em todos os testes (exceto serial)",
                 menor_a_partir_de(params, ok_unif)])
    conc.append([f"G2 {cifra} (NIST subconjunto)", "ambos os critérios", f"menor {nome_param}",
                 menor_a_partir_de(params, [a and b for a, b in zip(ok_prop, ok_unif)])])
    conc.append([f"G2 {cifra} (NIST subconjunto)", f"{nome_param} = {params[0]} (menor testado)", "nº de reprovações (esperado ≈ %.0f)" % (m * esp),
                 int(round(falhas[params[0]]))])


# ------------------------------------------------------------------ item 3
def item3():
    a = pd.read_csv("dados/g1_aes_resumo_padrao.csv")
    b = pd.read_csv("dados/g1_aes_resumo_mc.csv")
    n = a["n"].iloc[0]
    s1 = pd.read_csv("dados/g1_aes_sac_padrao.csv")
    s2 = pd.read_csv("dados/g1_aes_sac_mc.csv")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for tipo, mk in (("texto", "o"), ("chave", "s")):
        for df, nome, ls in ((a, "padrão", "-"), (b, "MC na última", "--")):
            d = df[(df.tipo == tipo) & (df.R <= 6)]
            ax[0].plot(d.R, d.media_hd, mk + ls, label=f"{tipo}, {nome}")
        for df, nome, ls in ((s1, "padrão", "-"), (s2, "MC na última", "--")):
            d = df[(df.tipo == tipo) & (df.R <= 6)]
            ax[1].semilogy(d.R, d.frac_fora_3sigma.clip(lower=1e-5), mk + ls, label=f"{tipo}, {nome}")
        for df, nome, ls in ((a, "padrão", "-"), (b, "MC na última", "--")):
            d = df[(df.tipo == tipo) & (df.R <= 6)]
            ax[2].plot(d.R, d.media_bytes, mk + ls, label=f"{tipo}, {nome}")
    ax[0].axhline(64, color="tab:green"); ax[0].set_ylabel("média de bits alterados"); ax[0].legend(fontsize=6)
    ax[0].set_title("(a) avalanche (média)")
    ax[1].axhline(0.0027, color="tab:green"); ax[1].set_ylabel("fração SAC fora de ±3σ"); ax[1].set_title("(b) SAC")
    ax[2].axhline(16 * 255 / 256, color="tab:green"); ax[2].set_ylabel("bytes alterados (de 16)"); ax[2].set_title("(c) bytes")
    for x in ax:
        x.set_xlabel("R"); x.set_xticks(range(1, 7))
    fig.tight_layout(); fig.savefig("graficos/item3_mixcolumns.png"); plt.close(fig)
    for tipo in ("texto", "chave"):
        for df, nome in ((a, "padrão"), (b, "MC na última")):
            d = df[(df.tipo == tipo) & (df.R <= 6)]
            ok = [(abs(m - 64) < 3 * SIG_HD / math.sqrt(n)) and (abs(s - SIG_HD) < 3 * SIG_HD / math.sqrt(2 * n))
                  for m, s in zip(d.media_hd, d.desvio_hd)]
            conc.append(["Item 3 avalanche", f"{tipo}, {nome}", "menor R indistinguível (±3σ)", menor_a_partir_de(d.R, ok)])


# ------------------------------------------------------------------ bônus
def bonus():
    d = pd.read_csv("dados/bonus_integral_aes.csv")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for v, mk in (("padrao", "o-"), ("com_MC_final", "s--")):
        x = d[d.variante == v]
        ax.semilogy(x.R, x.frac_bytes_balanceados.clip(lower=1e-4), mk, label=v)
    ax.axhline(1 / 256, color="tab:green", label="aleatório: 1/256")
    ax.set_xlabel("R"); ax.set_ylabel("fração de bytes balanceados"); ax.set_xticks(range(1, 11)); ax.legend()
    fig.tight_layout(); fig.savefig("graficos/bonus_integral_aes.png"); plt.close(fig)
    for v in ("padrao", "com_MC_final"):
        x = d[d.variante == v]
        ult = x.R[(x.frac_bytes_balanceados > 0.05).values].max()
        conc.append(["Bônus: integral (Square) no AES", v, "último R em que o teste enxerga estrutura", int(ult)])

    c = pd.read_csv("dados/bonus_cubo_trivium.csv")
    K = c.chaves.iloc[0]
    sK = 0.5 / math.sqrt(K)
    dims = sorted(c.dim_cubo.unique())
    fig, ax = plt.subplots(1, len(dims), figsize=(11, 3), sharey=True)
    for a, dm in zip(ax, dims):
        for cb in sorted(c[c.dim_cubo == dm].cubo.unique()):
            x = c[(c.dim_cubo == dm) & (c.cubo == cb)]
            a.plot(x.N, x.frac_superpoly_1, lw=0.7)
            ult = x.N[((x.frac_superpoly_1 - 0.5).abs() > 4 * sK).values]
            conc.append(["Bônus: cube tester no Trivium", f"cubo de dimensão {dm} (#{cb})", "último N com |p̂−0,5|>4σ",
                         int(ult.max()) if len(ult) else None])
        a.axhspan(0.5 - 3 * sK, 0.5 + 3 * sK, color="tab:green", alpha=0.18)
        a.set_title(f"cubo com d={dm} bits do IV"); a.set_xlabel("N"); a.set_xscale("symlog", linthresh=16); a.set_xlim(0, 1152)
    ax[0].set_ylabel(f"fração de chaves com superpoly = 1 (K={K})")
    fig.tight_layout(); fig.savefig("graficos/bonus_cubo_trivium.png"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs("graficos", exist_ok=True)
    for f in (g1_aes, g1_trivium, item3, bonus):
        try:
            f()
        except FileNotFoundError as e:
            print("pulando", f.__name__, e)
    for args in (("aes", "dados/g2_aes_pvalores.csv", "R (rodadas)", 0, "R"),
                 ("trivium", "dados/g2_trivium_pvalores.csv", "N (passos de pré-aquecimento)", -1, "N"),
                 ("trivium_ivcont", "dados/g2_trivium_ivcont_pvalores.csv", "N (passos de pré-aquecimento)", None, "N")):
        try:
            g2(*args)
        except FileNotFoundError as e:
            print("pulando g2", args[0], e)
    pd.DataFrame(conc, columns=["experimento", "caso", "criterio", "resultado"]).to_csv(
        "dados/resumo_conclusoes.csv", index=False)
    print(pd.DataFrame(conc, columns=["experimento", "caso", "criterio", "resultado"]).to_string())
