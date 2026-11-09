"""
G1 (Trivium): avalanche em função de N (passos de pré-aquecimento), N = 0..1152.
Inverte 1 bit do IV (ou da chave) e conta quantos dos 128 primeiros bits de
keystream (após N passos) mudam. Todos os 80 bits de chave e de IV são testados,
cada um com M pares (chave, IV) aleatórios, em bitslice.
"""
import csv, random, time
import numpy as np
from bitslice import estado_aleatorio, executar, pares_de_bits, pos_chave, pos_iv
from util import registrar_tempo

M = 2048                  # amostras por posição de bit invertido (múltiplo de 8)
NBITS = 128
NMAX = 1152
N_HIST = [0, 50, 100, 150, 200, 300, 400, 600, 800, 1152]


def main(semente=2026):
    t0 = time.time()
    rng = random.Random(semente)
    mask = (1 << M) - 1
    Ns = np.arange(NMAX + 1)
    soma = {t: np.zeros(NMAX + 1) for t in ("iv", "chave")}
    soma2 = {t: np.zeros(NMAX + 1) for t in ("iv", "chave")}
    cont = {t: 0 for t in ("iv", "chave")}
    hist = {(t, N): np.zeros(NBITS + 1, dtype=np.int64) for t in ("iv", "chave") for N in N_HIST}
    hbyt = {(t, N): np.zeros(17, dtype=np.int64) for t in ("iv", "chave") for N in N_HIST}
    for tipo, posf in (("iv", pos_iv), ("chave", pos_chave)):
        for j in range(80):
            s0 = estado_aleatorio(M, rng)
            s1 = list(s0)
            s1[posf(j)] ^= mask
            z0 = executar(s0, NMAX + NBITS)
            z1 = executar(s1, NMAX + NBITS)
            D = pares_de_bits([a ^ b for a, b in zip(z0, z1)], M).astype(np.int32)   # (T, M)
            cs = np.vstack([np.zeros((1, M), dtype=np.int32), np.cumsum(D, axis=0)])
            HD = cs[NBITS:NBITS + NMAX + 1] - cs[:NMAX + 1]                           # (1153, M)
            soma[tipo] += HD.sum(axis=1)
            soma2[tipo] += (HD.astype(np.int64) ** 2).sum(axis=1)
            cont[tipo] += M
            for N in N_HIST:
                hist[(tipo, N)] += np.bincount(HD[N], minlength=NBITS + 1)
                janela = D[N:N + NBITS].reshape(16, 8, M).any(axis=1).sum(axis=0)
                hbyt[(tipo, N)] += np.bincount(janela, minlength=17)
        print("g1_trivium", tipo, "ok", round(time.time() - t0, 1), "s", flush=True)
    with open("dados/g1_trivium_curva.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tipo", "N", "n", "media_hd", "desvio_hd"])
        for t in ("iv", "chave"):
            n = cont[t]
            media = soma[t] / n
            dp = np.sqrt(soma2[t] / n - media ** 2)
            for N in Ns:
                w.writerow([t, int(N), n, round(float(media[N]), 4), round(float(dp[N]), 4)])
    with open("dados/g1_trivium_hist.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tipo", "N", "hd", "contagem"])
        for (t, N), h in hist.items():
            for d, c in enumerate(h):
                if c:
                    w.writerow([t, N, d, int(c)])
    with open("dados/g1_trivium_hist_bytes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tipo", "N", "bytes_alterados", "contagem"])
        for (t, N), h in hbyt.items():
            for d, c in enumerate(h):
                if c:
                    w.writerow([t, N, d, int(c)])
    registrar_tempo("g1_trivium", time.time() - t0)


if __name__ == "__main__":
    main()
