"""
G1 (AES): avalanche. Inverte 1 bit do texto claro (ou da chave) e mede quantos
bits (e bytes) do bloco cifrado mudam, para R = 1..10.
Também mede o SAC: P[i][j] = prob. de o bit de saída j mudar quando o bit de
entrada i é invertido (matriz 128x128).
Uso: python3 g1_aes.py            (variante padrão, R=1..10)
     import g1_aes; g1_aes.executar(mc=True, Rs=range(1,7), tag='mc')
"""
import csv, random, sys, time
import numpy as np
from cifras_reduzidas import aes_cifrar, inverter_bit, hamming
from util import registrar_tempo

N_AMOSTRAS = 20000      # pares por (R, tipo)
SAC_POR_BIT = 100       # amostras por bit de entrada na matriz SAC


def avalanche(R, tipo, n, rng, mc=False):
    hist = np.zeros(129, dtype=np.int64)
    hbytes = np.zeros(17, dtype=np.int64)
    for _ in range(n):
        p, k, j = rng.randbytes(16), rng.randbytes(16), rng.randrange(128)
        c1 = aes_cifrar(p, k, R, mc)
        if tipo == "texto":
            c2 = aes_cifrar(inverter_bit(p, j), k, R, mc)
        else:
            c2 = aes_cifrar(p, inverter_bit(k, j), R, mc)
        hist[hamming(c1, c2)] += 1
        hbytes[sum(a != b for a, b in zip(c1, c2))] += 1
    return hist, hbytes


def sac(R, tipo, por_bit, rng, mc=False):
    P = np.zeros((128, 128))
    for i in range(128):
        acc = np.zeros(128, dtype=np.int64)
        for _ in range(por_bit):
            p, k = rng.randbytes(16), rng.randbytes(16)
            c1 = aes_cifrar(p, k, R, mc)
            if tipo == "texto":
                c2 = aes_cifrar(inverter_bit(p, i), k, R, mc)
            else:
                c2 = aes_cifrar(p, inverter_bit(k, i), R, mc)
            d = bytes(a ^ b for a, b in zip(c1, c2))
            acc += np.unpackbits(np.frombuffer(d, dtype=np.uint8), bitorder="little")
        P[i] = acc / por_bit
    return P


def executar(mc=False, Rs=range(1, 11), tag="padrao", n=N_AMOSTRAS, por_bit=SAC_POR_BIT, semente=2026):
    t0 = time.time()
    rng = random.Random(semente)
    resumo, hist_rows, bytes_rows, sac_rows = [], [], [], []
    for R in Rs:
        for tipo in ("texto", "chave"):
            h, hb = avalanche(R, tipo, n, rng, mc)
            x = np.arange(129)
            media = (h * x).sum() / n
            dp = np.sqrt((h * (x - media) ** 2).sum() / n)
            xb = np.arange(17)
            mb = (hb * xb).sum() / n
            resumo.append([tag, R, tipo, n, round(media, 4), round(dp, 4), round(mb, 4)])
            hist_rows += [[tag, R, tipo, d, int(c)] for d, c in enumerate(h) if c]
            bytes_rows += [[tag, R, tipo, d, int(c)] for d, c in enumerate(hb) if c]
            P = sac(R, tipo, por_bit, rng, mc)
            dev = np.abs(P - 0.5)
            lim = 3 * 0.5 / np.sqrt(por_bit)
            sac_rows.append([tag, R, tipo, por_bit, round(float(P.mean()), 4),
                             round(float((dev > lim).mean()), 4),
                             round(float(((P == 0) | (P == 1)).mean()), 4)])
            if tag == "padrao" and R in (1, 2, 3, 4) or tag == "mc" and R in (1, 2, 3, 4):
                np.savetxt(f"dados/sac_matriz_{tag}_R{R}_{tipo}.csv", P, fmt="%.2f", delimiter=",")
            print(tag, "R", R, tipo, "media", round(media, 3), "dp", round(dp, 3), flush=True)
    def gravar(nome, cab, rows):
        with open(f"dados/{nome}_{tag}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(cab); w.writerows(rows)
    gravar("g1_aes_resumo", ["variante", "R", "tipo", "n", "media_hd", "desvio_hd", "media_bytes"], resumo)
    gravar("g1_aes_hist_bits", ["variante", "R", "tipo", "hd", "contagem"], hist_rows)
    gravar("g1_aes_hist_bytes", ["variante", "R", "tipo", "bytes_alterados", "contagem"], bytes_rows)
    gravar("g1_aes_sac", ["variante", "R", "tipo", "amostras_por_bit", "media_P", "frac_fora_3sigma", "frac_0_ou_1"], sac_rows)
    registrar_tempo(f"g1_aes_{tag}", time.time() - t0)


if __name__ == "__main__":
    executar()
