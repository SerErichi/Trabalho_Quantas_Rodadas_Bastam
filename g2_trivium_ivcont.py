"""
G2 (Trivium), desenho alternativo: IV em CONTADOR. Cada sequência usa UMA chave aleatória
e IVs consecutivos 0,1,2,...,4095 (contador nos 12 bits menos significativos do IV);
de cada par toma-se os 128 primeiros bits de keystream após N passos. IVs vizinhos
diferem em poucos bits, então este desenho expõe a difusão incompleta do IV em N pequeno
(o desenho com chave e IV totalmente aleatórios não expõe: combinações lineares de bits
aleatórios já parecem aleatórias).
"""
import csv, os, random, time
import numpy as np
from bitslice import executar, int_para_bits, pos_chave, pos_iv
from bonus_cubo import padrao
from nist_tests import rodar_bateria
from util import registrar_tempo
from g2_trivium import N_BITS, M_SEQ, JANELA, NS

D = 12


def main(semente=2026):
    t0 = time.time()
    rng = random.Random(semente)
    por_seq = 1 << D
    S = por_seq * M_SEQ
    mask = (1 << S) - 1
    s = [0] * 289
    for j in range(80):
        bits = np.array([rng.getrandbits(1) for _ in range(M_SEQ)], dtype=np.uint8)
        rep = np.repeat(bits, por_seq)
        s[pos_chave(j)] = int.from_bytes(np.packbits(rep, bitorder="little").tobytes(), "little")
    for j in range(D):
        s[pos_iv(j)] = int.from_bytes(padrao(j, D).to_bytes(por_seq // 8, "little") * M_SEQ, "little")
    s[286] = s[287] = s[288] = mask
    z = executar(s, max(NS) + JANELA)
    print("g2_trivium_ivcont: simulado", round(time.time() - t0, 1), "s", flush=True)
    linhas = []
    for N in NS:
        a = np.stack([int_para_bits(z[N + k], S) for k in range(JANELA)])
        a = a.reshape(JANELA, M_SEQ, por_seq).transpose(1, 2, 0).reshape(M_SEQ, -1)
        for sq, b in enumerate(a):
            for teste, p in rodar_bateria(b).items():
                linhas.append([N, sq, teste, p])
        print("g2_trivium_ivcont N", N, round(time.time() - t0, 1), "s", flush=True)
    with open("dados/g2_trivium_ivcont_pvalores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["N", "sequencia", "teste", "p_valor"])
        w.writerows(linhas)
    registrar_tempo("g2_trivium_ivcont", time.time() - t0)


if __name__ == "__main__":
    main()
