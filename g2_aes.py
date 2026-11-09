"""
G2 (AES): bateria NIST (subconjunto, ver nist_tests.py) sobre a saída do AES-R em
MODO CONTADOR: para cada sequência, uma chave aleatória e blocos de contador
0,1,2,... cifrados e concatenados (n bits por sequência, m sequências por R).
Inclui um controle com bits do os.urandom.
"""
import csv, os, random, time
import numpy as np
from cifras_reduzidas import aes_cifrar
from nist_tests import rodar_bateria
from util import registrar_tempo

N_BITS = 2 ** 19
M_SEQ = 100


def seq_aes(R, rng, n=N_BITS, mc=False):
    k = rng.randbytes(16)
    blocos = n // 128
    ct = b"".join(aes_cifrar(i.to_bytes(16, "big"), k, R, mc) for i in range(blocos))
    return np.unpackbits(np.frombuffer(ct, dtype=np.uint8))


def main(semente=2026):
    t0 = time.time()
    rng = random.Random(semente)
    linhas = []
    for R in list(range(1, 11)) + [0]:           # R=0 -> controle (os.urandom)
        for s in range(M_SEQ):
            if R == 0:
                b = np.unpackbits(np.frombuffer(os.urandom(N_BITS // 8), dtype=np.uint8))
            else:
                b = seq_aes(R, rng)
            for teste, p in rodar_bateria(b).items():
                linhas.append([R, s, teste, p])
        print("g2_aes R", R, round(time.time() - t0, 1), "s", flush=True)
    with open("dados/g2_aes_pvalores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["R(0=controle)", "sequencia", "teste", "p_valor"])
        w.writerows(linhas)
    registrar_tempo("g2_aes", time.time() - t0)


if __name__ == "__main__":
    main()
