"""
G2 (Trivium): para cada N, a sequência testada é formada por M_SEQ x (n/128) pares
(chave, IV) ALEATÓRIOS DIFERENTES, tomando de cada um os 128 primeiros bits de
keystream após N passos de pré-aquecimento (concatenados). Um keystream longo
de um único par (chave, IV) não mede o efeito de N: depois do aquecimento o estado
já está misturado, e o início do keystream (onde N importa) seria só uma fração
minúscula da sequência.
"""
import csv, os, random, time
import numpy as np
from bitslice import estado_aleatorio, executar, int_para_bits
from nist_tests import rodar_bateria
from util import registrar_tempo

N_BITS = 2 ** 19
M_SEQ = 100
JANELA = 128
NS = [0, 16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 640, 768, 1152]


def main(semente=2026):
    t0 = time.time()
    rng = random.Random(semente)
    por_seq = N_BITS // JANELA                  # pares (chave, IV) por sequência
    S = por_seq * M_SEQ
    z = executar(estado_aleatorio(S, rng), max(NS) + JANELA)
    print("g2_trivium: estados simulados", round(time.time() - t0, 1), "s", flush=True)
    linhas = []
    for N in NS + [-1]:                         # N=-1 -> controle (os.urandom)
        if N == -1:
            seqs = [np.unpackbits(np.frombuffer(os.urandom(N_BITS // 8), dtype=np.uint8)) for _ in range(M_SEQ)]
        else:
            a = np.stack([int_para_bits(z[N + k], S) for k in range(JANELA)])      # (128, S)
            a = a.reshape(JANELA, M_SEQ, por_seq).transpose(1, 2, 0).reshape(M_SEQ, -1)
            seqs = list(a)
        for s, b in enumerate(seqs):
            for teste, p in rodar_bateria(b).items():
                linhas.append([N, s, teste, p])
        print("g2_trivium N", N, round(time.time() - t0, 1), "s", flush=True)
    with open("dados/g2_trivium_pvalores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["N(-1=controle)", "sequencia", "teste", "p_valor"])
        w.writerows(linhas)
    registrar_tempo("g2_trivium", time.time() - t0)


if __name__ == "__main__":
    main()
