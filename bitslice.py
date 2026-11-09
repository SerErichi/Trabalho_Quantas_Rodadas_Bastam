"""
bitslice.py: Trivium "bitsliced" em Python puro.

Em vez de simular UM par (chave, IV) por vez, cada bit do estado s[i] é um
inteiro grande de M bits, onde o bit m do inteiro é o valor de s[i] na amostra m.
As operações XOR/AND do Trivium viram operações em inteiros de M bits e
processam M pares (chave, IV) de uma só vez.

Convenções idênticas às de cifras_reduzidas.py (estado s[1..288]).
"""
import random
import numpy as np


def pos_chave(j: int) -> int:
    """Posição no estado (1..80) do bit j da chave (bit j%8 do byte j//8)."""
    p, k = divmod(j, 8)
    return 80 - 8 * p - k


def pos_iv(j: int) -> int:
    """Posição no estado (94..173) do bit j do IV."""
    p, k = divmod(j, 8)
    return 93 + 80 - 8 * p - k


def estado_de_bytes(chaves, ivs):
    """Monta o estado bitsliced a partir de listas de chaves/IVs (bytes de 10 bytes)."""
    M = len(chaves)
    s = [0] * 289
    for m in range(M):
        for j in range(80):
            if (chaves[m][j // 8] >> (j % 8)) & 1:
                s[pos_chave(j)] |= 1 << m
            if (ivs[m][j // 8] >> (j % 8)) & 1:
                s[pos_iv(j)] |= 1 << m
    mask = (1 << M) - 1
    s[286] = s[287] = s[288] = mask
    return s


def estado_aleatorio(M: int, rng: random.Random):
    """Estado inicial com M pares (chave, IV) uniformemente aleatórios."""
    s = [0] * 289
    for i in range(1, 81):
        s[i] = rng.getrandbits(M)
    for i in range(94, 174):
        s[i] = rng.getrandbits(M)
    mask = (1 << M) - 1
    s[286] = s[287] = s[288] = mask
    return s


def executar(s0, T: int):
    """Roda T passos e devolve z[0..T-1]: a saída que o passo t produziria.

    z[t] é o primeiro bit de keystream de quem fez N = t passos de pré-aquecimento.
    Ou seja, UMA execução de 1152 + 128 passos contém o keystream para TODOS os N.
    """
    s = list(s0)
    z = []
    for _ in range(T):
        t1 = s[66] ^ s[93]
        t2 = s[162] ^ s[177]
        t3 = s[243] ^ s[288]
        z.append(t1 ^ t2 ^ t3)
        t1 ^= (s[91] & s[92]) ^ s[171]
        t2 ^= (s[175] & s[176]) ^ s[264]
        t3 ^= (s[286] & s[287]) ^ s[69]
        s[1:94] = [t3] + s[1:93]
        s[94:178] = [t1] + s[94:177]
        s[178:289] = [t2] + s[178:288]
    return z


def int_para_bits(x: int, M: int) -> np.ndarray:
    """Inteiro de M bits (M múltiplo de 8) -> array uint8 com M bits (bit m = amostra m)."""
    return np.unpackbits(np.frombuffer(x.to_bytes(M // 8, "little"), dtype=np.uint8),
                         bitorder="little")


def pares_de_bits(zs, M):
    """Lista de T inteiros -> matriz uint8 (T, M)."""
    return np.stack([int_para_bits(x, M) for x in zs])


def _validar():
    from cifras_reduzidas import trivium
    import os
    M = 16
    ch = [os.urandom(10) for _ in range(M)]
    iv = [os.urandom(10) for _ in range(M)]
    z = executar(estado_de_bytes(ch, iv), 1152 + 128)
    for N in (0, 1, 77, 288, 1152):
        for m in range(M):
            ref = trivium(ch[m], iv[m], 128, passos_init=N)
            got = [(z[N + k] >> m) & 1 for k in range(128)]
            assert ref == got, (N, m)
    print("bitslice.py: Trivium bitsliced confere com o módulo de referência.")


if __name__ == "__main__":
    _validar()
