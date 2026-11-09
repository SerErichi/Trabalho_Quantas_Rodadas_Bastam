"""
Item 3: MixColumns na última rodada. A avalanche melhora; a segurança melhora?
 (a) avalanche + SAC com mixcolumns_na_ultima=True para R=1..6 (via g1_aes)
 (b) equivalência: MC(.) é linear, público e sem chave, e MC(x) xor K = MC(x xor MC^-1(K)).
     Logo  AES-R-com-MC-final(chave K0..KR)  =  MC( AES-R-padrão com última subchave K'_R = MC^-1(K_R) ).
     Quem vê o texto cifrado aplica MC^-1 (sem chave) e volta ao AES-R padrão: o ganho de
     avalanche é "cosmético" e não acrescenta nada contra um atacante.
"""
import csv, random, time
import g1_aes
from cifras_reduzidas import (aes_cifrar, aes_expandir_chave, xtime, _sub_bytes, _shift_rows,
                              _mix_columns, _add_round_key)
from util import registrar_tempo


def mul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = xtime(a)
        b >>= 1
    return r


def inv_mix_columns(s):
    M = [[14, 11, 13, 9], [9, 14, 11, 13], [13, 9, 14, 11], [11, 13, 9, 14]]
    out = []
    for c in range(4):
        a = s[4 * c:4 * c + 4]
        for i in range(4):
            out.append(mul(a[0], M[i][0]) ^ mul(a[1], M[i][1]) ^ mul(a[2], M[i][2]) ^ mul(a[3], M[i][3]))
    return out


def aes_padrao_com_ultima_subchave(texto, chave, R, subchave_final):
    """AES-R padrão (sem MC na última), mas trocando K_R por `subchave_final`."""
    k = list(aes_expandir_chave(bytes(chave)))
    k[R] = bytes(subchave_final)
    s = _add_round_key(list(texto), k[0])
    for r in range(1, R + 1):
        s = _shift_rows(_sub_bytes(s))
        if r < R:
            s = _mix_columns(s)
        s = _add_round_key(s, k[r])
    return bytes(s)


def verificar_equivalencia(n=500, semente=7):
    rng = random.Random(semente)
    linhas = []
    for R in range(1, 11):
        ok = 0
        for _ in range(n):
            p, k = rng.randbytes(16), rng.randbytes(16)
            c_var = aes_cifrar(p, k, R, mixcolumns_na_ultima=True)
            KR = aes_expandir_chave(k)[R]
            c_std = aes_padrao_com_ultima_subchave(p, k, R, bytes(inv_mix_columns(list(KR))))
            ok += bytes(_mix_columns(list(c_std))) == c_var and bytes(inv_mix_columns(list(c_var))) == c_std
        linhas.append([R, n, ok])
        print("equivalência R", R, ok, "/", n, flush=True)
    with open("dados/item3_equivalencia.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["R", "testes", "equivalentes"]); w.writerows(linhas)


if __name__ == "__main__":
    t0 = time.time()
    verificar_equivalencia()
    g1_aes.executar(mc=True, Rs=range(1, 7), tag="mc")
    registrar_tempo("item3", time.time() - t0)
