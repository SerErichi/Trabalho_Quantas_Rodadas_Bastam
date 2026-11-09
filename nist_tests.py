"""
nist_tests.py: subconjunto da suíte NIST SP 800-22 Rev.1a em numpy/scipy.

Testes: monobit, block_frequency, runs, longest_run, matrix_rank, dft,
serial, approx_entropy, cusum_f, cusum_b.
Cada função recebe um array de bits (uint8, 0/1) e devolve o p-valor
(serial devolve o menor dos dois p-valores).
Validados contra os exemplos numéricos da própria norma (ver _validar()).
Não implementados (custo alto em Python): non-overlapping template, Maurer,
linear complexity, random excursions.
"""
import math
import numpy as np
from scipy.special import erfc, gammaincc
from scipy.stats import norm


def monobit(b):
    n = len(b)
    S = int(b.sum()) * 2 - n
    return float(erfc(abs(S) / math.sqrt(n) / math.sqrt(2)))


def block_frequency(b, M=128):
    N = len(b) // M
    pi = b[:N * M].reshape(N, M).mean(axis=1)
    chi = 4.0 * M * float(((pi - 0.5) ** 2).sum())
    return float(gammaincc(N / 2.0, chi / 2.0))


def runs(b):
    n = len(b)
    pi = b.mean()
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
        return 0.0
    V = 1 + int(np.count_nonzero(b[1:] != b[:-1]))
    return float(erfc(abs(V - 2 * n * pi * (1 - pi)) / (2 * math.sqrt(2 * n) * pi * (1 - pi))))


_LR = {8: (8, [1, 2, 3, 4], [0.2148, 0.3672, 0.2305, 0.1875]),
       128: (128, [4, 5, 6, 7, 8, 9], [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124])}


def _maior_corrida(blocos):
    out = np.zeros(len(blocos), dtype=np.int64)
    for i, row in enumerate(blocos):
        z = np.flatnonzero(np.concatenate(([0], row == 0, [0]) if False else ([True], row == 0, [True])))
        out[i] = int(np.diff(z).max()) - 1
    return out


def longest_run(b, M=128):
    _, lim, pis = _LR[M]
    N = len(b) // M
    L = _maior_corrida(b[:N * M].reshape(N, M))
    K = len(lim) - 1
    v = np.zeros(K + 1)
    for l in L:
        idx = min(max(int(l) - lim[0], 0), K)
        v[idx] += 1
    chi = float((((v - N * np.array(pis)) ** 2) / (N * np.array(pis))).sum())
    return float(gammaincc(K / 2.0, chi / 2.0))


def _rank_gf2(rows):
    rank = 0
    rows = list(rows)
    for bit in range(31, -1, -1):
        piv = None
        for i in range(rank, len(rows)):
            if (rows[i] >> bit) & 1:
                piv = i
                break
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(len(rows)):
            if i != rank and (rows[i] >> bit) & 1:
                rows[i] ^= rows[rank]
        rank += 1
    return rank


def matrix_rank(b):
    N = len(b) // 1024
    words = np.packbits(b[:N * 1024].reshape(N * 32, 32), axis=1).view(">u4").reshape(N, 32)
    f = [0, 0, 0]
    for mat in words:
        r = _rank_gf2([int(x) for x in mat])
        f[0 if r == 32 else 1 if r == 31 else 2] += 1
    p = [0.2888, 0.5776, 0.1336]
    chi = sum((f[i] - N * p[i]) ** 2 / (N * p[i]) for i in range(3))
    return float(math.exp(-chi / 2.0))


def dft(b):
    n = len(b)
    S = np.abs(np.fft.fft(2.0 * b - 1.0))[: n // 2]
    T = math.sqrt(math.log(1 / 0.05) * n)
    N0 = 0.95 * n / 2
    N1 = int(np.count_nonzero(S < T))
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4)
    return float(erfc(abs(d) / math.sqrt(2)))


def _psi2(b, m):
    n = len(b)
    if m == 0:
        return 0.0
    ext = np.concatenate([b, b[: m - 1]]).astype(np.int64)
    code = np.zeros(n, dtype=np.int64)
    for i in range(m):
        code = (code << 1) | ext[i:i + n]
    cnt = np.bincount(code, minlength=2 ** m).astype(np.float64)
    return float((2 ** m) / n * (cnt ** 2).sum() - n)


def serial(b, m=8):
    p0, p1, p2 = _psi2(b, m), _psi2(b, m - 1), _psi2(b, m - 2)
    d1, d2 = p0 - p1, p0 - 2 * p1 + p2
    return min(float(gammaincc(2 ** (m - 2), d1 / 2.0)), float(gammaincc(2 ** (m - 3), d2 / 2.0)))


def _phi(b, m):
    n = len(b)
    ext = np.concatenate([b, b[: m - 1]]).astype(np.int64)
    code = np.zeros(n, dtype=np.int64)
    for i in range(m):
        code = (code << 1) | ext[i:i + n]
    c = np.bincount(code, minlength=2 ** m).astype(np.float64) / n
    c = c[c > 0]
    return float((c * np.log(c)).sum())


def approx_entropy(b, m=5):
    n = len(b)
    ap = _phi(b, m) - _phi(b, m + 1)
    chi = 2.0 * n * (math.log(2) - ap)
    return float(gammaincc(2 ** (m - 1), chi / 2.0))


def _cusum(b, reverse=False):
    n = len(b)
    x = 2 * b.astype(np.int64) - 1
    if reverse:
        x = x[::-1]
    z = int(np.abs(np.cumsum(x)).max())
    if z == 0:
        return 1.0
    sn = math.sqrt(n)
    k1 = np.arange(math.ceil((-n / z + 1) / 4), math.floor((n / z - 1) / 4) + 1)
    k2 = np.arange(math.ceil((-n / z - 3) / 4), math.floor((n / z - 1) / 4) + 1)
    s1 = (norm.cdf((4 * k1 + 1) * z / sn) - norm.cdf((4 * k1 - 1) * z / sn)).sum()
    s2 = (norm.cdf((4 * k2 + 3) * z / sn) - norm.cdf((4 * k2 + 1) * z / sn)).sum()
    return float(1 - s1 + s2)


def cusum_f(b):
    return _cusum(b, False)


def cusum_b(b):
    return _cusum(b, True)


TESTES = {
    "monobit": monobit,
    "block_frequency": block_frequency,
    "runs": runs,
    "longest_run": longest_run,
    "matrix_rank": matrix_rank,
    "dft": dft,
    "serial": serial,
    "approx_entropy": approx_entropy,
    "cusum_f": cusum_f,
    "cusum_b": cusum_b,
}


def rodar_bateria(bits):
    """bits: array uint8 0/1. Devolve {teste: p-valor}."""
    return {nome: f(bits) for nome, f in TESTES.items()}


def _bits(s):
    return np.array([int(c) for c in s if c in "01"], dtype=np.uint8)


def _validar():
    """Exemplos numéricos do NIST SP 800-22 Rev.1a (seções 2.x.4)."""
    e100 = _bits("1100100100001111110110101010001000100001011010001100001000110100"
                 "110001001100011001100010100010111000")
    def ok(nome, got, esp, tol=2e-4):
        assert abs(got - esp) < tol, (nome, got, esp)
    ok("monobit", monobit(e100), 0.109599)
    ok("runs", runs(e100), 0.500798)
    ok("cusum_f", cusum_f(e100), 0.219194)
    ok("cusum_b", cusum_b(e100), 0.114866)
    ok("block_frequency(M=10)", block_frequency(_bits("0110011010"), 3), 0.801252)
    ok("dft", dft(e100), 0.646355)
    e10 = _bits("0011011101")
    p0, p1, p2 = _psi2(e10, 3), _psi2(e10, 2), _psi2(e10, 1)
    ok("serial p1", float(gammaincc(2, (p0 - p1) / 2)), 0.808792)
    ok("serial p2", float(gammaincc(1, (p0 - 2 * p1 + p2) / 2)), 0.670320)
    e = _bits("0100110101")
    ap = _phi(e, 3) - _phi(e, 4)
    ok("apen", float(gammaincc(2 ** 2, 2 * 10 * (math.log(2) - ap) / 2)), 0.261961)
    lr = _bits("11001100 00010101 01101100 01001100 11100000 00000010 01001101 01010001 "
               "00010011 11010110 10000000 11010111 11001100 11100110 11011000 10110010")
    ok("longest_run", longest_run(lr, 8), 0.180609)
    print("nist_tests.py: exemplos numéricos da norma conferem.")


def _validar_uniformidade(m=100, n=2 ** 18):
    """Fonte realmente aleatória (os.urandom): ~99% de aprovação em cada teste."""
    import os
    aprov = {k: 0 for k in TESTES}
    for _ in range(m):
        b = np.unpackbits(np.frombuffer(os.urandom(n // 8), dtype=np.uint8))
        for k, p in rodar_bateria(b).items():
            aprov[k] += p >= 0.01
    print({k: v / m for k, v in aprov.items()})


if __name__ == "__main__":
    _validar()
    _validar_uniformidade()
