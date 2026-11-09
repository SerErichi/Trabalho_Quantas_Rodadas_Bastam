"""
cifras_reduzidas.py: AES-128 e Trivium com número de rodadas configurável.

Material de apoio do Trabalho Prático "Quantas Rodadas Bastam?"
Confiabilidade e Segurança de Software (98G08-04), PUCRS.

Implementação didática em Python puro: prioriza legibilidade, não velocidade.
Para gerar muitas amostras você provavelmente vai querer otimizar (PyPy,
numpy, C, bitslicing...). Se escrever a sua própria versão, valide-a contra
esta (e contra os vetores de teste em _autoteste) antes de confiar nela.

Uso rápido:
    python3 cifras_reduzidas.py          # roda o autoteste

    from cifras_reduzidas import aes_cifrar, trivium
    c = aes_cifrar(bytes(16), bytes(16), rodadas=3)
    z = trivium(bytes(10), bytes(10), nbits=128, passos_init=400)

"""

from functools import lru_cache

# ---------------------------------------------------------------------------
# Utilidades de bits
# ---------------------------------------------------------------------------


def inverter_bit(dados: bytes, j: int) -> bytes:
    """Devolve uma cópia de `dados` com o bit j invertido (bit j = bit j%8 do byte j//8)."""
    b = bytearray(dados)
    b[j // 8] ^= 1 << (j % 8)
    return bytes(b)


def hamming(a: bytes, b: bytes) -> int:
    """Número de bits diferentes entre duas sequências de bytes de mesmo tamanho."""
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def bytes_para_bits(dados: bytes) -> list[int]:
    """Bytes -> lista de bits, na convenção bit j = bit j%8 do byte j//8."""
    return [(dados[j // 8] >> (j % 8)) & 1 for j in range(8 * len(dados))]


def bits_para_bytes(bits: list[int]) -> bytes:
    """Lista de bits -> bytes (inverso de bytes_para_bits). len(bits) deve ser múltiplo de 8."""
    return bytes(sum(bits[8 * i + k] << k for k in range(8)) for i in range(len(bits) // 8))


# ---------------------------------------------------------------------------
# AES-128 com R rodadas
# ---------------------------------------------------------------------------

SBOX = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
]

INV_SBOX = [0] * 256
for _x, _y in enumerate(SBOX):
    INV_SBOX[_y] = _x

RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


def xtime(a: int) -> int:
    """Multiplicação por x (isto é, por 02) em GF(2^8) módulo x^8 + x^4 + x^3 + x + 1."""
    a <<= 1
    return (a ^ 0x1B) & 0xFF if a & 0x100 else a


@lru_cache(maxsize=1024)
def aes_expandir_chave(chave: bytes) -> tuple[bytes, ...]:
    """Key schedule do AES-128: devolve as 11 subchaves K0..K10 (16 bytes cada)."""
    assert len(chave) == 16, "AES-128: a chave deve ter 16 bytes"
    w = [list(chave[4 * i:4 * i + 4]) for i in range(4)]
    for i in range(4, 44):
        t = list(w[i - 1])
        if i % 4 == 0:
            t = t[1:] + t[:1]                   # RotWord
            t = [SBOX[b] for b in t]            # SubWord
            t[0] ^= RCON[i // 4 - 1]
        w.append([a ^ b for a, b in zip(w[i - 4], t)])
    return tuple(bytes(sum(w[4 * r:4 * r + 4], [])) for r in range(11))


# O estado é uma lista de 16 bytes em ordem de coluna (FIPS-197):
# estado[i] = linha i % 4, coluna i // 4.

def _sub_bytes(s):
    return [SBOX[b] for b in s]


def _shift_rows(s):
    # a linha r é rotacionada r posições para a esquerda
    return [s[(i + 4 * (i % 4)) % 16] for i in range(16)]


def _mix_columns(s):
    out = []
    for c in range(4):
        a = s[4 * c:4 * c + 4]
        t = a[0] ^ a[1] ^ a[2] ^ a[3]
        out += [a[i] ^ t ^ xtime(a[i] ^ a[(i + 1) % 4]) for i in range(4)]
    return out


def _add_round_key(s, k):
    return [a ^ b for a, b in zip(s, k)]


def aes_cifrar(texto: bytes, chave: bytes, rodadas: int = 10,
               mixcolumns_na_ultima: bool = False) -> bytes:
    """Cifra um bloco de 16 bytes com AES-128 reduzido a `rodadas` rodadas (1 a 10).

    mixcolumns_na_ultima=True aplica MixColumns também na última rodada
    (variante NÃO padrão, útil para uma das perguntas do trabalho).
    """
    assert len(texto) == 16, "o bloco deve ter 16 bytes"
    assert 1 <= rodadas <= 10, "rodadas deve estar entre 1 e 10"
    k = aes_expandir_chave(bytes(chave))
    s = _add_round_key(list(texto), k[0])
    for r in range(1, rodadas + 1):
        s = _shift_rows(_sub_bytes(s))
        if r < rodadas or mixcolumns_na_ultima:
            s = _mix_columns(s)
        s = _add_round_key(s, k[r])
    return bytes(s)


# ---------------------------------------------------------------------------
# Trivium com N passos de inicialização
# ---------------------------------------------------------------------------


def trivium(chave: bytes, iv: bytes, nbits: int, passos_init: int = 1152) -> list[int]:
    """Gera `nbits` bits de keystream do Trivium após `passos_init` passos de inicialização.

    Com passos_init = 1152 (= 4 x 288) é o Trivium padrão.
    Notação da especificação: estado s_1..s_288 (aqui s[1]..s[288]; s[0] não é usado).
    """
    assert len(chave) == 10 and len(iv) == 10, "chave e IV têm 80 bits (10 bytes)"
    s = [0] * 289
    for p in range(10):
        for k in range(8):
            s[80 - 8 * p - k] = (chave[p] >> k) & 1         # registrador 1: s_1..s_93
            s[93 + 80 - 8 * p - k] = (iv[p] >> k) & 1       # registrador 2: s_94..s_177
    s[286] = s[287] = s[288] = 1                            # registrador 3: s_178..s_288

    z = []
    for passo in range(passos_init + nbits):
        t1 = s[66] ^ s[93]
        t2 = s[162] ^ s[177]
        t3 = s[243] ^ s[288]
        if passo >= passos_init:
            z.append(t1 ^ t2 ^ t3)
        t1 ^= (s[91] & s[92]) ^ s[171]
        t2 ^= (s[175] & s[176]) ^ s[264]
        t3 ^= (s[286] & s[287]) ^ s[69]
        s[1:94] = [t3] + s[1:93]
        s[94:178] = [t1] + s[94:177]
        s[178:289] = [t2] + s[178:288]
    return z


# ---------------------------------------------------------------------------
# Autoteste
# ---------------------------------------------------------------------------


def _autoteste():
    # FIPS-197, Apêndices B e C.1
    assert aes_cifrar(bytes.fromhex("3243f6a8885a308d313198a2e0370734"),
                      bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")).hex() \
        == "3925841d02dc09fbdc118597196a0b32"
    pt = bytes.fromhex("00112233445566778899aabbccddeeff")
    k = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    assert aes_cifrar(pt, k).hex() == "69c4e0d86a7b0430d8cdb78070b4c55a"
    # FIPS-197 C.1, round[2].start: estado após 1 rodada completa (com MixColumns)
    assert aes_cifrar(pt, k, rodadas=1, mixcolumns_na_ultima=True).hex() \
        == "89d810e8855ace682d1843d8cb128fe4"

    # Trivium: vetores de teste do eSTREAM
    z = trivium(bytes.fromhex("80000000000000000000"), bytes(10), 8 * 272)
    ks = bits_para_bytes(z)
    assert ks[:16].hex().upper() == "38EB86FF730D7A9CAF8DF13A4420540D"
    assert ks[256:272].hex().upper() == "9A203CF7B2F3F09C43D188AA13A5A202"
    z = trivium(bytes.fromhex("0053A6F94C9FF24598EB"), bytes.fromhex("0D74DB42A91077DE45AC"), 128)
    assert bits_para_bytes(z).hex().upper() == "F4CD954A717F26A7D6930830C4E7CF08"

    # Convenções de bits
    assert inverter_bit(bytes(2), 9) == bytes([0, 2])
    assert bits_para_bytes(bytes_para_bits(b"\x12\xab")) == b"\x12\xab"
    print("autoteste OK: AES-128 (FIPS-197) e Trivium (eSTREAM) conferem.")


if __name__ == "__main__":
    _autoteste()
