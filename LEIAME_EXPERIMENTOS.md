# Quantas Rodadas Bastam? — código dos experimentos

## Como rodar (ordem; seeds fixas, resultados reproduzíveis)
    pip install numpy scipy pandas matplotlib
    python3 cifras_reduzidas.py      # autoteste do módulo de referência
    python3 bitslice.py              # valida o Trivium bitsliced contra a referência
    python3 nist_tests.py            # valida os testes NIST contra os exemplos da norma + os.urandom
    python3 g1_trivium.py            # G1 Trivium (avalanche em função de N=0..1152)
    python3 g1_aes.py                # G1 AES (avalanche + SAC, R=1..10)
    python3 item3.py                 # item 3: MixColumns na última rodada (+ equivalência)
    python3 bonus_integral.py        # bônus: propriedade integral (Square) no AES
    python3 bonus_cubo.py            # bônus: cube tester no Trivium
    python3 g2_aes.py                # G2 AES em modo contador (NIST)
    python3 g2_trivium.py            # G2 Trivium, chave e IV aleatórios
    python3 g2_trivium_ivcont.py     # G2 Trivium, IV em contador
    python3 graficos.py              # gráficos (graficos/) e dados/resumo_conclusoes.csv

## Arquivos
- bitslice.py: Trivium com bitslice (M pares chave/IV por inteiro grande). Uma execução de 1152+128 passos
  contém o keystream de TODOS os N (z[N+k] é o k-ésimo bit de quem aqueceu N passos).
- nist_tests.py: monobit, block frequency, runs, longest run, matrix rank, DFT, serial, approximate entropy,
  cusum (frente/trás). NÃO implementados: template, Maurer, linear complexity, random excursions.
  Serial é reportado como min(p1,p2): taxa nula de reprovação ~2%, e a uniformidade dos seus p-valores não é aplicável.
- dados/: CSVs brutos de cada gráfico; dados/tempos.csv: tempo de cada experimento (item 4).

## Infraestrutura em que ESTES dados foram gerados (item 4: substituam pela de vocês!)
Intel Xeon 2,1 GHz, 1 núcleo/1 thread, 3 GB RAM, Ubuntu 24, Python 3.12.3, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2.
Tempos em dados/tempos.csv.
