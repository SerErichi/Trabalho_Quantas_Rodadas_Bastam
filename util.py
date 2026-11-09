"""Utilidades comuns: registro de tempos de execução (item 4 do enunciado)."""
import csv, os, time

def registrar_tempo(nome, seg):
    novo = not os.path.exists("dados/tempos.csv")
    with open("dados/tempos.csv", "a", newline="") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["experimento", "segundos"])
        w.writerow([nome, round(seg, 1)])
