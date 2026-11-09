#!/bin/bash
cd /home/claude/trab
for s in g1_aes item3 bonus_integral bonus_cubo g2_aes g2_trivium; do
  echo "=== $s $(date +%T)"
  python3 $s.py
done
echo "=== FIM $(date +%T)"
