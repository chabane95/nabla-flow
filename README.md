# nabla-flow
Kleine Lern-Implementierung von Backpropagation auf einem Computational Graph.

## Worum es hier geht
- Reverse-Mode Automatic Differentiation (Backprop) from scratch.
- Fokus auf mathematische Korrektheit und Nachvollziehbarkeit.
- Kein Framework-Overhead, nur Kernmechanik.

## Kernidee in 3 Punkten
- Jeder Knoten speichert:
  - `data` (Forward-Wert)
  - `grad` (Adjungierte `dL/dv`)
- Rueckwaerts wird in umgekehrter topologischer Reihenfolge akkumuliert.
- Pro Kante `u -> w` gilt lokal:
  - `grad[u] += grad[w] * (dw/du)`

## Vollstaendige Herleitungen (LaTeX)
Alle Herleitungen aus First Principles liegen in:
- `latex/backprop_first_principles.tex`

Inhalt der LaTeX-Ausarbeitung:
- Ableitung und partielle Ableitung aus Grenzwertdefinitionen
- Reverse-Accumulation-Gleichung auf DAGs
- Lokale Regeln (Addition, Multiplikation, Potenz)
- Vollstaendiges Kettenregel-Beispiel
- Finite-Difference Gradient-Check

## Build
```bash
cd latex
pdflatex backprop_first_principles.tex
```

## Referenzen
- Spivak, *Calculus*
- Rudin, *Principles of Mathematical Analysis*
- Bishop, *Pattern Recognition and Machine Learning*
- Goodfellow, Bengio, Courville, *Deep Learning*
- Nocedal, Wright, *Numerical Optimization*
