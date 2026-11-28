# nabla-flow

## GPT-2 Medium · RTX 4070 SUPER

- [Trainingslauf](results/gpt2-medium-355m/run.json) · 354.823.168 Parameter, FP32, 64 Updates auf *Tiny Shakespeare*, je 4 Fenster à 128 Tokens
- [Validierungsverlust](results/gpt2-medium-355m/history.csv) · 11,096 → 6,365 Nats pro Token

| Grafikspeicher | GiB |
| --- | ---: |
| Gewichte allein | 1,32 |
| [Trainerpuffer einschließlich Gewichten](results/gpt2-medium-355m/memory.csv) | 6,25 |
| Gesamte Karte vor dem Start | 1,42 |
| [Höchste gemessene Kartenbelegung (244 Messpunkte)](results/gpt2-medium-355m/board-memory.csv) | **7,92** |
| [Zuwachs gegenüber dem Startwert (GIF)](results/gpt2-medium-355m/memory.gif) | **6,50** |

## GPT-2 Small · Tiny Shakespeare

- [Trainingslauf](results/gpt2-124m/history.csv) · 124.439.808 Parameter, 2.400 Updates, 1.228.800 Zieltokens
- [Trainingsverlauf (GIF)](results/gpt2-124m/training.gif) · 121 s bis zur letzten Validierung, gemessen nach der Initialisierung

| Modell und Datensatz | Verlust (Nats/Token) | Perplexität |
| --- | ---: | ---: |
| [GPT-2 Small, Validierung](results/gpt2-124m/validation-reload.csv) | 4,145 | 63,10 |
| [GPT-2 Small, Test](results/gpt2-124m/test.csv) | 5,171 | 176,05 |
| [Unigramm, derselbe Test](results/gpt2-124m/unigram-test.csv) | 6,684 | 799,28 |

## Byte-Modell · Gewichtung des Antwortworts

- [Vergleich nach 1.800 Updates](results/answer-weight/comparison.csv) · 35.776 identische Startgewichte, gleiche Batches, Seed 42
- [Doppeltreffer](results/answer-weight/pairs.csv) · beide Fragen zur selben Belegung exakt beantwortet

| Belegungen mit Doppeltreffer | Antwortgewicht 1× | Antwortgewicht 4× |
| --- | ---: | ---: |
| Validierung gesamt | 4/28 | 10/28 |
| Test gesamt | 7/28 | 6/28 |
| Test · Farben | 3/14 | 6/14 |
| Test · Materialien | 4/14 | 0/14 |
