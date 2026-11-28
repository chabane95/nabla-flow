# 6 · Vom Byte-Modell zu GPT-2

Das Modell für die Fachfragen hat **35.776 lernbare Parameter**. Ein einzelner Eintrag der Query-Matrix ist ein solcher Parameter. Er hat vor dem ersten Trainingsschritt einen Anfangswert und kann sich bei jedem Update ändern. Der Faktor vier aus dem Antwortversuch ist dagegen ein *Hyperparameter*. Er steht vor dem Training fest und gewichtet den Fehler bei den Antwortbytes. Beide Läufe haben also gleich viele Parameter.

## Wo stecken die 35.776 Zahlen?

Das Fachmodell braucht für jedes seiner 258 Tokens einen Vektor mit 32 Einträgen. Dazu kommen 64 Positionsvektoren derselben Breite. Die beiden Tabellen enthalten zusammen $(258+64)\cdot32=10\,304$ Zahlen. In jedem der zwei Blöcke stecken $12\cdot32^2+13\cdot32=12\,704$ Einträge für Matrizen, Verschiebungen und Normierung. Die letzte LayerNorm hat für jede der 32 Komponenten einen Skalenfaktor und eine Verschiebung. Damit ergibt sich

$$
N=10\,304+2\cdot12\,704+2\cdot32=35\,776.
$$

Bei der Ausgabe dient dieselbe Tokenmatrix dazu, die Punktprodukte mit den 258 Tokenvektoren zu berechnen. Ihre Einträge zählen deshalb nur einmal. [Herleitung der Zählformel](02-transformer.md#zählen-als-unabhängige-kontrolle)

| Modell im Projekt | Vokabulargröße | Blöcke × Breite | Köpfe je Block | Lernbare Parameter |
| --- | ---: | ---: | ---: | ---: |
| Fach- und Farbfragen, beide Versuchsläufe | 258 | 2 × 32 | 4 | 35.776 |
| Kleines Shakespeare-Byte-Modell | 258 | 1 × 48 | 4 | 43.056 |
| GPT-2 Small auf *Tiny Shakespeare* | 50.257 | 12 × 768 | 12 | 124.439.808 |
| GPT-2 Medium, Kurzlauf auf *Tiny Shakespeare* | 50.257 | 24 × 1.024 | 16 | 354.823.168 |

Die Byte-Modelle kennen 256 Bytewerte und je ein Symbol für Anfang und Ende. GPT-2 Small und Medium verwenden dasselbe Vokabular mit 50.257 Tokens. Ein Token kann hier mehrere Bytes umfassen. Beide Modelle haben 1.024 gelernte Positionsvektoren. Medium hat doppelt so viele Blöcke wie Small. Da die großen Matrizen im Block mit dem Quadrat der Breite wachsen, fällt auch der Schritt von 768 auf 1.024 Komponenten ins Gewicht. [Fachmodell](../../results/summary.csv) · [Shakespeare-Byte-Modell](../../results/shakespeare/summary.csv) · [Konfigurationen im GPU-Trainer](../../tools/nfgpu.c) · [OpenAIs Small-Konfiguration](https://openaipublic.blob.core.windows.net/gpt-2/models/124M/hparams.json) · [OpenAIs Medium-Konfiguration](https://openaipublic.blob.core.windows.net/gpt-2/models/355M/hparams.json)

## Was hat das Training mit GPT-2 gemeinsam?

Im Fachmodell wie bei GPT-2 liefert der Text selbst das Trainingsziel. Für jede Position berechnet der Decoder aus dem bisherigen Text die Wahrscheinlichkeiten für das nächste Token. Die Wahrscheinlichkeit des tatsächlich folgenden Tokens bestimmt den Verlust. Backpropagation berechnet dessen Ableitungen nach den Gewichten. Hier im Projekt verwendet AdamW diese Gradienten für das nächste Update. Die Gewichte werden von Grund auf trainiert. Im Versuch mit vierfachem Antwortgewicht tragen die Antwortbytes entsprechend stärker zum Verlust bei. [Rückwärtsrechnung und Update](../../src/gpt.c) · [GPT-2-Bericht, Abschnitt 2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=3)

Der GPU-Lauf mit GPT-2 Small auf der RTX 4070 SUPER zeigt, was die Parameterzahl offenlässt. Sein Trainingskorpus enthielt 311.356 GPT-2-Tokens aus *Tiny Shakespeare*. Der Trainer zog daraus überlappende Fenster und verarbeitete in 2.400 Schritten mit vier Fenstern der Länge 128 insgesamt $2\,400\cdot4\cdot128=1\,228\,800$ Zielpositionen. Eine Textstelle konnte mehrfach vorkommen. Pro Fenster kamen nur die ersten 128 der 1.024 möglichen Positionen zum Einsatz. OpenAI trainierte GPT-2 auf WebText mit gut acht Millionen Dokumenten und 40 GB Text. Die Small-Architektur ist in beiden Fällen dieselbe, die Menge und Vielfalt der gelesenen Texte unterscheiden sich erheblich. [GPU-Referenzlauf](../training-124m.md) · [Aufteilung der Trainingsdaten](../../results/gpt2-124m/data.json) · [WebText und GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=3)

## Wo liegt die Grenze bei GPT-3?

Unter dem Namen GPT-3 untersuchten Brown et al. acht Modelle mit **125 Millionen bis 175 Milliarden Parametern**. Das kleinste hatte zwölf Blöcke der Breite 768, wie GPT-2 Small. GPT-3 verwendete ein Kontextfenster von 2.048 Tokens und von Schicht zu Schicht abwechselnd volle und lokal begrenzte kausale Attention. Auch das Trainingsbudget unterschied sich. Die Autoren berichten 300 Milliarden verarbeitete Tokens je Modell. Schon bei der kleinsten Variante entspricht das rund dem 244.000-Fachen des Shakespeare-Laufs mit GPT-2 Small. [GPT-3, Abschnitt 2.1 und Tabelle 2.1](https://arxiv.org/pdf/2005.14165#page=8)

Für die größte Variante lässt sich die Speicherfrage schon vor dem ersten Versuch beantworten. Bei $N=175\cdot10^9$ Parametern und zwei Byte je Gewicht belegen allein die Gewichte **350 GB**. Der GPU-Trainer des Projekts speichert Gewicht, Gradient und die beiden Adam-Momente mit je vier Byte. Für diese vier Felder wären $16N=2{,}8\cdot10^{12}$ Byte nötig, also **2,8 TB** in dezimalen Einheiten. Die Zwischenwerte des Rückwärtslaufs kommen hinzu. Die RTX 4070 SUPER des Versuchs meldet 12.282 MiB Gesamtspeicher. Schon der Gewichtssatz des größten GPT-3-Modells übersteigt diesen Speicher um ein Vielfaches. Für ihre großen Modelle verteilten die Autoren sowohl Matrixmultiplikationen als auch Schichten auf mehrere GPUs. [Speicherreservierung im GPU-Code](../../src/gpu.c) · [Hardware des Projektlaufs](../training-124m.md) · [GPT-3-Trainingsverfahren, Abschnitt 2.3](https://arxiv.org/pdf/2005.14165#page=9)

Eine andere Frage lässt sich mit den vorhandenen 124 Millionen Parametern untersuchen. **Verbessert mehr Textvielfalt die Vorhersage in einem ungelesenen Stück?** Für diesen Vergleich bleiben ganze Theaterstücke von beiden Trainingsläufen ausgeschlossen. Der erste Lauf verwendet den kleineren Trainingskorpus, der zweite zieht seine Fenster zusätzlich aus weiteren Dramentexten. Architektur, Startgewichte und Zahl der verarbeiteten Tokens bleiben gleich. Die Messgröße ist der mittlere Verlust beider Modelle auf denselben zurückgehaltenen Stücken.

[Weiter zu GPT-2 Medium und dem gemessenen Grafikspeicherverbrauch](07-gpt2-medium.md) · [Übersicht](README.md)
