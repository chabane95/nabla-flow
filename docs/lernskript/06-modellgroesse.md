# 6 · Wie groß ist mein Modell? Trainiere ich wie bei GPT-2?

Im Versuch mit den Fachfragen trainierst du **35.776 Parameter**. Ein Parameter ist eine gelernte Zahl, etwa der Eintrag einer Token- oder Query-Matrix. Bei der Antwortgewichtung von vier legst du einen *Hyperparameter* fest. Er bestimmt, wie stark bestimmte Vorhersagefehler zur Aktualisierung der Gewichte beitragen. Beide Versuchsläufe haben deshalb dieselbe Zahl lernbarer Parameter.

| Modell im Projekt | Einheiten im Vokabular | Blöcke × Breite | Köpfe je Block | Lernbare Parameter |
| --- | ---: | ---: | ---: | ---: |
| Fach- und Farbfragen, beide Versuchsläufe | 258 | 2 × 32 | 4 | 35.776 |
| Kleines Shakespeare-Byte-Modell | 258 | 1 × 48 | 4 | 43.056 |
| GPT-2 Small auf *Tiny Shakespeare* | 50.257 | 12 × 768 | 12 | 124.439.808 |
| GPT-2 Medium, Kurzlauf auf *Tiny Shakespeare* | 50.257 | 24 × 1.024 | 16 | 354.823.168 |

Die ersten beiden Modelle lesen UTF-8-Bytes und verwenden zwei besondere Symbole für Anfang und Ende. Die beiden GPT-2-Modelle lesen dieselben GPT-2-Tokens, die auch mehrere Bytes zusammenfassen können. Small und Medium haben jeweils 1.024 gelernte Positionen. Medium verdoppelt die Zahl der Blöcke und erhöht die Breite von 768 auf 1.024. [Fachmodell](../../results/summary.csv) · [Shakespeare-Byte-Modell](../../results/shakespeare/summary.csv) · [Konfigurationen im GPU-Trainer](../../tools/nfgpu.c) · [OpenAIs Small-Konfiguration](https://openaipublic.blob.core.windows.net/gpt-2/models/124M/hparams.json) · [OpenAIs Medium-Konfiguration](https://openaipublic.blob.core.windows.net/gpt-2/models/355M/hparams.json)

**Probe auf Papier.** Beim Fachmodell liefern die Token- und Positionstabellen zusammen $(258+64)\cdot32=10\,304$ Einträge. Jeder der zwei Blöcke hat $12\cdot32^2+13\cdot32=12\,704$ Gewichte, Bias- und Normierungsparameter. Die finale LayerNorm fügt $2\cdot32=64$ hinzu. Als Summe ergibt sich $10\,304+2\cdot12\,704+64=35\,776$. Die Ausgabe greift nochmals auf die Tokenmatrix zu. [Herleitung der Zählformel](02-transformer.md#zählen-als-unabhängige-kontrolle)

## Welche Teile des GPT-2-Trainings führe ich tatsächlich aus?

Bei jeder Position berechnet dein Decoder aus den vorherigen Tokens eine Verteilung für das nächste. Das tatsächlich folgende Token liefert das Ziel. Backpropagation berechnet für jedes Gewicht, wie empfindlich der Verlust auf eine kleine Änderung reagiert. AdamW verwendet diese Gradienten für einen Trainingsschritt. Das ist echtes autoregressives Training mit von Grund auf gelernten Gewichten. Bei vierfacher Gewichtung schätzt du eine gezielt auf das Antwortwort ausgerichtete Variante des Verlusts. [Rückwärtsrechnung und Update](../../src/gpt.c) · [GPT-2-Bericht, Abschnitt 2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=3)

Der *separate* GPU-Lauf im Projekt hat die GPT-2-Small-Architektur bereits auf einer RTX 4070 SUPER trainiert. Aus 311.356 GPT-2-Tokens im Trainingssplit zog er überlappende Fenster und verarbeitete $2\,400\cdot4\cdot128=1\,228\,800$ Zielpositionen. Die Positionstabelle hat 1.024 Plätze, das Training nutzte je Fenster die ersten 128. Das Originaltraining von GPT-2 verwendete WebText mit gut acht Millionen Dokumenten und 40 GB Text. Mit denselben Modellabmessungen kann man also ein Modell auf Shakespeare trainieren. Welche Texte es anschließend zuverlässig fortsetzen kann, ist eine empirische Frage. [GPU-Referenzlauf](../training-124m.md) · [Aufteilung der Trainingsdaten](../../results/gpt2-124m/data.json) · [WebText und GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=3)

## Könnte ich auch GPT-3 trainieren?

Brown et al. untersuchten acht Modelle von **125 Millionen bis 175 Milliarden Parametern**. Die kleinste Variante hatte zwölf Blöcke der Breite 768. Diese Größenordnung erreicht dein Projekt bereits. Für ein Experiment näher an GPT-3 Small müsstest du unter anderem die im Paper abwechselnd verwendete volle und lokal begrenzte kausale Attention nachbilden. Das Paper verwendet ein Kontextfenster von 2.048 Tokens und berichtet 300 Milliarden verarbeitete Trainingstokens je Modell. Schon für die kleinste Variante entspricht dieses Tokenbudget rund dem 244.000-Fachen des Shakespeare-Laufs. [GPT-3, Abschnitt 2.1 und Tabelle 2.1](https://arxiv.org/pdf/2005.14165#page=8)

Für das größte Modell reicht eine Speicherrechnung. Setze $N=175\cdot10^9$ und rechne in dezimalen GB und TB. Schon die Gewichte allein belegen bei zwei Byte pro Parameter **350 GB**. Der GPU-Trainer hier hält pro Parameter das Gewicht, den Gradienten und die beiden Adam-Momente als vier FP32-Zahlen im Speicher. Das wären $16N=2{,}8$ **TB** allein für diese vier Felder. Zwischenwerte für den Rückwärtspass kommen hinzu. Deine RTX 4070 SUPER meldet 12.282 MiB Gesamtspeicher. Ein Vortraining des 175-Milliarden-Modells auf dieser einzelnen Karte ist damit praktisch außer Reichweite. Die Autoren verteilten die großen Modelle innerhalb von Matrixmultiplikationen und über Schichten hinweg auf viele GPUs. [Speicherreservierung im GPU-Code](../../src/gpu.c) · [Hardware des Projektlaufs](../training-124m.md) · [GPT-3-Trainingsverfahren, Abschnitt 2.3](https://arxiv.org/pdf/2005.14165#page=9)

Ein lokal durchführbarer nächster Versuch ist der Einfluss der **Textvielfalt** bei 124 Millionen Parametern. Reserviere vorab ganze Theaterstücke für die Auswertung. Trainiere mit gleicher Architektur, gleichen Startgewichten und gleichem Tokenbudget einmal auf dem kleineren Trainingskorpus und einmal mit zusätzlichen, vom Prüfsatz getrennten Dramentexten. Ein niedrigerer Verlust auf denselben zurückgehaltenen Stücken wäre ein messbarer Gewinn durch die andere Datenmischung.

[Weiter zu GPT-2 Medium und dem gemessenen Grafikspeicherverbrauch](07-gpt2-medium.md) · [Übersicht](README.md)
