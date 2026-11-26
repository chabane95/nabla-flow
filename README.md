# nabla-flow

Wie weit kommt GPT-2 auf einer einzelnen RTX 4070 SUPER, wenn man mit zufälligen Gewichten und gut einem Megabyte Shakespeare anfängt? Für diesen Versuch läuft der Decoder in C und CUDA. Er hat die 124.439.808 Parameter von GPT-2 Small. Vor dem ersten Update und danach alle 100 Schritte bekommt er denselben Redeanfang, `ROMEO:` mit einem Zeilenumbruch.

Nach 2.400 Schritten steht da diese Zeile.

```text
ROMEO:
Now, that is the time of me to hear.
```

Ab **Now** ist alles generiert. Nach der Leerzeile hat das Modell selbst das Ende der Rede gesetzt. Am Anfang des Trainings stand an derselben Stelle noch ` Arsenal stained slap`. Was mich an der letzten Zeile interessiert, ist das holprige *the time of me to hear*. Der Anfang funktioniert gesprochen ganz gut. Gegen Ende passen die Wörter nur noch ungefähr zusammen. Die längste wörtliche Übereinstimmung mit einer Trainingsrede ist `, that is`, gerade mal drei Token. [Alle 25 Textproben mit Token-IDs und Fundstellen](results/gpt2-124m/samples-report.json)

## Was in den zwei Minuten passiert

[![Links der Verlust auf festen Trainingsfenstern und auf der Validierung. Rechts die Validierungs-Perplexität über den verarbeiteten Zieltoken.](results/gpt2-124m/training.png)](results/gpt2-124m/training.svg)

Nach 400 Updates ist der größte Sprung in der Kurve schon vorbei. Zu diesem Zeitpunkt schreibt das Modell `But I will I have thy good that I have not`. Man hört den Tonfall, aber schon bei *will I have* verliert sich der Satz. Bei jeder Textprobe sind Temperatur, Top-k und Zufallsseed gleich. So kann man beim Lesen verfolgen, was sich mit den Gewichten ändert. [Training als Animation](results/gpt2-124m/training.gif) · [Messwerte der Abbildung](results/gpt2-124m/history.csv)

Pro Update zieht der Trainer vier Fenster zu je 128 Token. Nach 2.400 Updates sind das 1.228.800 Vorhersagen. Der Trainingsabschnitt selbst enthält nur 311.356 Token. Viele Stellen kommen also mehrmals dran. Das kurze Fenster ist auch die Stelle, an der ich für den eigenen Rechner gespart habe. Alle zwölf Transformerblöcke werden trainiert, von den 1.024 möglichen Positionen sieht das Modell während des Trainings nur die ersten 128. Bis zur letzten Validierung vergingen auf der 12-GB-Karte 121 Sekunden. [Lauf wiederholen oder einen eigenen Prompt eingeben](docs/training-124m.md)

## Wie gut sagt es fremden Text voraus?

361 ganze Reden sind für die Validierung reserviert. Der Trainer sieht sie alle 100 Schritte und misst, wie viel Wahrscheinlichkeit das Modell den tatsächlich folgenden Token gibt. Die Gewichte werden dabei nur gelesen. Schritt 2.400 hat den niedrigsten mittleren Verlust. Erst mit diesen gespeicherten Gewichten kommt der Test auf den letzten 362 Reden.

| Modell und Textabschnitt | Vorhergesagte Token | Verlust in Nat pro Token | Perplexität |
| --- | ---: | ---: | ---: |
| GPT-2, Validierung | 13.106 | 4,145 | 63,10 |
| GPT-2, Test | 13.564 | 5,171 | 176,05 |
| Unigramm, derselbe Test | 13.564 | 6,684 | 799,28 |

Das Unigramm zählt, wie häufig jedes Token im Training vorkommt. Nach `my` gibt es dieselben Wahrscheinlichkeiten aus wie nach `ROMEO:`. Der Abstand von 799 zu 176 zeigt, was der Decoder auf diesem Text gegenüber der reinen Häufigkeitszählung gewinnt. Perplexität ist hier `exp(mittlerer Verlust)`, gerechnet mit dem GPT-2-Tokenizer. [Testmessung](results/gpt2-124m/test.csv) · [Unigramm-Messung](results/gpt2-124m/unigram-test.csv)

Warum liegen Validierung und Test so weit auseinander? Ein Unterschied fällt beim Lesen auf. Die Reden bleiben in der Reihenfolge der Vorlage. In der Validierung tragen 86 von 361 Reden eine Sprecherangabe, die im Training fehlt. Im Test sind es 290 von 362, darunter Prospero und Miranda. Wie viel vom höheren Verlust auf die Namen selbst entfällt, wäre eine eigene Messung. [Aufteilung von Tiny Shakespeare](results/gpt2-124m/data.json)

## Zwei Stellen im Code

Beim Lesen würde ich mit der kausalen Maske anfangen. Für die Vorhersage hinter `Now,` darf der Attention-Kopf die Wörter rechts vom Komma noch nicht kennen. Das lässt sich direkt prüfen. Man tauscht das Ende der Eingabe aus und vergleicht die Ausgaben vor der Änderung. Im skalaren C-Kern müssen die früheren Logits Bit für Bit gleich bleiben. Im kleineren Byte-Modell kann man sich die berechneten Gewichte für ein kurzes Beispiel auch als Matrix ansehen. [C-Kern und Maskentest](src/gpt.c) · [Animierte Attention-Matrix](figures/attention.gif) · [Vaswani et al., Abschnitt 3.2](https://arxiv.org/abs/1706.03762)

Die zweite Stelle ist die Tokenmatrix. GPT-2 benutzt sie zweimal, am Eingang zum Nachschlagen eines Tokenvektors und am Ausgang für die 50.257 möglichen nächsten Token. Beim Rückwärtsrechnen bekommt dieselbe Matrix deshalb Gradienten aus beiden Verwendungen. Man kann das am kleinen Netz mit finiten Differenzen für jeden Parameter prüfen. Auf der GPU übernimmt cuBLAS die Matrixprodukte. Die Kernel für Attention, Normalisierung und AdamW stehen hier im Projekt. [GPU-Code](src/gpu.c) · [CUDA-Kernel](src/gpu_kernels.cu) · [Parameterzahl nachrechnen](docs/gpt2-small.md) · [GPT-2-Bericht](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## Zum Nachrechnen

Die beiden kleinen Decoder lassen sich direkt aus dem Repository auf der CPU ausprobieren. Beim Labormodell reicht dafür eine Zeile. Auf `Fach A: Glas. Fach B: Harz. In B liegt ` folgt ausgerechnet `Glas.`. Der Satzbau sitzt, die Zuordnung geht schief. Deshalb zählt der Versuch neben der Perplexität auch die vollständig richtigen Antworten. Es sind 26 von 56. [Byte-Modelle bauen und ausprobieren](docs/byte-modelle.md) · [Kleiner Shakespeare-Versuch](results/shakespeare/README.md)

Das Lernheft fängt bei einem einzelnen Gewicht an. Der Weg zur Attention führt über den Gradienten der Softmax-Regression, den man bei Andrew Ng auf Papier nachrechnen kann. Für den Übergang vom Bigramm zum Decoder und für die 124M-Konfiguration waren die beiden Vorlesungen von Andrej Karpathy die Vorlage. [Lernheft für Word](docs/Lernheft.docx?raw=1) · [Eigene Ableitungen im LaTeX-Skript](latex/backprop_first_principles.tex) · [CS229-Notizen](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) · [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=535s) · [Let's reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU)
