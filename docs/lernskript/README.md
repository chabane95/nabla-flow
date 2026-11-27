# Wie lernt ein Sprachmodell, welches Fach gemeint ist?

Ein Lernskript zu **nabla-flow**. Für den Einstieg genügen bedingte Wahrscheinlichkeiten, die Ableitung einer Funktion und das Skalarprodukt zweier Vektoren. Die Rechnungen beginnen auf Papier. Für den praktischen Teil reichen ein C-Compiler und die kleinen, im Repository enthaltenen Modellgewichte.

> **Der erste Versuch.** In Fach A liegt Glas, in Fach B liegt Harz. Ergänze „In A liegt …“ und „In B liegt …“. Unterstreiche für jede Antwort die zwei Stellen im Text, die du miteinander verbinden musst. Welche Information hätte jemand, der vor der Antwort nur das letzte Leerzeichen sehen dürfte?

Genau dieses Experiment steckt im Datensatz. Beim Prompt `Fach A: Glas. Fach B: Harz. In B liegt ` erzeugt das gespeicherte Byte-Modell `Glas.`. Über den gesamten Test erreicht es trotzdem eine Perplexität von 1,270. Um beides zu verstehen, brauchen wir zwei verschiedene Messungen am selben Modell. Der eine Versuch fragt, wie viel Wahrscheinlichkeit das Modell jedem nächsten Byte gibt. Der andere fragt, ob beim selbstständigen Antworten das richtige Wort bis einschließlich Punkt entsteht. [Testmessung](../../results/test.csv) · [Antwortaufgaben](../../data/recall.tsv) · [Gespeicherte Beispielantwort](../../results/sample.txt)

## Der Rechenweg

| Kapitel | Frage | Was du am Ende selbst prüfen kannst |
| --- | --- | --- |
| [1 · Vom Text zur messbaren Vorhersage](01-vorhersage.md) | Wie wird aus einer Zeile Text eine Reihe von Wahrscheinlichkeitsaussagen? | Zielsequenz aufschreiben, eine Bigramm-Tabelle aus Zählungen bestimmen und ihren Verlust in Nats berechnen. |
| [2 · Wie der Decoder früheren Text nutzt](02-transformer.md) | Wie kann derselbe letzte Buchstabe je nach Vorgeschichte eine andere Prognose auslösen? | Eine Attention-Zeile von Hand ausrechnen, die kausale Grenze einzeichnen und die Modellparameter zählen. |
| [3 · Wie ein Fehler ein Gewicht verändert](03-lernen.md) | Wie kommt der Fehler für `H` oder `G` bis zu einem Gewicht, das beim Lesen des Fachnamens benutzt wurde? | Den Gradienten entlang der Rechenwege herleiten und sein Vorzeichen sowie seine Größe mit einem Differenzenquotienten prüfen. |
| [4 · Experimente mit den gespeicherten Modellen](04-labor.md) | Welche Eigenschaft weist ein bestandener Test nach? | Kausalität, Vorhersagegüte und exakte Antworten getrennt messen und die Shakespeare-Messungen mit der passenden Kontrollrechnung vergleichen. |
| [5 · Das Antwortwort vierfach gewichten](05-antwortgewicht.md) | Hilft eine geänderte Zielfunktion beim Wechsel der Frage? | Zwei Trainingsläufe mit identischen Startgewichten wiederholen und Doppeltreffer auf denselben Belegungen zählen. |
| [6 · Modellgröße, GPT-2 und GPT-3](06-modellgroesse.md) | Was trennt das kleine Experiment vom Training eines großen Sprachmodells? | Die Parameter zählen und aus dem Trainingsbudget und Speicherbedarf ableiten, was auf einer einzelnen GPU möglich ist. |
| [7 · GPT-2 Medium auf der RTX 4070 SUPER](07-gpt2-medium.md) | Wie viel Grafikspeicher braucht ein echter Trainingsschritt bei 355 Millionen Parametern? | Den Speicherbedarf aus der Modellform berechnen und mit Messungen des CUDA-Treibers und der gesamten Karte vergleichen. |

Bearbeite Kapitel 1 und 2 mit einem Blatt Papier. Für Kapitel 3 lohnt es sich, in den eigenen Formeln jedes mehrfach verwendete Gewicht zu markieren. Im Labor schreibe vor dem Start des Programms einen Satz der Form „Wenn meine Erklärung stimmt, muss sich bei diesem Eingriff … ändern“ und daneben ein Ergebnis, das dich zum Umdenken bringen würde.

## Beim Lesen des Codes

Die kleinste vollständige Schleife für das Byte-Modell steht in `train`. Verfolge dort einen einzigen Block. `x` zeigt auf das erste Eingabetoken, `x+1` auf die verschobenen Ziele. `gpt_forward` berechnet die Scores für alle Positionen, `gpt_backward` sammelt die Beiträge zum Gradienten und `gpt_step` aktualisiert nach mehreren Blöcken die Gewichte. Folge beim ersten Durchgang nur diesen vier Stellen. Die Reihenfolge, in der das Programm rechnet, ist dieselbe wie im Skript. [Trainingsschleife](../../src/main.c) · [Schnittstelle mit Konventionen für Summen und Mittelwerte](../../src/gpt.h)

Wer die Kettenregel erst an einem einzelnen Neuron auffrischen möchte, findet dafür ein ausgearbeitetes Übungsheft und eine mathematische Herleitung im Projekt. Zur anschließenden Lektüre eignen sich Abschnitt 3.2 von Vaswani et al. für Attention, Abschnitt 2 von Radford et al. für das autoregressive Modell und die Arbeit von Loshchilov und Hutter für den Optimierungsschritt. [Übungsheft](../workbook.docx) · [Mathematische Grundlagen](../../latex/backprop_first_principles.tex) · [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [GPT-2-Bericht](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) · [AdamW](https://arxiv.org/abs/1711.05101)
