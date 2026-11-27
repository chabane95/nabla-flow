# Ein Buchstabe als Gegenprobe

## Einstieg für die Website

In Fach A liegt Glas, in Fach B liegt Harz. Auf „In A liegt …“ antwortet ein kleines Sprachmodell mit „Glas.“ Ich ändere einen einzigen Buchstaben und frage nach B. Die Antwort lautet wieder „Glas.“ Wer nur die erste Frage gestellt hätte, hätte dem Modell diesen Treffer als Beleg dafür anrechnen können, dass es weiß, wo das Glas liegt. Die Gegenfrage kostet einen Buchstaben. [Beispiel und Laborprotokoll](lernskript/04-labor.md)

Was konnte das Modell überhaupt lesen? Im selben Byte-Modell lässt sich ein einzelner Attention-Kopf beim Lesen von `Links: rot.` beobachten. Beim `t` gibt er dem vorangehenden `o` ein Gewicht von 0,71. Das folgende Satzzeichen ist an dieser Position noch unsichtbar. Eine dunkle Zelle verrät, wie stark der Kopf den dort gespeicherten Vektor beimischt. Ob das am Ende zur richtigen Antwort führt, kann man wieder mit den zwei Fragen prüfen. [Attention als GIF](../figures/attention.gif) · [Gewicht jeder Zelle](../results/attention.csv)

In zwei Läufen mit identischen Startgewichten bekommt jedes Byte im Antwortwort einmal einfaches, einmal vierfaches Gewicht. Nach 1.800 Lernschritten lösen die beiden Modelle auf der Validierung 4 beziehungsweise 10 von 28 Belegungen vollständig. Der Blick auf jede einzelne Belegung ändert die Geschichte. Bei den Farben steigt die Zahl der Doppeltreffer von 1 auf 10, bei den Materialfächern fällt sie von 3 auf 0. Auf dem getrennten Test sinkt die Gesamtzahl sogar von sieben auf sechs. Was würde passieren, wenn Glas und Harz links und rechts lägen und die Farben in den Fächern A und B? So ließen sich Wortschatz und Frageform gegeneinander testen. [Versuch und Herleitung](lernskript/05-antwortgewicht.md) · [Farb- und Materialfragen im Bild](../results/answer-weight/paired-recall.svg) · [Rohdaten pro Belegung](../results/answer-weight/pairs.csv)

**Zum Weitererzählen.** Bei einem zweiten Experiment lernt GPT-2 Small, Shakespeare fortzusetzen. Wie viel Wahrscheinlichkeit bekommt das nächste tatsächlich geschriebene Token? Schau im vergrößerten Ausschnitt auf Schritt 1.200 und schätze für beide Reihen in Panel (a), um wie viele Nats pro Token sie von dort bis Schritt 2.400 fallen. Wie können beide Kurven fallen und ihr Abstand trotzdem wachsen? [Trainingskurven als GIF](../results/gpt2-124m/training.gif) · [Standbild](../results/gpt2-124m/training.svg) · [25 Messzeitpunkte](../results/gpt2-124m/history.csv)

<details>
<summary>Die zwei Differenzen nachrechnen</summary>

Im festen Trainingsausschnitt fällt der Verlust von 4,058 auf 3,611 Nats pro Token, also um rund 0,448. Auf der Validierung fällt er von 4,375 auf 4,145, um rund 0,230. Beide Reihen fallen, während ihr Abstand von 0,317 auf 0,534 wächst. Die zusätzliche Übung auf den Trainingsfenstern bringt dem Modell auf zurückgehaltenem Text einen kleineren Gewinn.

</details>

## Einstieg für LinkedIn

354.823.168 Gewichte zu je vier Byte ergeben 1,32 GiB. Wie viel Speicher zeigt eine 12-GB-Grafikkarte an, wenn sie dieses Modell tatsächlich trainiert?

Ein Trainingsschritt braucht für jedes Gewicht auch einen Gradienten und zwei Zahlen für AdamW. Aus einem Feld werden vier. Zusammen sind das 5,29 GiB. Bei vier Textfenstern mit je 128 Tokens kommen rund 0,97 GiB für die weiteren Rechenpuffer hinzu. Meine Vorhersage für die beiden großen Speicherreservierungen von GPT-2 Medium lautet also 6,25 GiB, mit den ungerundeten Werten gerechnet.

Der C/CUDA-Trainer hat auf einer RTX 4070 SUPER 64 Updates ausgeführt. Seine beiden Reservierungen ergaben die vorausgesagten 6,25 GiB. Eine zweite Messung beobachtete die gesamte Grafikkarte von außen. Vor dem Start waren dort bereits 1.450 MiB belegt, der höchste beobachtete Wert lag bei 8.111 MiB oder 7,92 GiB. Der Unterschied zum Ausgangswert beträgt 6,50 GiB. Im Standbild ist der höchste Messpunkt mit einem Dreieck markiert. Welche der drei Zahlen beschreibt die reservierten Trainerpuffer, welche den Anstieg und welche die ganze Karte?

[Standbild mit den drei Messgrößen](../results/gpt2-medium-355m/memory.svg) · [Speicherverlauf als GIF](../results/gpt2-medium-355m/memory.gif) · [Messreihe der Karte](../results/gpt2-medium-355m/board-memory.csv) · [Messpunkte des Trainers](../results/gpt2-medium-355m/memory.csv) · [Versuch mit Rechenweg](lernskript/07-gpt2-medium.md)
