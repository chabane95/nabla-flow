# Vom nächsten Byte zu GPT-2

Im Testprotokoll steht `Fach A: Glas. Fach B: Harz.` Zu dieser Belegung gibt es zwei Fragen. Der gespeicherte Decoder mit 35.776 Parametern gibt bei beiden dieselbe Antwort.

| Satzanfang | Erwartete Fortsetzung | Ausgabe des Modells |
| --- | --- | --- |
| `In A liegt …` | `Glas.` | `Glas.` |
| `In B liegt …` | `Harz.` | `Glas.` |

Bei der ersten Frage könnte das Modell das richtige Fach gefunden oder schlicht das erste Material wiederholt haben. Erst die zweite Frage unterscheidet diese Erklärungen für diese Belegung. Über den gesamten Testsatz stimmen 26 von 56 Antworten. Zugleich erreicht dieses Modell eine Perplexität von 1,270. Wie passen die beiden Ergebnisse zusammen? [Testfragen](../../data/recall.tsv) · [Antwortwertung](../../results/recall.csv) · [Verlust und Perplexität](../../results/test.csv)

Der Verlust mittelt über alle Zielpositionen. Bekommt der Punkt am Satzende eine hohe Wahrscheinlichkeit, trägt er wenig zum Mittelwert bei. Für einen Treffer muss hingegen das ganze Wort samt Punkt stimmen. Bei der zweiten Frage genügt schon das erste `G` anstelle des `H`, damit die Antwort als falsch zählt. Was ändert sich, wenn beim Training jedes Byte des Antwortworts vierfach zählt? Dieser Versuch steht in Kapitel 5.

## Kapitel

| Kapitel | Rechnung oder Versuch |
| --- | --- |
| [1 · Vorhersage und Verlust](01-vorhersage.md) | Aus gezählten Byte-Paaren eine Wahrscheinlichkeit bestimmen. Was kostet es, wenn das beobachtete Byte nur Wahrscheinlichkeit ¼ hatte? |
| [2 · Der Vorwärtslauf](02-transformer.md) | Für eine Position aus Query und Keys die Attention-Gewichte berechnen. |
| [3 · Die Ableitung nach einem Gewicht](03-lernen.md) | Den Gradienten durch den Decoder verfolgen und mit einer kleinen Änderung eines Gewichts numerisch prüfen. |
| [4 · Messungen am trainierten Modell](04-labor.md) | Beide Fachfragen an den gespeicherten Decoder stellen und die Antworten mit dem Verlust derselben Testdaten vergleichen. |
| [5 · Vierfaches Gewicht für das Antwortwort](05-antwortgewicht.md) | Bei gleichen Startgewichten messen, welche Farb- und Materialbelegungen das Modell nach 1.800 Updates vollständig löst. |
| [6 · Vom Byte-Modell zu GPT-2](06-modellgroesse.md) | Parameter zählen und den Speicher für Gewichte, Gradienten und Optimierer berechnen. |
| [7 · GPT-2 Medium auf der RTX 4070 SUPER](07-gpt2-medium.md) | Die Speicherrechnung für 355 Millionen Parameter mit der gemessenen Belegung der Grafikkarte vergleichen. |

Die Herleitungen setzen bedingte Wahrscheinlichkeit, Skalarprodukt und Kettenregel voraus. Die Versuche am Byte-Modell laufen mit einem C11-Compiler auf der CPU. Für den Vergleich in Kapitel 7 stehen die Messwerte des GPU-Laufs als CSV bereit. [Mathematische Grundlagen](../../latex/backprop_first_principles.tex) · [GPU-Messreihe](../../results/gpt2-medium-355m/board-memory.csv)
