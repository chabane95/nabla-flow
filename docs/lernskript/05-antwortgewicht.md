# 5 · Was ändert sich, wenn das Antwortwort vierfach zählt?

`Fach A: Glas. Fach B: Harz. In B liegt Harz.` ist gleichzeitig ein Sprachbeispiel und eine Zuordnungsaufgabe. Das erste `Harz` legt die Belegung fest, das letzte `Harz` beantwortet die Frage. Im bisherigen Training tragen die Buchstaben an beiden Stellen mit demselben Gewicht zum Verlust bei. Unsere Hypothese ist, dass stärker gewichtete Fehler **beim Antworten** das Modell empfindlicher für die Frage machen. Das lässt sich prüfen, indem wir dieselbe Belegung einmal nach A und einmal nach B abfragen.

## Ein Eingriff in die Zielfunktion

Für Zielbyte $y_i$ mit dem sichtbaren Kontext $c_i$ beträgt der Verlust $\ell_i=-\ln p_\theta(y_i\mid c_i)$. Im Kontrolllauf A hat jedes Ziel das Gewicht $w_i=1$. Im Versuch B erhält jedes UTF-8-Byte **im letzten Antwortwort** $w_i=4$. In unserem Beispiel sind das die vier Bytes des letzten `Harz`. Die frühere Erwähnung, das vorangestellte Leerzeichen, der abschließende Punkt und der Endmarker behalten Gewicht eins. Ein `ü` im Antwortwort belegt zwei Bytes, die beide vierfach zählen. [Markierung der Antwortbytes](../../src/data.c)

Für die Zielpositionen eines Batches berechnet das Training

$$
L_B=\frac{\sum_i w_i\ell_i}{\sum_i w_i},
\qquad
\frac{\partial L_B}{\partial z_{ik}}
=\frac{w_i}{\sum_j w_j}\bigl(p_\theta(k\mid c_i)-\mathbf 1_{k=y_i}\bigr).
$$

Hier ist $z_{ik}$ der Logit der Klasse $k$ an Position $i$. Die Gewichte ändern den relativen Beitrag der Zielpositionen zum Gradienten. Der Nenner hält die Rechnung als gewichteten Mittelwert normiert. Im gesamten Trainingskorpus sind 1.000 von 10.998 Zielen Bytes des Antwortworts. Ihr Anteil am Gesamtgewicht wächst von $1\,000/10\,998=9{,}09\,\%$ auf $4\,000/(9\,998+4\,000)=28{,}58\,\%$. Beim Training wird der Nenner für jeden einzelnen Batch gebildet. [Trainingskorpus](../../data/train.txt) · [Gradient und Optimierung](../../src/gpt.c)

## Was halten wir konstant?

Beide Läufe starten mit Seed 42 und demselben Byte-Decoder mit **35.776 Parametern**. Beide sehen dieselben 248 Trainingszeilen und dieselben Batches in derselben Reihenfolge. Die Reihenfolge wird mit einem eigenen Zufallszahlengenerator aus dem Seed bestimmt. Batchgröße 8, Kontext 64, Breite 32, vier Köpfe, zwei Blöcke, die AdamW-Einstellungen und der Lernratenverlauf mit Spitzenwert 0,002 sind gleich. Jeder Lauf macht 1.800 Updates. Die Auswertung misst in beiden Fällen den gewöhnlichen, ungewichteten Verlust. [Trainingsschleife](../../src/main.c)

Wir vergleichen die Gewichte zunächst **nach genau 1.800 Updates**. Zusätzlich untersuchen wir den jeweils gespeicherten Stand mit der niedrigsten ungewichteten Validierungs-NLL unter den Messpunkten bei Schritt 0 und danach alle 50 Schritte. Das ist dieselbe Auswahlregel wie im bisherigen Projekt. Die Paarwertung spielt bei dieser Auswahl keine Rolle.

Ein **Doppeltreffer** bedeutet, dass `recall` bei derselben Belegung beide Gegenfragen bis einschließlich Punkt exakt beantwortet. Das gilt für Materialfächer (A/B) und Farben (Links/Rechts). Die 56 Validierungsfragen bilden 28 Belegungen. Vertauschte Werte ergeben eine neue Belegung, daher beruhen diese 28 auf 14 ungeordneten Wortpaaren. Ein Modell, das immer den ersten Wert kopiert, trifft 28 einzelne Antworten und erzielt null Doppeltreffer. [Validierungszeilen](../../data/val.txt) · [Paarbildung und Bewertung](../../tools/recall_pairs.py)

## Messung

Die 35.776 Startparameter haben in beiden Läufen denselben FNV-1a-Fingerabdruck `aa301ceacc78564c`. Auch die ungewichtete Validierungs-NLL bei Schritt null ist mit 5,55346731407 identisch. Der nach Validierungsverlust ausgewählte Kontroll-Checkpoint ist bytegleich mit dem bisher veröffentlichten `results/model.nfg`. Bei einer Wiederholung mit dem unten angegebenen Rezept und GCC 14.2.0 unter Windows waren alle vier Checkpoint-Dateien sowie die Verlustkurven bytegleich mit denen des ersten Versuchs. [Protokoll 1×](../../results/answer-weight/1x/summary.csv) · [Protokoll 4×](../../results/answer-weight/4x/summary.csv)

| Vergleich | Antwortgewicht | Schritt | Beide richtig, Validierung | Einzelantworten, Validierung | Beide richtig, Test | Einzelantworten, Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Festes Trainingsende | 1× | 1.800 | 4/28 | 29/56 | 7/28 | 34/56 |
| Festes Trainingsende | 4× | 1.800 | **10/28** | **38/56** | 6/28 | 32/56 |
| Minimum der Validierungs-NLL | 1× | 1.000 | 0/28 | 20/56 | 3/28 | 26/56 |
| Minimum der Validierungs-NLL | 4× | 1.350 | **8/28** | **35/56** | **8/28** | **34/56** |

Bei festem Budget gewinnt der gewichtete Lauf auf der Validierung neun zuvor ungelöste Belegungen und verliert drei zuvor vollständig gelöste. Das ergibt netto sechs zusätzliche Doppeltreffer. Auf dem separaten Testsatz gewinnt er fünf und verliert sechs. Die Testdaten wurden nach dem Training mit den festgelegten Checkpoints ausgewertet. Sie waren aus früheren Projektversuchen bereits bekannt. [Alle acht Messungen einschließlich NLL](../../results/answer-weight/comparison.csv) · [Ergebnis jeder einzelnen Belegung](../../results/answer-weight/pairs.csv)

### Welche Aufgabe profitiert?

| Belegungstyp bei Schritt 1.800 | Validierung 1× → 4× | Test 1× → 4× |
| --- | ---: | ---: |
| Farben mit `Links` und `Rechts` | 1/14 → **10/14** | 3/14 → **6/14** |
| Materialien in `Fach A` und `Fach B` | 3/14 → **0/14** | 4/14 → **0/14** |

Bei `Links: beige. Rechts: lila.` beantwortet der Kontrolllauf eine Gegenfrage richtig, der gewichtete Lauf beide. Bei `Fach A: Blei. Fach B: Glas.` geschieht das Umgekehrte. Am festen Trainingsende löst das gewichtete Modell Farbzuordnungen häufiger und vollständige Materialzuordnungen seltener. Auch unter der Auswahl nach Validierungs-NLL stammen alle acht Doppeltreffer des gewichteten Modells je Split aus den Farbfragen. [Rohdaten nach Belegung](../../results/answer-weight/pairs.csv)

**Eine Frage für die Auswertung.** Die ungewichtete Test-NLL bei Schritt 1.800 sinkt mit der neuen Gewichtung von 0,24687 auf 0,22332. Wie kann gleichzeitig die Zahl der Test-Doppeltreffer von sieben auf sechs sinken?

<details>
<summary>Lösungsweg und Folgeexperiment</summary>

Die NLL mittelt über 2.490 Zielpositionen und berücksichtigt jede Änderung der dem richtigen Token zugewiesenen Wahrscheinlichkeit. Ein Doppeltreffer verlangt zwei fehlerfreie, selbst erzeugte Antworten. Schon ein falscher Anfangsbuchstabe bei einer der beiden Antworten kostet den ganzen Doppeltreffer. Die beiden Maße können sich deshalb in verschiedene Richtungen bewegen.

Wir haben einen Seed und pro Aufgabentyp nur 14 Validierungsbelegungen untersucht. Zudem kommen Farben stets mit `Links/Rechts` und Materialien stets mit `A/B` vor. Um zu prüfen, ob Wortschatz oder Frageform den Unterschied bewirkt, könnte man Farben *und* Materialien mit beiden Satzformen trainieren und alle vier Kombinationen getrennt auswerten. Eine Wiederholung mit weiteren gemeinsamen Startseeds würde zusätzlich zeigen, wie stark der beobachtete Effekt vom Start abhängt.

</details>

## Selbst wiederholen

Führe die folgenden PowerShell-Blöcke aus dem Stammverzeichnis aus. GCC mit C11-Unterstützung baut den CPU-Trainer. Python 3 genügt für die Paarwertung. Falls `python` auf deinem System ein Platzhalter ist, trage in `$py` den Pfad zum installierten Interpreter ein. Der Versuch legt einen neuen Ordner an, sodass die gespeicherten Referenzläufe erhalten bleiben.

~~~powershell
$runDir = New-Item -ItemType Directory -Path ('run-antwortgewicht-' + [Guid]::NewGuid().ToString('N'))
$nf = Join-Path $runDir.FullName 'nablaflow.exe'
$py = 'python'
gcc -O3 -std=c11 -Wall -Wextra -Wpedantic src/main.c src/gpt.c src/data.c src/checkpoint.c src/recall.c src/scale.c -o $nf -lm
if ($LASTEXITCODE) { throw 'Build fehlgeschlagen' }
& $nf check
if ($LASTEXITCODE) { throw 'Selbsttest fehlgeschlagen' }

$settings = @('--train','data/train.txt','--val','data/val.txt','--steps','1800',
    '--ctx','64','--width','32','--heads','4','--layers','2','--batch','8',
    '--lr','0.002','--eval-every','50','--seed','42','--save-final','1')
foreach ($factor in 1,4) {
    $out = Join-Path $runDir.FullName ("{0}x" -f $factor)
    & $nf train @settings --answer-weight $factor --out $out
    if ($LASTEXITCODE) { throw "Training fehlgeschlagen (Faktor $factor)" }
    Import-Csv (Join-Path $out 'summary.csv') | Format-Table answer_weight,initial_parameters_fnv1a64,best_step
}
~~~

`final.nfg` enthält den Endstand, `model.nfg` den nach Validierungs-NLL ausgewählten Stand. Der folgende Block gibt für jeden Vergleich `both` (Doppeltreffer) und `exact` (korrekte Einzelantworten) als Teil der JSON-Ausgabe aus. Die CSV-Dateien erlauben den direkten Vergleich derselben Belegung über die beiden Läufe.

~~~powershell
foreach ($factor in 1,4) {
    $out = Join-Path $runDir.FullName ("{0}x" -f $factor)
    foreach ($state in 'final','model') {
        foreach ($split in 'val','test') {
            Write-Output "$factor x / $state / $split"
            $weights = Join-Path $out "$state.nfg"
            $detail = Join-Path $out "$state-$split.csv"
            & $py tools/recall_pairs.py --exe $nf --model $weights --data "data/$split.txt" --details $detail
            if ($LASTEXITCODE) { throw 'Paarwertung fehlgeschlagen' }
            & $nf eval --model $weights --data "data/$split.txt"
            if ($LASTEXITCODE) { throw 'NLL-Auswertung fehlgeschlagen' }
        }
    }
}
~~~

Für die Schlüsselmessung genügt das gespeicherte Modell. Mit `$py` und `$nf` aus dem Build-Abschnitt liefert `& $py tools/recall_pairs.py --exe $nf --model results/answer-weight/4x/final.nfg --data data/val.txt` direkt die zehn Doppeltreffer. [Gespeicherte Läufe und Gewichte](../../results/answer-weight/) · [Weiter zu Modellgröße, GPT-2 und GPT-3](06-modellgroesse.md) · [Übersicht](README.md)
