# 4 · Das Modell als Experiment

In Fach A liegt Glas, in Fach B liegt Harz. Aus „In A liegt …“ wird „In B liegt …“. Ein Buchstabe in der Frage entscheidet darüber, welches ganze Wort die richtige Antwort ist. An dieser Belegung können wir die Wahrscheinlichkeit jedes nächsten Bytes messen und zählen, wie viele der beiden Antworten vollständig stimmen.

Die CPU-Versuche verwenden die Daten und Gewichte im Repository. Die Befehle laufen in derselben PowerShell-Sitzung aus dem Projektverzeichnis. Für den Windows-Build genügt GCC mit C11-Unterstützung. Der Vergleich mit GPT-2 Small beruht auf den gespeicherten Messreihen.

## 4.1 Kausalität des Decoders

Sind zwei Eingaben bis einschließlich Position $t$ identisch, müssen die Logits an allen Positionen $0,\ldots,t$ übereinstimmen. Im kausalen Decoder gibt es keinen Rechenpfad von einem späteren Token zu einem früheren Logit. Diese Vorhersage gilt für jede Wahl der Gewichte.

```powershell
$labor = Join-Path (Get-Location).Path ('build-lernlabor-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $labor | Out-Null
$nf = Join-Path $labor 'nablaflow.exe'
gcc -O3 -std=c11 -Wall -Wextra -Wpedantic src/main.c src/gpt.c src/data.c src/checkpoint.c src/recall.c src/scale.c -o $nf -lm
if ($LASTEXITCODE) { throw 'Der C-Build ist fehlgeschlagen.' }
& $nf check
if ($LASTEXITCODE) { throw 'Ein Selbsttest ist fehlgeschlagen.' }
```

Unter Linux und macOS lauten die entsprechenden Befehle `make` und `./nablaflow check`.

Der Test berechnet die Logits zuerst für `0 1 2 3 4`, dann für `0 1 2 8 9` und schließlich für `0 1 2`. An den ersten drei Positionen stimmten alle Logits in allen drei Rechnungen bitweise überein. Der separate Gradiententest über 1.128 Parameter ergab eine maximale absolute Abweichung von $6{,}78\cdot10^{-5}$. [Kausalitäts- und Gradiententest](../../src/gpt.c)

**Messung an einem trainierten Kopf.** Die elf ASCII-Bytes von `Links: rot.` und das Startsymbol ergeben zwölf Positionen. Der Export enthält also zwölf Zeilen zu je zwölf Gewichten. Die Softmax verlangt Zeilensumme eins. Die Maske schließt die $12\cdot11/2=66$ Positionen oberhalb der Diagonalen aus.

```powershell
$a = & $nf inspect --model results/model.nfg --prompt 'Links: rot.' --layer 1 --head 0 | ConvertFrom-Csv
if ($LASTEXITCODE -or $a.Count -ne 144) { throw 'Der Attention-Export ist unvollständig.' }
$future = @($a | Where-Object { [int]$_.key_pos -gt [int]$_.query_pos -and [double]$_.weight -ne 0 }).Count
$errors = $a | Group-Object query_pos | ForEach-Object {
    $sum = ($_.Group | ForEach-Object { [double]$_.weight } | Measure-Object -Sum).Sum
    [Math]::Abs($sum - 1)
}
$future
($errors | Measure-Object -Maximum).Maximum
```

Alle 66 Einträge oberhalb der Diagonalen waren null. Die größte Abweichung einer Zeilensumme von eins betrug $4{,}5\cdot10^{-8}$. Die Exportfunktion trägt die Nullen selbst ein. Der Test mit dem veränderten Satzende prüft die stärkere Aussage, dass der spätere Text auch am Ausgang des gesamten Decoders keinen Einfluss hat. [Gemessene Attention-Matrix](../../results/attention.csv) · [Exportfunktion `gpt_attention_row`](../../src/gpt.c)

## 4.2 Zwanzig Trainingsschritte

Ein gleichverteiltes Modell gibt jedem Ziel die Wahrscheinlichkeit $1/258$. Sein Verlust beträgt $-\ln(1/258)=5{,}553$ Nats pro Ziel. Die kleinen zufälligen Anfangsgewichte des Decoders streuen die Logits um null. Zu Beginn erwarten wir einen Verlust nahe 5,553. Einen zweiten Vergleichswert liefert eine Bigramm-Tabelle, die aus den Trainingsdaten die Übergänge zwischen benachbarten Tokens zählt.

```powershell
$run = Join-Path $labor 'kurzlauf'
& $nf train --train data/train.txt --val data/val.txt --out $run --steps 20 --ctx 64 --width 16 --heads 2 --layers 1 --batch 4 --lr 0.002 --eval-every 5 --seed 42
if ($LASTEXITCODE) { throw 'Das kurze Training ist fehlgeschlagen.' }
Import-Csv (Join-Path $run 'history.csv') | Format-Table step,train_nll,val_nll
& $nf eval --model (Join-Path $run 'model.nfg') --data data/val.txt
```

Der Decoder mit 8.464 Parametern startete bei 5,5476 Nats pro Validierungsziel. Nach 20 Schritten lag er bei 4,9881, dem niedrigsten gemessenen Wert. Das erneute Laden der Gewichte ergab auf denselben 2.496 Zielen wieder 4,9881. Die Bigramm-Tabelle erreichte dort 1,3375. Der Decoder hat also im Mittel höhere Wahrscheinlichkeiten für die beobachteten Tokens gelernt, liegt nach zwanzig Schritten aber noch weit hinter der Tabelle. [Training und Auswahl der Gewichte](../../src/main.c) · [Bigramm-Zählung](../../src/data.c)

## 4.3 Ein richtiges Byte und eine richtige Antwort

Das gespeicherte Labormodell hat 35.776 Parameter. Der Datengenerator teilt nach *ungeordneten* Wortpaaren auf. „Glas und Harz“ gehört also mit beiden Anordnungen zu genau einem Split. Im Test gibt es 28 Belegungen mit jeweils zwei Gegenfragen, insgesamt 56 Aufgaben. [Generator und Aufteilung](../../tools/corpus.c) · [Testfragen mit Sollantworten](../../data/recall.tsv)

```powershell
& $nf eval --model results/model.nfg --data data/test.txt --train data/train.txt
& $nf recall results/model.nfg data/recall.tsv
& $nf sample --model results/model.nfg --prompt 'Fach A: Glas. Fach B: Harz. In A liegt ' --temperature 0 --tokens 8
& $nf sample --model results/model.nfg --prompt 'Fach A: Glas. Fach B: Harz. In B liegt ' --temperature 0 --tokens 8
```

* **Tokenwahrscheinlichkeit.** `eval` gibt bis zu jeder Zielposition die im Text tatsächlich vorkommenden Tokens vor. Über die 2.490 Ziele beträgt die NLL 0,23928, die Perplexität 1,27034. Die Bigramm-Tabelle kommt auf eine Perplexität von 3,68206. Leerzeichen und Endmarker gehen ebenso in den Mittelwert ein wie die Buchstaben des gesuchten Wortes. [Testmessung](../../results/test.csv)
* **Vollständige Antwort.** `recall` erzeugt bei jedem Schritt das wahrscheinlichste der 258 Symbole. Bis einschließlich Punkt müssen alle Bytes stimmen. Das Modell erfüllt diese Bedingung bei 26 von 56 Fragen. Wer stets den zuerst genannten Wert kopiert, löst bei jeder Belegung genau eine der beiden Fragen und erreicht 28 von 56. [Gemessene Treffer](../../results/recall.csv) · [Bewertung der Antworten](../../src/recall.c)
* **Glas und Harz.** Bei Temperatur null ergänzt `sample` beide Fragen mit `Glas.`. Für Fach B lautet die richtige Antwort `Harz.`. Beim freien Schreiben lässt `sample` nur zulässige, im Training beobachtete Ausgabebytes und das Endsymbol zu. Für die Trefferzahlen gilt durchgehend die Auswahlregel von `recall`. [Byte-Sampler](../../src/main.c)

**Gegenprobe.** Wie oft stimmen *beide* Antworten zur selben Belegung? Eine vollständige Zuordnung ergäbe 28 Doppeltreffer, das Kopieren des ersten Werts null. In der TSV-Datei stehen die Gegenfragen zu jeder Belegung unmittelbar nacheinander.

```powershell
$rows = @(Get-Content -Encoding UTF8 data/recall.tsv)
$utf8 = [Text.UTF8Encoding]::new($false)
$paired = Join-Path $labor 'paar.tsv'
$counts = @(0, 0, 0)
for ($i = 0; $i -lt $rows.Count; $i += 2) {
    [IO.File]::WriteAllLines($paired, [string[]]$rows[$i..($i+1)], $utf8)
    $score = & $nf recall results/model.nfg $paired | ConvertFrom-Csv
    if ($LASTEXITCODE -or [int]$score.records -ne 2) { throw 'Fehlendes Fragenpaar.' }
    $counts[[int]$score.exact]++
}
$counts    # Belegungen mit null, einer und zwei richtigen Antworten
```

Bei **5 Belegungen** sind beide Antworten falsch, bei **20** ist eine richtig und bei **3** sind beide richtig. Das ergibt $5+20+3=28$ Belegungen und $20+2\cdot3=26$ richtige Einzelantworten. Bei den drei Doppeltreffern führt allein der Wechsel der Frage zur passenden, jeweils anderen Antwort.

**Versuch ohne Belegung.** Aus jeder Testzeile entfallen beide Zuweisungen. Übrig bleiben die Frage und ihre ursprüngliche Sollantwort. Die Gewichte des Modells bleiben fest.

```powershell
$short = $rows -replace '^[^.]+\. [^.]+\. ', ''
$file = Join-Path $labor 'ohne-belegung.tsv'
[IO.File]::WriteAllLines($file, [string[]]$short, $utf8)
& $nf recall results/model.nfg $file
```

Die 56 gekürzten Prompts haben vier verschiedene Formen. Das Modell trifft jetzt **0 von 56** Antworten. Mit der Belegung fehlen auch der vertraute Satzanfang und die bisherige Position der Frage. Aus diesem Eingriff lässt sich daher kein einzelner Attention-Kopf als Ursache bestimmen. Beide Versuche beschreiben den gespeicherten Checkpoint auf seinem bekannten Testsatz.

## 4.4 Shakespeare und die Einheit des Vergleichs

Die Test-NLL der drei Referenzläufe ist jeweils die mittlere negative Log-Wahrscheinlichkeit des beobachteten Tokens bei gegebener Vorgeschichte. Die Häufigkeiten für Bigramm und Unigramm stammen aus dem zugehörigen Trainingstext. Innerhalb jeder Tabellenzeile sind die gewerteten Zielpositionen identisch.

| Testkorpus | Einheit und Anzahl der Ziele | Referenzmodell, NLL | Transformer, NLL | Gewählter Schritt |
| --- | --- | ---: | ---: | ---: |
| Deutsche Laborprotokolle | Bytes und Satzendmarker, 2.490 | Bigramm 1,30347 | 0,23928 | 1.000 |
| *Coriolanus*-Auszug | Bytes und Passagenendmarker, 6.819 | Bigramm 2,36989 | 1,94319 | 1.700 |
| *Tiny Shakespeare*, GPT-2 Small | GPT-2-BPE-Tokens einschließlich EOT, 13.564 | Unigramm 6,68371 | 5,17077 | 2.400 |

Die beiden Shakespeare-Läufe verwenden verschiedene Textausschnitte und verschiedene Tokenisierungen. [Labormodell](../byte-models.md) · [Kleines Shakespeare-Modell](../../results/shakespeare/README.md) · [GPT-2 Small](../training-124m.md)

**Übung 1. Auswahl der Gewichte.** Zwischen Schritt 1.000 und 1.800 sinkt im Laborlauf die NLL auf der festen Trainingsstichprobe von 0,1660 auf 0,1498. Die Validierungs-NLL steigt von 0,2374 auf 0,2490. Welcher der beiden Stände wird für den Test ausgewählt, wenn die Validierungs-NLL entscheidet? [Vollständige Messreihe](../../results/history.csv)

Schritt 1.000 hat den kleineren Validierungsverlust und erreicht auch unter allen Messpunkten das Minimum. Für GPT-2 Small lässt sich die Auswahl mit der Auswertung des neu geladenen Checkpoints vergleichen.

```powershell
Import-Csv results/gpt2-124m/history.csv | Sort-Object { [double]$_.val_nll } | Select-Object -First 1 | Format-Table step,tokens,val_nll
Import-Csv results/gpt2-124m/validation-reload.csv | Format-Table targets,nll,perplexity
```

Das Minimum und die Auswertung nach dem Laden ergeben beide 4,14468463 Nats pro BPE-Ziel. Bei Schritt 2.400 wurden $2400\cdot4\cdot128=1\,228\,800$ Trainingsziele verarbeitet. Fenster können überlappen und Tokens mehrfach vorkommen. Im Test-Split stehen 13.565 IDs hintereinander. Jeder Übergang liefert ein Ziel, also $13\,565-1=13\,564$. [GPU-Fenster und Auswertung](../../tools/nfgpu.c) · [Datenaufteilung](../../results/gpt2-124m/data.json)

**Übung 2. Wie viel gewinnt ein Modell gegenüber seiner Referenz?** Drücke für *Coriolanus* den Gewinn in Bits pro Ziel aus. Bestimme für GPT-2 das Verhältnis der beiden Test-Perplexitäten, ohne zuerst beide Perplexitäten auszurechnen.

**Auflösung.** Ein Bit entspricht $\ln 2$ Nats. Der *Coriolanus*-Gewinn beträgt $(2{,}36989-1{,}94319)/\ln 2=0{,}616$ Bits pro Byte- oder Endmarkerziel. Für GPT-2 folgt aus $\mathrm{PPL}=e^{\mathcal L}$ das Verhältnis $e^{6{,}68371-5{,}17077}=4{,}54$. So viel größer ist auch das geometrische Mittel der Wahrscheinlichkeit, die GPT-2 den richtigen Testtokens gibt, gegenüber dem Unigramm. Für einen direkten Größenvergleich zwischen Byte-Modell und GPT-2 bräuchten wir denselben Text, vergleichbare Kontextgrenzen und eine gemeinsame Bezugsgröße, etwa die gesamte negative Log-Wahrscheinlichkeit in Bits pro Originalbyte bei festgelegter Behandlung der Endmarker. [Byte-Testwerte](../../results/shakespeare/test.csv) · [BPE-Testwerte](../../results/gpt2-124m/test.csv) · [Unigramm](../../results/gpt2-124m/unigram-test.csv)

**Hypothese zum Unterschied zwischen Validierung und Test.** Die Abschnitte folgen der Reihenfolge im Quelltext. In der Validierung tragen 86 von 361 Reden eine im Training ungesehene Sprecherüberschrift, im Test 290 von 362. Ein höherer Verlust auf den Überschriften könnte deshalb einen Teil des höheren Testverlusts erklären. Eine Gegenprobe würde für Überschrift und gesprochenen Text getrennt die Summe der negativen Log-Wahrscheinlichkeiten und die Zahl der Ziele erfassen. Ein großer verbleibender Unterschied im gesprochenen Text würde eine Erklärung allein aus den Überschriften ausschließen. [Metadaten und Zählregel](../../results/gpt2-124m/data.json)

[Weiter zum kontrollierten Versuch mit vierfachem Antwortgewicht](05-antwortgewicht.md) · [Übersicht](README.md)
