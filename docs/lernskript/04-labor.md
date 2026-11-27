# 4 · Das Modell als Experiment

Im Fach A liegt Glas, im Fach B Harz. Wenn man im letzten Satz nur den Buchstaben `A` durch `B` ersetzt, muss sich die richtige Antwort ändern. Ist genau das der Effekt, den ein niedriger Trainingsverlust misst? Du kannst diese Frage mit dem mitgelieferten Modell beantworten. Vorher prüfen wir, ob der Decoder überhaupt unter den Bedingungen rechnet, die seine Herleitung voraussetzt.

Die folgenden PowerShell-Blöcke laufen der Reihe nach vom Stammverzeichnis des Repositorys aus. Für den Windows-Build unten brauchst du GCC mit C11-Unterstützung. Die CPU-Versuche verwenden die mitgelieferten Daten. Für den großen Lauf werten wir die gespeicherten Messreihen aus.

## 4.1 Kann die Zukunft eine frühere Vorhersage verändern?

**Vorhersage.** Sehen zwei Eingaben bis einschließlich Position $t$ gleich aus, dann müssen ihre Logits an allen Positionen $0,\ldots,t$ übereinstimmen. Das gilt für zufällige Gewichte genauso wie für trainierte. Der kausale Decoder hat keinen Rechenpfad von einem späteren Token zu einem früheren Logit.

```powershell
$labor = Join-Path (Get-Location).Path ('build-lernlabor-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $labor | Out-Null
$nf = Join-Path $labor 'nablaflow.exe'
gcc -O3 -std=c11 -Wall -Wextra -Wpedantic src/main.c src/gpt.c src/data.c src/checkpoint.c src/recall.c src/scale.c -o $nf -lm
if ($LASTEXITCODE) { throw 'Der C-Build ist fehlgeschlagen.' }
& $nf check
if ($LASTEXITCODE) { throw 'Ein Selbsttest ist fehlgeschlagen.' }
```

Auf Linux oder macOS baut `make` dieselbe CPU-Anwendung. Ihr Selbsttest läuft mit `./nablaflow check`. Die Rechen- und Auswertungsbefehle in den folgenden Blöcken verwenden PowerShell.

Der Test füttert ein kleines Modell erst mit `0 1 2 3 4`, dann mit `0 1 2 8 9` und schließlich mit `0 1 2`. Er vergleicht *alle* Logits der ersten drei Positionen bitweise. Schon eine Abweichung würde die Vorhersage für diese Eingaben widerlegen. Im geprüften CPU-Build bestanden alle Vergleiche. Der separate Gradiententest über 1.128 Parameter ergab eine maximale absolute Abweichung von $6{,}78\cdot10^{-5}$. [Tests `test_causality` und `test_gradients`](../../src/gpt.c)

**Ein Blick in einen trainierten Kopf.** Der Prompt `Links: rot.` enthält elf ASCII-Bytes. Mit dem Startsymbol erwarten wir zwölf Zeilen zu je zwölf Gewichten. Welche Summe muss jede Zeile haben?

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

Alle 66 Gewichte oberhalb der Diagonalen waren null. Die größte gemessene Abweichung der Zeilensumme von eins betrug $4{,}5\cdot10^{-8}$. Die Exportfunktion füllt die ausgeblendeten Zellen selbst mit Nullen auf. Erst der Eingriff in die *Eingabe* und der Vergleich der *Logits* prüfen, ob der gesamte Decoder die Kausalität einhält. [Gespeicherte Matrix](../../results/attention.csv) · [Exportfunktion `gpt_attention_row`](../../src/gpt.c)

## 4.2 Zwanzig Updates unter dem Mikroskop

Ein gleichverteiltes Modell gibt dem richtigen Ziel $1/258$ Wahrscheinlichkeit. Auf Papier folgt daraus $-\ln(1/258)=5{,}553$ Nats pro Ziel. Zufällige Anfangsgewichte erzeugen leicht verschiedene Logits. Erwarte deshalb zu Beginn einen Wert in der Nähe dieser Zahl. Schreibe vor dem Start auf, ob du nach nur 20 Updates schon die Bigram-Tabelle schlagen würdest.

```powershell
$run = Join-Path $labor 'kurzlauf'
& $nf train --train data/train.txt --val data/val.txt --out $run --steps 20 --ctx 64 --width 16 --heads 2 --layers 1 --batch 4 --lr 0.002 --eval-every 5 --seed 42
if ($LASTEXITCODE) { throw 'Das kurze Training ist fehlgeschlagen.' }
Import-Csv (Join-Path $run 'history.csv') | Format-Table step,train_nll,val_nll
& $nf eval --model (Join-Path $run 'model.nfg') --data data/val.txt
```

Der hier ausgeführte Kurzlauf hatte 8.464 Parameter. Die Validierungs-NLL sank von 5,5476 auf 4,9881. Das kleinste gemessene Ergebnis lag bei Schritt 20. Das erneute Laden der Gewichte ergab 4,9881 auf denselben 2.496 Validierungszielen. Das Bigram kam mit seinen Häufigkeiten aus dem **Trainingssatz** schon auf 1,3375. Der kurze Lauf zeigt Lernen und liefert zugleich ein Beispiel dafür, weshalb der Anfangswert allein ein zu leichter Vergleich ist. Der Ausgabeordner für einen weiteren Lauf braucht einen neuen Namen. [Training, Messzeitpunkte und Auswahl der Gewichte](../../src/main.c) · [Bigram-Zählung](../../src/data.c)

## 4.3 Sagt das trainierte Labormodell das richtige Fach voraus?

Hier laden wir das umfangreicher trainierte, mitgelieferte Modell mit 35.776 Parametern. Für Training, Auswahl der Gewichte und Abschlusstest wurden ganze *ungeordnete* Wortpaare getrennt. Jede ausgewählte Belegung kommt im Test mit zwei Fragen vor. Insgesamt sind es 56 Aufgaben. [Generator und Aufteilung](../../tools/corpus.c) · [Testfragen mit Sollantworten](../../data/recall.tsv)

```powershell
& $nf eval --model results/model.nfg --data data/test.txt --train data/train.txt
& $nf recall results/model.nfg data/recall.tsv
& $nf sample --model results/model.nfg --prompt 'Fach A: Glas. Fach B: Harz. In A liegt ' --temperature 0 --tokens 8
& $nf sample --model results/model.nfg --prompt 'Fach A: Glas. Fach B: Harz. In B liegt ' --temperature 0 --tokens 8
```

* **Nächster Token.** Bei `eval` bekommt das Modell für jeden Schritt die tatsächlichen vorhergehenden Tokens im Block. Auf 2.490 Zielen ergibt sich eine NLL von 0,23928 und damit eine Perplexität von 1,27034. Das Bigram erreicht 3,68206. Leerzeichen, Satzmuster, Namen und Endmarker gehen alle in diesen Mittelwert ein. [Gespeicherte Testmessung](../../results/test.csv)
* **Ganze Antwort.** `recall` wählt pro Schritt das wahrscheinlichste der 258 Symbole und bewertet die Antwort bis einschließlich Punkt. Ein einziger falscher Schritt kostet den Treffer. Der mitgelieferte Checkpoint erreicht beim erneuten Ausführen 26/56. Die Regel „wiederhole den ersten genannten Wert“ trifft bei den beiden möglichen Fragen immer genau eine Antwort, insgesamt also 28/56. [Gespeicherter Recall](../../results/recall.csv) · [Bewertung der Antworten](../../src/recall.c)
* **Einzelfall.** Die beiden Beispielprompts ergeben jeweils `Glas.`. Beim Wechsel zu B wäre `Harz.` richtig. `sample` mit Temperatur null wählt unter den zulässigen, im Training beobachteten Ausgabebytes und dem Endsymbol. Das eignet sich zum Lesen einzelner Antworten. Für alle folgenden Zählungen verwenden wir die feste Definition aus `recall`. [Byte-Sampler](../../src/main.c)

**Gegenprobe mit konstant gehaltener Belegung.** Die Gesamtzahl 26 lässt offen, ob das Modell jemals *beide* Fragen zur selben Belegung richtig beantwortet. Formuliere zwei Vorhersagen. Ein Modell, das die Zuordnung auf jedem Beispiel beherrscht, sollte 28 Doppel-Treffer erzielen. Ein Modell, das unabhängig von der Frage immer den ersten Wert kopiert, sollte null erzielen. Im TSV stehen die zwei Gegenfragen unmittelbar nacheinander.

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

Die erneute Messung ergibt **5, 20, 3**. Die Summen stimmen mit den Einzelmessungen überein. Es sind $5+20+3=28$ Belegungen und $20+2\cdot3=26$ Treffer. In drei Belegungen führt die geänderte Frage zur jeweils richtigen, unterschiedlichen Antwort. Die fünf vollständigen Fehlschläge widerlegen die Erklärung, der Decoder kopiere ausnahmslos den ersten Wert. Welche interne Rechnung die drei Erfolge ermöglicht, bleibt offen.

**Ablation.** Entferne aus jedem Prompt die beiden Zuweisungen. Frage und Sollantwort bleiben identisch, derselbe Checkpoint beantwortet dieselben 56 Aufgaben. Überlege vorab, wie viele verschiedene Prompts danach noch übrig sind.

```powershell
$short = $rows -replace '^[^.]+\. [^.]+\. ', ''
$file = Join-Path $labor 'ohne-belegung.tsv'
[IO.File]::WriteAllLines($file, [string[]]$short, $utf8)
& $nf recall results/model.nfg $file
```

Es bleiben vier Frageformen. Bei unveränderten Sollantworten sinkt die Zahl der Treffer durch das Entfernen der Belegung von 26 auf 0 bei 56 Aufgaben. Gleichzeitig ändern sich Länge, Position der Frage und Satzform. Die Ablation allein kann den Rückgang keinem bestimmten Attention-Kopf zuschreiben. Beide Zusatzauswertungen beschreiben den bereits veröffentlichten Checkpoint auf seinem bekannten Testset. Für die Auswahl einer neuen Modellvariante gehören solche Vergleiche auf die Validierungsdaten.

## 4.4 Shakespeare und die Einheit des Vergleichs

Die folgende Tabelle verwendet die **gespeicherten Referenzläufe**. Die Test-NLL ist jeweils die mittlere Überraschung des tatsächlich folgenden Tokens unter seinem vorhandenen Kontext. Die Referenzmodelle beziehen ihre Häufigkeiten aus dem zugehörigen Trainingstext. Innerhalb jeder Zeile werden dieselben Zielpositionen gewertet.

| Testkorpus | Einheit und Anzahl der Ziele | Referenzmodell, NLL | Transformer, NLL | Gewählter Schritt |
| --- | --- | ---: | ---: | ---: |
| Deutsche Laborprotokolle | Bytes und Satzendmarker, 2.490 | Bigram 1,30347 | 0,23928 | 1.000 |
| *Coriolanus*-Auszug | Bytes und Passagenendmarker, 6.819 | Bigram 2,36989 | 1,94319 | 1.700 |
| *Tiny Shakespeare*, GPT-2 Small | GPT-2-BPE-Tokens einschließlich EOT, 13.564 | Unigramm 6,68371 | 5,17077 | 2.400 |

Die Versuchsprotokolle dokumentieren die Auswahl der Texte, die Trainingsbedingungen und die Testwerte. [Labormodell](../byte-models.md) · [Kleines Shakespeare-Modell](../../results/shakespeare/README.md) · [GPT-2 Small](../training-124m.md)

**Übung 1. Welcher Trainingsschritt zählt?** Im gespeicherten Laborlauf sinkt die NLL auf der festen Trainingsstichprobe zwischen Schritt 1.000 und 1.800 von 0,1660 auf 0,1498. Die Validierungs-NLL steigt von 0,2374 auf 0,2490. Mit welchen Gewichten würdest du den Test ausführen? [Vollständige Messreihe](../../results/history.csv)

**Auflösung.** Mit denen aus Schritt 1.000. Das war das Minimum auf der vollständigen Validierungsmenge unter den geprüften Schritten. Die Testdaten bleiben bei dieser Entscheidung außen vor. Bei GPT-2 lässt sich dieselbe Auswahl aus der Datei rekonstruieren und mit dem gespeicherten Nachladen vergleichen.

```powershell
Import-Csv results/gpt2-124m/history.csv | Sort-Object { [double]$_.val_nll } | Select-Object -First 1 | Format-Table step,tokens,val_nll
Import-Csv results/gpt2-124m/validation-reload.csv | Format-Table targets,nll,perplexity
```

Das Minimum und die Auswertung nach dem Laden ergeben beide 4,14468463 Nats pro BPE-Ziel. Bei Schritt 2.400 wurden $2400\cdot4\cdot128=1\,228\,800$ Trainingsziele verarbeitet. Fenster können überlappen und Tokens mehrfach vorkommen. Im Test-Split stehen 13.565 IDs hintereinander. Jeder Übergang liefert ein Ziel, also $13\,565-1=13\,564$. [GPU-Fenster und Auswertung](../../tools/nfgpu.c) · [Datenaufteilung](../../results/gpt2-124m/data.json)

**Übung 2. Wie viel gewinnt ein Modell gegenüber seiner Referenz?** Drücke für *Coriolanus* den Gewinn in Bits pro Ziel aus. Bestimme für GPT-2 das Verhältnis der beiden Test-Perplexitäten, ohne zuerst beide Perplexitäten auszurechnen.

**Auflösung.** Ein Bit entspricht $\ln 2$ Nats. Der *Coriolanus*-Gewinn beträgt $(2{,}36989-1{,}94319)/\ln 2=0{,}616$ Bits pro Byte- oder Endmarkerziel. Für GPT-2 folgt aus $\mathrm{PPL}=e^{\mathcal L}$ das Verhältnis $e^{6{,}68371-5{,}17077}=4{,}54$. So viel größer ist auch das geometrische Mittel der Wahrscheinlichkeit, die GPT-2 den richtigen Testtokens gibt, gegenüber dem Unigramm. Für einen direkten Größenvergleich zwischen Byte-Modell und GPT-2 bräuchten wir denselben Text, vergleichbare Kontextgrenzen und eine gemeinsame Bezugsgröße, etwa die gesamte negative Log-Wahrscheinlichkeit in Bits pro Originalbyte bei festgelegter Behandlung der Endmarker. [Byte-Testwerte](../../results/shakespeare/test.csv) · [BPE-Testwerte](../../results/gpt2-124m/test.csv) · [Unigramm](../../results/gpt2-124m/unigram-test.csv)

**Offene Messfrage.** Warum ist der GPT-2-Testverlust höher als der Validierungsverlust? Die in Quellreihenfolge gebildeten Splits enthalten unterschiedlich viele Reden mit im Training ungesehenen Sprecherüberschriften, 86/361 in der Validierung und 290/362 im Test. Das macht die Sprecherzeilen zu einer prüfbaren Hypothese. Werte als Folgeexperiment dieselben Modellvorhersagen einmal über die Überschrift und einmal über den restlichen Redetext aus. Pro Gruppe brauchst du die Summe der negativen Log-Wahrscheinlichkeiten und die Zahl der Ziele. Bleibt der Unterschied auch im Redetext groß, reichen die Überschriften als Erklärung nicht aus. [Metadaten und genaue Zählregel](../../results/gpt2-124m/data.json)

[Weiter zum kontrollierten Versuch mit vierfachem Antwortgewicht](05-antwortgewicht.md) · [Übersicht](README.md)
