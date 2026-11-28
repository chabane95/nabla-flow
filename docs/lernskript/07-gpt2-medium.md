# 7 · GPT-2 Medium auf der RTX 4070 SUPER

**Wie viel Grafikspeicher braucht ein Trainingsschritt mit 355 Millionen Parametern?** Für vier Textfenster mit je 128 Tokens reservierte der Trainer **6,25 GiB** an Puffern. Auf der gesamten RTX 4070 SUPER waren zeitweise **7,92 GiB** belegt. Mit dieser Konfiguration absolvierte GPT-2 Medium 64 Trainingsschritte.

[![Speicherverlauf der gesamten Karte und Vergleich von Kartenanstieg, CUDA-Messung und berechneter Puffersumme.](../../results/gpt2-medium-355m/memory.png)](../../results/gpt2-medium-355m/memory.svg)

**Abbildung 1.** (a) Belegter Speicher der gesamten Karte über die Zeit. Von den 244 Messpunkten sind der Wert vor Prozessstart (Quadrat) und das beobachtete Maximum (Dreieck) markiert. (b) Speicherzuwachs in GiB mit eigener Skala. Das Dreieck gibt das Maximum aus (a) abzüglich des Anfangswerts an. Der offene Kreis zeigt den größten von CUDA gemeldeten Zuwachs gegenüber dem bereits aktiven Kontext vor dem Laden von cuBLAS. Die Raute markiert die Summe der beiden Puffergrößen im Trainer. [Druckfassung](../../results/gpt2-medium-355m/memory.pdf) · [Animation der 244 Messpunkte](../../results/gpt2-medium-355m/memory.gif)

Die Zeitachse beginnt beim Start des äußeren Messprogramms. Vor dem steilen Anstieg erzeugt der Trainer die Anfangsgewichte auf der CPU. Beim Beenden fällt die Belegung wieder ab. Die interne Trainingsuhr beginnt erst nach der GPU-Initialisierung. [Messpunkte der Karte](../../results/gpt2-medium-355m/board-memory.csv) · [CUDA-Messpunkte](../../results/gpt2-medium-355m/memory.csv) · [Versuchsprotokoll](../../results/gpt2-medium-355m/run.json)

## Vom einzelnen Gewicht zum Checkpoint

GPT-2 Medium hat $L=24$ Blöcke der Breite $d=1\,024$. Auf jeden der 16 Attention-Köpfe entfallen damit 64 Komponenten. Das MLP erweitert den Vektor jeder Position vorübergehend auf 4.096 Komponenten. Die Tokentabelle enthält einen Vektor für jedes der $V=50\,257$ Tokens, die Positionstabelle einen für jede der $C=1\,024$ Positionen. Das entspricht den veröffentlichten Abmessungen von GPT-2 Medium. Der Kurzlauf nutzt davon die ersten 128 Positionen je Fenster. Er beginnt mit den zufällig initialisierten Gewichten des Projekts. [Originalkonfiguration](https://openaipublic.blob.core.windows.net/gpt-2/models/355M/hparams.json) · [Konfiguration und Trainingsschleife](../../tools/nfgpu.c)

Die Parameterzahl setzt sich aus den beiden Tabellen, den 24 Blöcken und der letzten Normierung zusammen. Mit der Zählformel aus Kapitel 2 ergibt sich

$$
N=(V+C)d+L(12d^2+13d)+2d
=354\,823\,168.
$$

Ein Gewicht belegt in FP32 vier Byte. Alle Gewichte zusammen brauchen also $4N=1\,419\,292\,672$ Byte oder **1,322 GiB**. Das lässt sich an der Checkpoint-Datei prüfen. Sie ist genau 300 Byte größer und enthält neben den Gewichten unter anderem die Modellabmessungen und eine Prüfsumme. Ein GiB entspricht $2^{30}$ Byte, ein MiB $2^{20}$ Byte. [Parameterzählung](02-transformer.md#zählen-als-unabhängige-kontrolle) · [Checkpointformat](../../src/checkpoint.c)

## Was kommt beim Training zu den Gewichten hinzu?

Für einen Eintrag der Query-Matrix liest der Vorwärtslauf dessen Wert $w$. Der Rückwärtslauf liefert den Gradienten $g=\partial\mathcal L/\partial w$. Für das Update speichert AdamW außerdem zwei gleitende Mittelwerte, einen für $g$ und einen für $g^2$. Gewicht, Gradient und beide Momente belegen zusammen vier FP32-Zahlen, also 16 Byte je Parameter. Für das ganze Modell ergeben sich **$16N=5\,677\,170\,688$ Byte**.

Ein zweiter Puffer hängt von der Fensterlänge ab. Vier Fenster ($B=4$) mit je $T=128$ Tokens liefern 512 Positionen. Pro Position hält der Trainer einen Logit für jedes Token im Vokabular. Allein diese Werte brauchen $512\cdot50\,257\cdot4=102\,926\,336$ Byte. Bei der Attention trifft jede der $T$ Query-Positionen auf $T$ mögliche Key-Positionen. Selbst mit Kausalmaske reserviert dieser Trainer pro Kopf und Fenster das ganze $T\times T$-Feld. Bei $H=16$ Köpfen belegt es $4BHT^2$ Byte je Block. Das sind 4 MiB pro Block, also 96 MiB für die 24 Blöcke. [Pufferaufteilung in `allocate_state`](../../src/gpu.c) · [CUDA-Kernel](../../src/gpu_kernels.cu)

| Vom Trainer reservierter Speicher | Byte | GiB |
| --- | ---: | ---: |
| Gewichte, Gradienten und beide Adam-Momente | 5.677.170.688 | 5,2873 |
| Zwischenwerte, Logits und Arbeitsfelder | 1.038.462.976 | 0,9671 |
| **Summe** | **6.715.633.664** | **6,2544** |

Beide Puffer stehen schon vor dem ersten Trainingsschritt bereit. Jeder weitere Schritt nutzt denselben Platz. Das erklärt, weshalb die Belegung nach dem Anstieg in Abb. 1a über viele Schritte fast konstant bleibt.

## Warum liefern die Messungen verschiedene Werte?

| Messung | Bezugspunkt | Höchste beobachtete Belegung | Zunahme |
| --- | ---: | ---: | ---: |
| CUDA-Treiber, `cuMemGetInfo` | 1,202 GiB nach Anlage des CUDA-Kontexts | 7,471 GiB an den synchronisierten Messpunkten | **6,270 GiB** |
| `nvidia-smi`, gesamte Karte | 1.450 MiB vor Prozessstart | 8.111 MiB während des Laufs | **6.661 MiB = 6,505 GiB** |

Der Trainer wartet an ausgewählten Stellen, bis die GPU ihre Arbeit beendet hat. Dann fragt er den freien und gesamten Speicher beim CUDA-Treiber ab. Ihre Differenz ist die Belegung. Nach dem Laden der Rechenbibliotheken und dem Hochladen der Gewichte lag sie 6.420 MiB über dem ersten Wert beim Anlegen des Kontexts. Das sind rund 15,5 MiB mehr als die beiden reservierten Puffer zusammen. Bei den späteren Abfragen nach Validierung, Textgenerierung, Rückwärtslauf und Update meldete CUDA dieselbe Belegung. [Interne Rohmessung](../../results/gpt2-medium-355m/memory.csv) · [NVIDIA-Dokumentation zur CUDA-Speicherabfrage](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)

Das äußere Messprogramm fragt `nvidia-smi` in einem angestrebten Abstand von 100 ms ab. Seine 244 Messpunkte umfassen die gesamte Laufzeit und schließen den Speicher anderer Anwendungen ein. Beim höchsten Messwert von 8.111 MiB waren laut Kartenanzeige noch **3.889 MiB frei**. Nach Prozessende lag die Belegung bei 1.423 MiB. Der Bezugspunkt liegt hier vor dem Start des Trainers. Auch die Speicherabrechnung unterscheidet sich zwischen CUDA und Kartenanzeige. Kürzere Spitzen können zwischen zwei Abfragen liegen. [Äußeres Messprogramm](../../tools/measure_gpu_run.py) · [NVIDIA-Dokumentation zur Kartenbelegung](https://docs.nvidia.com/deploy/nvidia-smi/index.html#fb-memory-usage)

## Was hat sich nach 64 Updates geändert?

Trainings- und Validierungstext stammen aus getrennten Teilen von *Tiny Shakespeare*. Seed 42 legt die Initialisierung und die Auswahl der Trainingsfenster fest. Jeder der 64 Schritte berechnet die Kreuzentropie für $4\cdot128=512$ Zielpositionen, führt den Rückwärtslauf aus und aktualisiert die Gewichte mit AdamW. Die Lernrate steigt in den ersten sechs Updates auf 0,0003 und sinkt bis zum letzten auf 0,00003. Die Matrizen liegen im Speicher als FP32-Zahlen vor. Für große Matrixprodukte lässt der Trainer TF32 zu. [Lernrate und Update](../../tools/nfgpu.c) · [64 Batchmessungen](../../results/gpt2-medium-355m/batches.csv)

| Updates | Verarbeitete Trainingsziele | Verlust auf 13.106 Validierungszielen, Nats/Token |
| ---: | ---: | ---: |
| 0 | 0 | 11,09613 |
| 32 | 16.384 | 6,52701 |
| 64 | 32.768 | **6,36462** |

Welche Wahrscheinlichkeit gibt das Modell den richtigen Folgetokens in einem ungelesenen Text? Bei einem mittleren Verlust $\mathcal L=-\langle\ln p\rangle$ ist $e^{-\mathcal L}$ das geometrische Mittel dieser Wahrscheinlichkeiten. Auf denselben 13.106 Validierungszielen stieg es um den Faktor $e^{11{,}09613-6{,}36462}\approx113$. Nach dem Laden des besten Checkpoints in einem neuen Prozess ergab sich erneut ein Validierungsverlust von **6,36461656** Nats pro Token, auf alle protokollierten Stellen gleich. Ob das Modell auch zusammenhängende Dialoge schreibt, wäre eine eigene Frage an die erzeugten Texte. [Verlustkurve](../../results/gpt2-medium-355m/history.csv) · [Messung nach erneutem Laden](../../results/gpt2-medium-355m/validation-reload.csv) · [Erzeugte Tokenfolgen](../../results/gpt2-medium-355m/samples.csv)

## Den Lauf wiederholen

Die Befehle laufen in einer PowerShell-Sitzung im Stammverzeichnis des Projekts. Die Abschnitte 1 und 2 der GPU-Anleitung beschreiben die Vorbereitung. Sie installieren die verwendeten cuBLAS- und NVRTC-Versionen und erzeugen die drei Datensätze für Training, Validierung und Test sowie den Prompt. Das äußere Messprogramm benötigt nur Python 3 mit Standardbibliothek. `$py` bezeichnet den verwendeten Python-Interpreter. [GPU-Einrichtung und Datensplits](../training-124m.md) · [Datei-Prüfsummen dieses Laufs](../../results/gpt2-medium-355m/run.json)

~~~powershell
$py = 'python'
$blas = (Resolve-Path 'build-gpu/runtime/nvidia/cublas/bin').Path
$rtc = (Resolve-Path 'build-gpu/runtime/nvidia/cuda_nvrtc/bin').Path
$env:PATH = "$blas;$rtc;$env:PATH"
$env:NFGPU_KERNELS = (Resolve-Path 'src/gpu_kernels.cu').Path
$gpu = Join-Path (Resolve-Path build-gpu).Path 'nfgpu-medium.exe'
gcc -O3 -std=c11 -Wall -Wextra -Wpedantic -Isrc tools/nfgpu.c src/gpu.c src/gpt.c src/checkpoint.c -lm -o $gpu
if ($LASTEXITCODE) { throw 'Build fehlgeschlagen' }
& $gpu check
if ($LASTEXITCODE) { throw 'CPU/GPU-Vergleich fehlgeschlagen' }
~~~

Jede Wiederholung erhält mit dem nächsten Befehl ein eigenes Verzeichnis. Darin schreibt der Trainer seine Dateien nach `run`, das äußere Messprogramm schreibt die Kartenbelegung nach `board-memory.csv`.

~~~powershell
$experiment = New-Item -ItemType Directory -Path ('run-medium-' + [Guid]::NewGuid().ToString('N'))
$run = Join-Path $experiment.FullName 'run'
$board = Join-Path $experiment.FullName 'board-memory.csv'
$data = 'build-gpu/data'
& $py tools/measure_gpu_run.py --out $board -- $gpu train "$data/train.u16" "$data/val.u16" $run 64 128 4 32 42 "$data/prompt.u16" medium
if ($LASTEXITCODE) { throw 'Training oder Messung fehlgeschlagen' }
& $gpu eval "$run/best.nfg" "$data/val.u16" 4 128
if ($LASTEXITCODE) { throw 'Nachladen fehlgeschlagen' }
Import-Csv $board | Sort-Object { [int]$_.used_mib } -Descending | Select-Object -First 1
Import-Csv "$run/memory.csv" | Where-Object phase -eq 'buffers_ready'
~~~

Die Argumente `64 128 4 32 42` stehen für Updates, Fensterlänge, Batchgröße, Auswertungsintervall und Seed. Das letzte Argument `medium` wählt GPT-2 Medium. Der veröffentlichte Checkpoint ist über Git LFS abrufbar und umfasst **1.419.292.972 Byte**. Seine SHA-256-Prüfsumme steht im Versuchsprotokoll. Bei einem eigenen Trainingslauf können parallele Fließkommaadditionen in anderer Reihenfolge ablaufen. Dann können sich trotz desselben Seeds einzelne Gewichte und damit auch die Prüfsumme ändern. [Checkpoint](../../results/gpt2-medium-355m/best.nfg) · [Prüfsumme und Konfiguration](../../results/gpt2-medium-355m/run.json)

~~~powershell
git lfs pull --include="results/gpt2-medium-355m/best.nfg"
Get-FileHash -Algorithm SHA256 results/gpt2-medium-355m/best.nfg
~~~

Das Plot-Skript zeichnet die veröffentlichte Abbildung aus den beiden CSV-Dateien. Im GIF laufen die Messpunkte etwa doppelt so schnell wie im Versuch durch, mit zwei Sekunden Pause am Ende. Wann entsteht der größte Teil des Anstiegs? Zwischen 6,34 und 6,55 Sekunden springt die Belegung von 1.643 auf 8.051 MiB. [Skript zur Abbildung](../../tools/plot_medium_memory.py)

~~~powershell
& $py tools/plot_medium_memory.py
& $py tools/plot_medium_memory.py --out results/gpt2-medium-355m/memory.gif
~~~

## Wie viel Speicher braucht ein doppelt so langes Fenster?

Die Batchgröße bleibt bei vier, die Fensterlänge steigt von 128 auf 256 Tokens. Welcher Teil der 6,25 GiB bleibt gleich, welcher verdoppelt und welcher vervierfacht sich?

<details>
<summary>Speicherrechnung für 256 und 1.024 Positionen</summary>

Die 354.823.168 Gewichte, ihre Gradienten und die beiden Adam-Momente benötigen weiterhin 5,2873 GiB. In den vier Fenstern gibt es nun doppelt so viele Positionen und damit doppelt so viele Logits. Das Attention-Feld wächst dagegen in beiden Richtungen. Bei doppelt so vielen Queries und Keys steigt sein Bedarf je Block von 4 auf 16 MiB.

Setzt man $T=256$ in die Reservierungsformel des Trainers ein, ergeben sich für beide Puffer zusammen rund **7,42 GiB**. Bei $T=1\,024$ wären es **18,49 GiB**. Beides sind Vorhersagen für diesen Trainer bei $B=4$. Der Messwert von 6,25 GiB gehört zum Lauf mit $T=128$.

</details>

[Zurück zur Modellgröße](06-modellgroesse.md) · [Übersicht](README.md)
