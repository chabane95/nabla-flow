# GPT-2 Small auf Tiny Shakespeare reproduzieren

Der Decoder mit **124.439.808 Parametern** wurde von zufälligen Gewichten aus auf einem lokalen Rechner trainiert. Nach 2.400 AdamW-Updates erreichte er eine Test-NLL von **5,1708 Nat pro GPT-2-Token**. Eine aus demselben Trainingstext geschätzte Unigrammverteilung erreichte **6,6837**. Die Testmessung verwendet den Checkpoint mit der niedrigsten Validierungs-NLL. [Trainingsverlauf](../results/gpt2-124m/history.csv) · [Testmessung](../results/gpt2-124m/test.csv) · [Unigrammvergleich](../results/gpt2-124m/unigram-test.csv)

| Referenzlauf | Wert |
| --- | --- |
| Rechner | Windows x64, Ryzen 9 5900X, GeForce RTX 4070 SUPER, 12.282 MiB VRAM, NVIDIA-Treiber 591.86 |
| Architektur | 12 Decoderblöcke, 12 Köpfe, Breite 768, MLP-Breite 3.072, gemeinsame Eingabe- und Ausgabematrix |
| Vokabular und Positionen | 50.257 GPT-2-BPE-IDs, 1.024 gelernte Positionen |
| Ein Trainingsschritt | 4 zufällig gezogene Fenster × 128 verschobene Zieltoken |
| Trainingsbudget | 2.400 Schritte × 512 Zieltoken = 1.228.800 verarbeitete Ziele |
| Zeit bis zur letzten protokollierten Validierung | 121,118 s nach der Modellinitialisierung |
| Gewählter Checkpoint | Schritt 2.400 von 2.400, Validierungs-NLL 4,14468463, Perplexität 63,0977199 |
| Lokale Gewichtsdatei | `run-gpt2-124m/best.nfg`, 497.759.532 Bytes |
| SHA-256 der Referenzgewichte | `1d53a5435ed0d530c96bcd983c63505659fad7014395c7787bd570f3e36098a3` |

Die Zeitmarke umfasst frühere Validierungen, Checkpoint-Schreibvorgänge und Samples. Die letzte Gewichtsdatei und die letzte Textprobe werden nach dieser Zeitmarke geschrieben. Die Tokenzahl zählt auch Positionen, die in mehreren zufällig gezogenen Fenstern vorkommen. Im Training werden die Positionsvektoren 0 bis 127 durch Textbeispiele genutzt. [C-Trainingstreiber](../tools/nfgpu.c)

## 1. Build und GPU prüfen

Die PowerShell-Befehle laufen im Hauptverzeichnis von `nabla-flow`, in derselben Sitzung. Auf dem Referenzrechner kamen GCC 14.2.0 und Python 3.12.14 für Windows x64 zum Einsatz. Die GPU braucht mindestens Compute Capability 8.0 und einen Treiber für CUDA 12. Die Pakete landen unter `build-gpu`. Falls Python unter einem anderen Namen installiert ist, kann in `$py` der volle Pfad zur eigenen `python.exe` stehen.

```powershell
$ErrorActionPreference = 'Stop'
$py = 'python.exe'
& $py --version
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
gcc --version

New-Item -ItemType Directory -Force -Path build-gpu | Out-Null
& $py -m pip install --only-binary=:all: --no-deps --target build-gpu/runtime nvidia-cublas-cu12==12.9.2.10 nvidia-cuda-nvrtc-cu12==12.9.86
if ($LASTEXITCODE) { throw 'CUDA-Pakete konnten nicht eingerichtet werden' }
& $py -m pip install --only-binary=:all: --target build-gpu/python tiktoken==0.14.0 numpy==2.5.5 matplotlib==3.11.2 Pillow==13.0.0
if ($LASTEXITCODE) { throw 'Python-Pakete konnten nicht eingerichtet werden' }

$env:PYTHONPATH = (Resolve-Path build-gpu/python).Path
$blas = (Resolve-Path build-gpu/runtime/nvidia/cublas/bin).Path
$rtc = (Resolve-Path build-gpu/runtime/nvidia/cuda_nvrtc/bin).Path
$env:PATH = "$blas;$rtc;$env:PATH"
$env:NFGPU_KERNELS = (Resolve-Path src/gpu_kernels.cu).Path
Get-Item (Join-Path $blas 'cublas64_12.dll'), (Join-Path $blas 'cublasLt64_12.dll'), (Join-Path $rtc 'nvrtc64_120_0.dll'), (Join-Path $rtc 'nvrtc-builtins64_129.dll')

gcc -O3 -std=c11 -Wall -Wextra -Wpedantic -Isrc tools/nfgpu.c src/gpu.c src/gpt.c src/checkpoint.c -lm -o build-gpu/nfgpu.exe
if ($LASTEXITCODE) { throw 'C-Build fehlgeschlagen' }
$gpu = (Resolve-Path build-gpu/nfgpu.exe).Path
& $gpu check
if ($LASTEXITCODE) { throw 'CPU/GPU-Vergleich fehlgeschlagen' }
```

Der C-Host lädt den CUDA-Treiber, cuBLAS und NVRTC. NVRTC übersetzt die CUDA-Kernel beim Programmstart. Die großen Matrixprodukte dürfen Tensor Cores mit TF32 verwenden. Parameter, Gradienten, Adam-Momente und Aktivierungen liegen als FP32 vor. Einzelne GPU-Additionen in der Rückwärtsrechnung verwenden Atomics. Ein fester Seed bestimmt Initialisierung und Fensterwahl. Unterschiede in Rundung, Hardware oder Bibliotheksversion können die genaue Gewichtsprüfsumme eines neuen Laufs verändern. [CUDA-Host](../src/gpu.c) · [CUDA-Kernel](../src/gpu_kernels.cu)

## 2. Text beschaffen und in BPE-IDs zerlegen

Die Quelle ist die 1.115.394 Bytes große Datei `input.txt` aus Karpathys `char-rnn`, Commit `370cbcd448eb7daf32f21a6be560b70e0b33c4e3`. Der Vorbereiter prüft den Hash und den Rückweg von den BPE-IDs zu den Originalbytes jeder Rede. Die 7.222 durch Leerzeilen getrennten Reden bleiben in ihrer ursprünglichen Reihenfolge. Alle Zeilenumbrüche bleiben erhalten. Auf jede Rede folgt das End-of-Text-Token 50256. [Originalquelle](https://github.com/karpathy/char-rnn/blob/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt) · [Aufbereitung](../tools/prepare_124m.py)

```powershell
$source = 'build-gpu/tinyshakespeare.txt'
$url = 'https://raw.githubusercontent.com/karpathy/char-rnn/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt'
curl.exe -fL $url -o $source
if ($LASTEXITCODE) { throw 'Textdownload fehlgeschlagen' }
$expected = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
if ((Get-FileHash -Algorithm SHA256 $source).Hash -ne $expected) {
    throw 'Quelltext-Hash stimmt nicht'
}
$env:TIKTOKEN_CACHE_DIR = Join-Path (Resolve-Path build-gpu).Path 'bpe-cache'
& $py tools/prepare_124m.py --source $source --out build-gpu/data
if ($LASTEXITCODE) { throw 'Datenaufbereitung fehlgeschlagen' }
```

| Split | Reden, ab 1 gezählt | BPE-IDs einschließlich EOT | SHA-256 der uint16-LE-Datei |
| --- | --- | ---: | --- |
| Training | 1–6.499 | 311.356 | `2aeea0da9f1bf26b6244ab22f41444ccf185c49dd9785c3c2cf650e71bb7bb02` |
| Validierung | 6.500–6.860 | 13.107 | `57ee490f58fc073c4c07f703cc86c4cb877b731d3efb009e93c76c67edfa56c0` |
| Test | 6.861–7.222 | 13.565 | `877e82b0e209cbee7ae6745beb284677b8e6616c855179efa666291e2a40a0d0` |

Ein Vergleich aller vollständigen Reden ergab keine exakten Wiederholungen über Splitgrenzen hinweg. Wiederholungen innerhalb eines Splits bleiben erhalten. Jeder Split bildet einen Tokenstrom. Trainingsfenster können über ein EOT hinausreichen. [Datenprotokoll](../results/gpt2-124m/data.json)

## 3. Trainieren und Checkpoint auswählen

Der Zielordner des Trainings muss neu sein. Er entsteht durch den C-Treiber. Die Argumente nach dem Zielordner sind Schritte, Fensterlänge, Batchgröße, Messintervall, Trainingsseed und Promptdatei.

```powershell
$run = 'run-gpt2-124m'
$data = 'build-gpu/data'
& $gpu train "$data/train.u16" "$data/val.u16" $run 2400 128 4 100 42 "$data/prompt.u16"
if ($LASTEXITCODE) { throw 'Training fehlgeschlagen' }
$model = Join-Path $run 'best.nfg'
Get-FileHash -Algorithm SHA256 $model
```

AdamW verwendet $\beta_1=0{,}9$, $\beta_2=0{,}999$, $\epsilon=10^{-8}$, Weight Decay 0,1 auf den Matrizen und eine maximale globale Gradientennorm von 1,0. Die Lernrate steigt in den ersten 100 Schritten auf 0,0003 und sinkt per Kosinus auf 0,00003. Die gemeldete Norm wird vor dem Clipping auf dem über die 512 Ziele gemittelten Gradienten bestimmt.

Für jeden Optimierungsschritt protokolliert `batches.csv` den Verlust **vor** der Gewichtsänderung. Alle 100 Schritte und bei Schritt null misst `history.csv` den aktuellen Zustand auf 32 festen Trainingsfenstern sowie auf sämtlichen 13.106 Validierungszielen. Jedes Ziel ab der zweiten ID wird in der Validierung einmal gezählt. Die Attention beginnt an jedem Fenster neu. `best.nfg` enthält den Zustand mit der bisher niedrigsten Validierungs-NLL und dient der Inferenz. [Einzelne Updates](../results/gpt2-124m/batches.csv) · [Kontrollmessungen](../results/gpt2-124m/history.csv)

## 4. Test, Vergleichsmodell und eigene Textprobe

```powershell
& $gpu eval $model "$data/val.u16" 4 128
& $gpu eval $model "$data/test.u16" 4 128
& $gpu unigram "$data/train.u16" "$data/test.u16"

$promptText = "JULIET:" + [char]10
$promptFile = Join-Path $run 'juliet-prompt.u16'
& $py tools/gpt2_tokens.py prompt $promptText --out $promptFile
if ($LASTEXITCODE) { throw 'Neue Promptdatei erforderlich' }
$newIds = & $gpu sample $model $promptFile 64 128
if ($LASTEXITCODE) { throw 'Sampling fehlgeschlagen' }
$newIds | & $py tools/gpt2_tokens.py decode
```

Der Test umfasst 13.564 Folgetoken. Die NLL ist die mittlere negative natürliche Log-Wahrscheinlichkeit des korrekten Tokens. Ihre Exponentialfunktion ist die Perplexität. Das Unigrammmodell schätzt die Häufigkeit jedes Tokens allein aus den Trainingszielen und addiert 0,1 pro Vokabulareintrag. Beide Modelle werden auf denselben Testpositionen verglichen.

| Auswertung | Zieltoken | NLL | Perplexität |
| --- | ---: | ---: | ---: |
| GPT, Validierung nach Laden der Datei | 13.106 | 4,14468463 | 63,0977199 |
| GPT, Test | 13.564 | 5,17076582 | 176,049608 |
| Unigramm, Test | 13.564 | 6,68371474 | 799,282728 |

Das Sampling gibt nur neu gezogene IDs aus. Top-k 40, Temperatur 0,8 und Samplingseed 42 sind im Treiber festgelegt. EOT beendet die Generierung vorzeitig. Der erste Aufruf von `prompt --out` legt die Datei neu an. Für eine weitere Probe kann eine bestehende Promptdatei direkt wieder an `sample` übergeben werden. [Erneute Validierung](../results/gpt2-124m/validation-reload.csv) · [Testvergleich](../results/gpt2-124m/unigram-test.csv)

## 5. Abbildungen und Textbericht

```powershell
& $py tools/plot_124m.py --run $run --out $run
& $py tools/report_124m.py --run $run --data $data --steps 2400 --out "$run/samples-report.json"
```

Die Grafik zeigt die 25 Messpunkte von Schritt 0 bis 2.400. Der Textbericht dekodiert jede während des Trainings erzeugte Probe zum festen Prompt `ROMEO:` mit anschließendem LF. Nach dem letzten Update lautet die Fortsetzung „Now, that is the time of me to hear.“ mit zwei LF und EOT. In dieser Probe umfasst die längste zusammenhängende, exakt gleiche Folge aus einer Trainingsrede drei BPE-Token. Für diesen Vergleich werden Prompt und EOT ausgeschlossen. [Lernkurven](../results/gpt2-124m/training.png) · [Alle Textproben und Trainingsübereinstimmungen](../results/gpt2-124m/samples-report.json)

## Numerische Kontrollmessungen

- **CPU gegen GPU beim Training.** `nfgpu check` vergleicht zwei unterschiedlich breite Decoder auf zwei Sequenzen mit jeweils sieben Positionen und einem zusätzlichen kürzeren Fenster. Der Test prüft Logits, sämtliche akkumulierten Parametergradienten, die Norm vor dem Clipping und zwei AdamW-Updates. Die relativen L2-Fehler der Gradienten betrugen im Referenzlauf $2{,}06\cdot10^{-7}$ bei Breite 12 und $8{,}80\cdot10^{-4}$ bei Breite 64 mit TF32.
- **CPU gegen GPU mit trainierten Gewichten.** Für die ersten 17 Validierungs-IDs wurden die 16 nächsten Vorhersagen mit demselben 124M-Checkpoint berechnet. Über 804.112 Logits betrug die maximale absolute Abweichung 0,002537, die Wurzel der mittleren quadratischen Abweichung 0,000371. Für alle 16 Positionen lieferten CPU und GPU dieselbe wahrscheinlichste ID. Die mittlere NLL lag auf der CPU bei 1,320187828 und auf der GPU bei 1,320191428.
- **Gewichtsauswahl und Laden.** Die erneut aus `best.nfg` berechnete NLL auf dem kompletten Validierungssplit stimmt in den ausgegebenen Stellen mit dem Minimum der Lernhistorie überein. [Messung des geladenen Checkpoints](../results/gpt2-124m/validation-reload.csv)
