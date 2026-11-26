# Ein kleiner GPT liest Shakespeare

Wie gut lässt sich das nächste Zeichen eines unbekannten Dialogs vorhersagen? Ein in C trainierter Decoder mit **43.056 Parametern** erreicht auf 64 zurückgehaltenen Passagen aus *Coriolanus* eine Test-Perplexität von **6,98**. Ein Bigrammmodell, das nur das unmittelbar vorhergehende Token kennt, erreicht **10,70**. Als Ziel zählt jedes Textbyte sowie ein Endezeichen pro Absatz. [Testmessung](test.csv)

| Modell | Validierungs-NLL | Test-NLL | Test-Perplexität |
| --- | ---: | ---: | ---: |
| Bigramm, Add-0,1-Glättung | 2,4167 | 2,3699 | 10,6962 |
| GPT, Schritt 1700 | 1,9742 | 1,9432 | 6,9810 |

Für jedes nächste Byte zählt, welche Wahrscheinlichkeit das Modell dem tatsächlich folgenden Byte gibt. Aus diesen Wahrscheinlichkeiten ergibt sich die mittlere negative Log-Likelihood (NLL). Der Unterschied zwischen den beiden Testwerten entspricht **0,616 Bit pro Token**. Was das beim Schreiben ausmacht, sieht man an der Textprobe unten. [Trainingsprotokoll](history.csv) · [Laufparameter](summary.csv)

### Was das Modell schreibt

Nach `MENENIUS: ` schreibt das Modell `I gay with picioustire hot sirke`. Man erkennt schon am Satzanfang, was die niedrigere Perplexität hier wert ist. Englische Wortstücke und kleine Funktionswörter tauchen auf, „picioustire“ ist frei erfunden. Von einer Rede des Menenius ist das noch weit entfernt. Verwendet wurden Temperatur 0,8, Top-k 12, Seed 7 und höchstens 220 neue Bytes. [Ganze Ausgabe](sample.txt) · [Inferenzgewichte](model.nfg)

### Text und Versuchsaufbau

Der Ausschnitt stammt aus Andrej Karpathys *Tiny Shakespeare*. Die Quelldatei hat 1.115.394 Bytes und den SHA-256-Wert `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. [Quelldatei im verwendeten Commit](https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt)

Ein oder mehrere leere Zeilen trennen die Absätze. Der C-Helfer erhält ihre Reihenfolge, verbindet die Zeilen innerhalb jedes Absatzes mit je einem ASCII-Leerzeichen und schreibt ein LF nach jedem Absatz. So bleiben Sprecherangaben, Großschreibung und Satzzeichen erhalten. Ein ganzer Absatz gehört immer zu genau einem Split. Kein Validierungs- oder Testabsatz stimmt vollständig mit einem Trainingsabsatz überein. [C-Helfer](../../tools/shakespeare.c)

| Split | Absätze im Original, inklusive | Zieltoken | SHA-256 der normalisierten Datei |
| --- | --- | ---: | --- |
| Training | 1–800 | 105.394 | `176b02cbb6193d1a152c782bfe9c2ee9a2d6d9c04c42f9b788d849961052fde0` |
| Validierung | 801–864 | 9.300 | `b455c181a6f5dec0a56167d7e8cc3ba3c189c9d88191838fa6ffae47c59c9b42` |
| Test | 865–928 | 6.819 | `b52508d91097b89bb9df01e77f2d96935555f4ed49bde7f332604307b1dea99f` |

Das Vokabular enthält 256 Bytewerte sowie BOS und EOS. Jeder Absatz erhält einen eigenen Anfang und ein eigenes Ende. Bei mehr als 48 Vorhersagen pro Absatz zerlegt der Trainer die verschobenen Eingabe-Ziel-Paare in aufeinanderfolgende Blöcke. Innerhalb eines Blocks kann jedes Token nur frühere Positionen und sich selbst sehen. Jede Zielposition wird bei der Auswertung genau einmal gezählt.

Der Decoder hat einen Transformerblock, vier Attention-Köpfe und eine Einbettungsbreite von 48. AdamW verarbeitet acht Blöcke pro Schritt. Die Lernrate steigt über 20 Schritte auf 0,002 und fällt danach per Kosinus auf 0,0002. Die Trainingskurve verwendet 64 feste Trainingsblöcke mit 2.667 Zieltoken. Die Validierung umfasst alle 9.300 Zieltoken. Unter den alle 100 Schritte geprüften Gewichten einschließlich Schritt null erzielt Schritt 1700 die niedrigste Validierungs-NLL. Nur diese Gewichte gehen in die Testmessung ein. Das Bigramm schätzt seine Zählungen ausschließlich auf den Trainingsabsätzen.

### Lauf reproduzieren

Im Wurzelverzeichnis des Repositorys in einer Shell mit `make`, `gcc`, `curl` und `sha256sum` ausführen. Der Referenzlauf nutzte GCC 14.2.0 auf einer Windows-CPU. Die Verzeichnisse `build-shakespeare/splits` und `run-shakespeare` werden vom jeweiligen Programm neu angelegt. Die Quelldatei wird nur für das erstmalige Herunterladen online benötigt.

```sh
make CC=gcc
mkdir -p build-shakespeare
gcc -O3 -std=c11 tools/shakespeare.c -o build-shakespeare/split
curl -fL https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt -o build-shakespeare/input.txt
printf '%s  %s\n' 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed build-shakespeare/input.txt | sha256sum -c -
./build-shakespeare/split build-shakespeare/input.txt build-shakespeare/splits
./nablaflow train --train build-shakespeare/splits/train.txt --val build-shakespeare/splits/val.txt --out run-shakespeare --steps 1800 --ctx 48 --width 48 --heads 4 --layers 1 --batch 8 --lr 0.002 --eval-every 100 --seed 42
./nablaflow eval --model run-shakespeare/model.nfg --data build-shakespeare/splits/test.txt --train build-shakespeare/splits/train.txt
./nablaflow sample --model run-shakespeare/model.nfg --prompt 'MENENIUS: ' --tokens 220 --temperature 0.8 --top-k 12 --seed 7
```

Die mitgelieferten Inferenzgewichte belegen 172.524 Bytes. Für eine eigene Fortsetzung mit diesen Gewichten genügt der letzte Befehl mit `--model results/shakespeare/model.nfg`.
