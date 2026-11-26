# GPT-2 Small in C

Die Einstellung `scale` benutzt denselben Decoder wie die kleinen Sprachversuche. Sie setzt 12 Blöcke mit je 12 Köpfen, eine Breite von 768, 1.024 gelernte Positionen und ein GPT-2-Vokabular mit 50.257 Einträgen. Damit lässt sich auf eigenen, vorab tokenisierten Texten von zufälligen Gewichten aus trainieren. Das Modellformat im Repository speichert Inferenzgewichte. Ein neuer Trainingsaufruf startet einen neuen Lauf.

## Nachzählen

Für Vokabulargröße $V$, Kontext $T$, Breite $d$ und Blockzahl $L$ ergibt sich bei gemeinsam genutzten Ein- und Ausgabegewichten

$$
N = (V+T)d + L(12d^2 + 13d) + 2d.
$$

Die vier Attention-Projektionen tragen $4d^2$ pro Block bei, die zwei MLP-Matrizen $8d^2$. Dazu kommen die Biaswerte und die beiden LayerNorms eines Blocks. Die letzten $2d$ Parameter sind die abschließende LayerNorm. Ein zweites Mal $Vd$ für den Ausgabekopf zu addieren wäre bei diesem Netz ein Fehler. Der Befehl `nablaflow scale info` gibt **124.439.808** aus. [GPT-2-Referenzcode](https://github.com/openai/gpt-2/blob/master/src/model.py) · [C-Kern](../src/gpt.c)

## Was bei voller Größe gelaufen ist

~~~powershell
.\nablaflow.exe scale info
.\nablaflow.exe scale smoke
~~~

Im Funktionstest bekommt der Transformer die GPT-2-IDs für `Hello, world!`. Vier Vorhersagen, also vier verschobene Ziele, gehen in die mittlere Kreuzentropie ein. Nach einem AdamW-Schritt auf dieser Sequenz sinkt sie von **11,052 auf 4,779 Nats je Token**. Die volle Parameterzahl ist auch bei diesem kurzen Fenster aktiv. Der C-Kern reserviert Aktivierungen nach der tatsächlich genutzten Länge. Beim Training mit vier Tokens liegen Gewichte, Gradienten, zwei Adam-Momente und Updatepuffer zusammen bei rund 2,32 GiB, zuzüglich Aktivierungen und Prozessspeicher. [Messzeile aus dem C-Lauf](../results/scale-smoke.csv)

Auch der Weg über Dateien wurde geprüft. Trainingszeile `Hello, world!`, separate Validierungszeile `Hello, friend!`, Seed 42, ein Update. Die Validierungs-NLL betrug zuerst 11,22694 und danach 6,61927. Der geschriebene Checkpoint war 497.759.532 Byte groß. Nach dem Laden setzte das Modell die IDs `15496 11` greedy mit `995 0 50256` fort. Das ist ein Test für Training, Serialisierung und Inferenz auf wenigen Wörtern. Ein Vortraining auf Webtext und eine Sprachqualitätsmessung für das große Modell stehen aus. Die beiden längeren Sprachversuche mit Messungen auf ganzen Testkorpora sind im Haupttext verlinkt.

## Eigene Texte

Pro Zeile steht ein Dokument in UTF-8. Training und Validierung brauchen verschiedene Texte. Der optionale Konverter benutzt das offizielle GPT-2-Encoding aus `tiktoken` und schreibt ein End-of-Text-Token (50256) hinter jedes Dokument. Das C-Programm liest rohe uint16-Werte in Little-Endian-Reihenfolge. Die zwei Dateien haben keinen Header. Beim ersten Gebrauch lädt `tiktoken` seine Vokabulardateien. Ein vorhandener kompatibler Token-Shard kann direkt an C übergeben werden. [GPT-2-Encoding in tiktoken](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)

~~~powershell
python -m pip install tiktoken
python tools/gpt2_tokens.py encode data/train.txt train.u16
python tools/gpt2_tokens.py encode data/val.txt val.u16
python tools/gpt2_tokens.py prompt "Hello, world!"
~~~

Das letzte Kommando liefert `15496 11 995 0`. Für einen kurzen eigenen Systemtest eignen sich vier bis sechzehn Tokens pro Fenster und wenige Schritte. Die obigen Beispieldateien umfassen mehrere Tausend Tokens. Die vollständige Validierung in C kann bei dieser Modellgröße entsprechend dauern.

~~~powershell
.\nablaflow.exe scale train train.u16 val.u16 5 16 mein-modell.nfg 1
.\nablaflow.exe scale generate mein-modell.nfg 12 15496 11 | python tools/gpt2_tokens.py decode
~~~

Nach den beiden Shards folgen **Schritte, Fensterlänge, neuer Checkpointpfad und Batchgröße**. Die Fensterlänge liegt zwischen 1 und 1.024 und bestimmt den Rechenaufwand pro Beispiel. Die Parameterzahl bleibt gleich. In jeder Iteration werden Trainingsfenster aus dem Shard gezogen, die Token-Gradienten gesammelt und ein AdamW-Schritt ausgeführt. Lernrate $3\cdot10^{-4}$, Weight Decay 0,1 und Normgrenze 1 sind in `src/scale.c` festgelegt. Die sequentielle Validierung zählt jedes Ziel ab dem zweiten Token des Shards genau einmal. Gemessen wird am Anfang, alle 100 Schritte und am Ende. Der Checkpoint hält den niedrigsten gemessenen Validierungsverlust fest. `generate` arbeitet greedy, schreibt dezimale IDs und endet beim EOT-Token oder am vorgegebenen Limit. [Trainings- und Inferenztreiber](../src/scale.c)
