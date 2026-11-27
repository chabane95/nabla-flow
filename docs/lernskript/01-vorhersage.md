# 1 · Vom Text zur messbaren Vorhersage

Halte bei `Fach A: Glas. Fach B: Harz. In B liegt ` unmittelbar nach dem letzten Leerzeichen an. Das erste unbekannte Byte ist das `H` von `Harz`. Ein Sprachmodell muss an dieser Stelle Zahlen liefern, an denen sich seine Prognose messen lässt. Mit „das Modell hat an Glas gedacht“ könnte man kaum rechnen. Mit „es gab `H` die Wahrscheinlichkeit $1/4$“ schon.

## Aus einer Zeile werden Beobachtungen

Als kleinsten Versuch nehmen wir die Zeile `Ab`. Das Byte-Modell liest eine nichtleere UTF-8-Zeile als einen Datensatz, fügt vorn ein Startsymbol (BOS) und hinten ein Endsymbol (EOS) an. Die Zahlen, die der Rechner für diese Zeile sieht, sind `256, 65, 98, 257`. Die folgende Tabelle enthält bereits das ganze Vorhersageproblem für diese Zeile bei ausreichend langem Kontext.

| Position $t$ | Sichtbare Eingabe bis $t$ | Gesuchtes Token $y_t$ |
| ---: | --- | --- |
| 0 | BOS | `A` (65) |
| 1 | BOS, `A` | `b` (98) |
| 2 | BOS, `A`, `b` | EOS (257) |

Die Eingabefolge lautet also `[256, 65, 98]`, die Zielfolge `[65, 98, 257]`. Beide lassen sich durch zwei um einen Eintrag versetzte Zeiger auf dieselben Daten darstellen. Zum Lernen kennt das Programm die gesamte Zeile. An jeder Position darf die Prognose nur die Einträge der mittleren Spalte verwenden. Bei der Auswertung der Textwahrscheinlichkeit bekommt es die tatsächlich beobachtete Vorgeschichte (*teacher forcing*). Beim freien Schreiben setzt es sein eigenes zuletzt erzeugtes Token ein. Ein früher Fehler kann dann die nächste Vorhersage verändern. [Datenaufbereitung und Selbsttest](../../src/data.c) · [Trainingsschleife und Textgenerierung](../../src/main.c)

**Ein Randfall zum Nachrechnen.** Bei einer Fensterlänge von zwei wird `Ab` in zwei Blöcken verarbeitet. Der erste hat die Eingaben `[BOS,A]` und Ziele `[A,b]`. Der zweite hat die Eingabe `[b]` und das Ziel `[EOS]`. Jedes Ziel zählt einmal. Beim zweiten Block ist `A` bereits aus dem sichtbaren Kontext verschwunden. Ein `ä` hat in UTF-8 die Bytes 195 und 164 und ergibt als eigene Zeile drei Ziele einschließlich EOS. So lässt sich die Einheit „pro Token“ für dieses Experiment genau angeben. Beim großen Shakespeare-Modell steht ein Token dagegen für einen Eintrag im GPT-2-Vokabular. [Definition der 258 IDs](../../src/data.h) · [GPT-2-Tokenisierung im Projekt](../../tools/prepare_124m.py)

**Warum wechselt das große Modell die Einheit?** Schon `Hello, world!` braucht 13 einzelne ASCII-Bytes. Wiederkehrende Bytefolgen lassen sich unter einer gemeinsamen ID speichern. Beim *Byte Pair Encoding* (BPE) lernt man dazu eine geordnete Folge von Zusammenführungen benachbarter Symbole. Beim Kodieren eines neuen Textes wendet man die festgelegten Regeln wieder an. Das Projekt verwendet den fertigen GPT-2-Tokenizer. Er kodiert diesen Prompt als `[15496, 11, 995, 0]`, also vier Tokens. Beim Aufbereiten der Shakespeare-Daten kommt nach jeder Rede das Endsymbol 50256 hinzu. Ein Fenster von 128 BPE-Tokens hat folglich keine feste Länge in Bytes. [Funktionierendes Beispiel mit genau diesen IDs](../gpt2-small.md) · [Kodierung der Reden](../../tools/prepare_124m.py)

## Warum ein Produkt von Wahrscheinlichkeiten?

Für Ereignisse mit $P(A)>0$ ist die bedingte Wahrscheinlichkeit definiert durch $P(B\mid A)=P(A,B)/P(A)$. Umstellen ergibt $P(A,B)=P(A)P(B\mid A)$. Setze für $A$ nun die gesamte bisherige Folge ein und wiederhole den Schritt. Für $x_0=\mathrm{BOS}$ und $x_n=\mathrm{EOS}$ erhält man

$$
P(x_1,\ldots,x_n\mid x_0)
=\prod_{i=1}^n P(x_i\mid x_0,\ldots,x_{i-1}).
$$

Die Zerlegung ist eine Identität der Wahrscheinlichkeitsrechnung und erlaubt beliebige Abhängigkeiten zwischen den Tokens. Die **Modellannahme** folgt beim Ersetzen jedes Faktors durch eine berechenbare Verteilung $p_\theta(\,\cdot\mid c_i)$ mit gemeinsamen Parametern $\theta$. Der Kontext $c_i$ enthält nur die für diese Vorhersage sichtbaren Tokens. Zwei identische sichtbare Kontexte müssen bei festen Gewichten dieselbe Verteilung ergeben. Diese Einschränkung können wir durch gezielt gewählte Eingaben prüfen.

## Wie viel kostet eine falsche Prognose?

Im Beispiel `Ab` gebe das Modell den drei tatsächlich eingetretenen Zielen der Reihe nach die Wahrscheinlichkeiten $1/2$, $1/4$ und $1/8$. Es weist dem beobachteten Ablauf das Produkt $1/64$ zu. Wenn wir Modellgewichte so wählen, dass beobachtete Daten möglichst wahrscheinlich werden, können wir auch ihren Logarithmus maximieren. Der Logarithmus wächst monoton und verwandelt das Produkt in eine Summe. Mit negativem Vorzeichen wird daraus ein zu minimierender Verlust. Für $N$ Zielpositionen lautet der Mittelwert

$$
\mathcal L(\theta)=-\frac{1}{N}\sum_{i=1}^{N}\ln p_\theta(y_i\mid c_i).
$$

In den Messdateien heißt dieser Mittelwert **NLL** (*negative log-likelihood*). Der natürliche Logarithmus gibt die Einheit **Nats pro Zieltoken**. Ein sicher richtig vorhergesagtes Token kostet null. Eine Halbierung seiner Wahrscheinlichkeit kostet zusätzlich $\ln 2$ Nats. Im Beispiel gilt $\mathcal L=\ln(64)/3=\ln 4$. Ein Modell, das jedem der drei Ziele die Wahrscheinlichkeit $1/4$ gibt, erreicht denselben mittleren Verlust. Um von Nats wieder zu einer anschaulichen Zahl zu gelangen, exponentieren wir den Mittelwert. Diese Größe heißt **Perplexität**.

$$
\mathrm{PPL}=e^{\mathcal L}
=\frac{1}{\left(\prod_{i=1}^{N} p_\theta(y_i\mid c_i)\right)^{1/N}}.
$$

Der Nenner ist das geometrische Mittel der den richtigen Tokens zugewiesenen Wahrscheinlichkeiten. Im Beispiel ist die PPL gleich 4. Bei Gleichverteilung über alle 258 Symbole des Byte-Modells ist sie 258. Im Labor-Test ist sie rund 1,270, ihr Kehrwert also rund 0,787. Dieser Wert gibt **keine Trefferquote** an. Eine Trefferquote berücksichtigt, welches Token an erster Stelle steht. Der Logarithmus reagiert auch darauf, ob das richtige Token die Wahrscheinlichkeit $0{,}4$ oder $0{,}0004$ erhielt. [Berechnung in `token_objective`](../../src/gpt.c) · [Testwerte des Fachmodells](../../results/test.csv)

**Warum jedes Ziel einzeln zählen?** Angenommen, ein Block mit nur einem Ziel hat den Verlust 2, ein Block mit drei Zielen hat mittleren Verlust 0. Der Gesamtmittelwert ist $(1\cdot2+3\cdot0)/4=0{,}5$. Der Mittelwert der beiden Blockmittel wäre 1 und gäbe dem einzelnen Ziel dreifaches Gewicht. Genau deshalb multipliziert `evaluate` den Blockmittelwert zunächst mit der Blocklänge. [Auswertung in C](../../src/main.c)

## Das kleinste Modell, das wir ausrechnen können

Ein **Bigramm-Modell** speichert für jedes unmittelbar vorhergehende Token $a$ eine Verteilung über das nächste Token $b$. Fixiere zunächst einen einzigen Vorgänger. In einem *erfundenen* Alphabet mit zwei möglichen Folgetokens beobachten wir dreimal `G` und einmal `H`. Sei $p$ die Wahrscheinlichkeit von `G`. Die vier Beobachtungen haben unter diesem Tabellenmodell die Likelihood $p^3(1-p)$. Ableiten liefert

$$
\frac{d}{dp}\ln[p^3(1-p)] = \frac{3}{p}-\frac{1}{1-p}=0
\quad\Longrightarrow\quad p=\frac34.
$$

Die zweite Ableitung $-3/p^2-1/(1-p)^2$ ist im Inneren negativ. Die gezählten Häufigkeiten liefern tatsächlich das Maximum. Das Argument verallgemeinert sich auf eine ganze Zeile. Bezeichne die Anzahl der Übergänge mit $n_{ab}$, ihre Summe mit $n_a$. Maximiere $\sum_b n_{ab}\ln p_b$ unter $\sum_b p_b=1$. Für Einträge mit positiver Häufigkeit liefert ein Lagrange-Multiplikator $n_{ab}/p_b=\lambda$. Die Normierung gibt $\lambda=n_a$, also $p_b=n_{ab}/n_a$. Ein unbeobachteter Übergang bekommt bei diesem Maximum die Wahrscheinlichkeit null und würde bei seinem ersten Auftreten im Test unendlichen Verlust verursachen.

Das Projekt addiert deshalb zu jeder Zelle $\alpha=0{,}1$ gedachte Beobachtungen. Beim realen Ausgabealphabet $V=258$ gilt

$$
p_{\rm bi}(b\mid a)=\frac{n_{ab}+\alpha}{n_a+\alpha V}.
$$

Der Nenner folgt direkt aus der Summe der neuen Zähler. Für einen nie gesehenen Vorgänger folgt $p_{\rm bi}=1/V$. Die Wahl von $\alpha$ ist eine Modellentscheidung. Die Zählungen stammen aus dem Training, die Bewertung erfolgt auf zurückgehaltenen Übergängen. [`bigram_nll`](../../src/data.c)

Jetzt wird die Grenze dieser Tabelle sichtbar. Lies `In A liegt ` und `In B liegt ` bis einschließlich Leerzeichen. Für das nächste Byte greift das Bigramm zweimal auf dieselbe Tabellenzeile zu. Es kann `G` und `H` in diesen zwei Fällen keine unterschiedlichen Chancen geben. Auf einem ausgeglichenen Spielzeugdatensatz mit genau diesen beiden Antworten erreicht die beste gemeinsame Verteilung $p_G=p_H=1/2$ und kostet im Mittel $\ln 2$ Nats an der entscheidenden Stelle. Ein Modell, das Fachinhalt und Frage im Kontext eindeutig zuordnet, kann den Verlust dort beliebig nahe an null bringen. Das ist ein überprüfbarer Grund, mehr Kontext zuzulassen.

## Vom Tabellenwert zum trainierbaren Score

Wie kann eine Funktion des Kontextes eine ganze gültige Verteilung ausgeben? Beginne mit einer Verteilung mit strikt positiven Einträgen. Wähle eine Referenzklasse $r$ und definiere ihre logarithmischen Chancenverhältnisse $z_k=\ln(p_k/p_r)$. Dann gilt $p_k=p_r e^{z_k}$. Summieren liefert $1=p_r\sum_j e^{z_j}$, also

$$
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}.
$$

Damit haben wir die **Softmax** konstruiert. Umgekehrt dürfen wir beliebige reelle Scores $z_k$ (Logits) vorgeben und erhalten immer positive, auf eins normierte Wahrscheinlichkeiten. Für zwei Klassen mit $z_G=\ln 3$ und $z_H=0$ entstehen die gerade hergeleiteten Gewichte $3/4$ und $1/4$. Addiere zu beiden Scores 100. Da nur Differenzen zählen, bleibt die Prognose gleich. Im Programm zieht man den größten Logit ab und verhindert so einen Überlauf beim Exponenzieren.

Für ein beobachtetes Ziel $y$ setze die Softmax in seinen Verlust ein. Mit $m=\max_j z_j$ gilt

$$
\ell=-\ln p_y
= (m-z_y)+\ln\sum_j e^{z_j-m}.
$$

Die Ableitung lässt sich aus der algebraisch gleichen Form $\ln\sum_j e^{z_j}-z_y$ gewinnen. Der erste Term liefert $e^{z_k}/\sum_j e^{z_j}=p_k$, der zweite liefert $-1$, falls $k=y$, sonst null. Somit ist $\partial\ell/\partial z_k=p_k-\mathbf 1_{k=y}$. Bei $p_G=3/4$ und beobachtetem `H` hat der Gradient die Komponenten $(+3/4,-3/4)$. Ein kleiner Schritt entgegen dem Gradienten senkt den Score für `G` und hebt den für `H`. Die Komponenten summieren sich zu null, wie es die Invarianz unter einer gemeinsamen Verschiebung verlangt. [Stabiler Verlust und Stabilitätstest](../../src/gpt.c)

Das Wort **Kreuzentropie** bezeichnet genau die hier verwendete Mittelung. Für eine feste Vorgeschichte sei $q_k$ die beobachtete relative Häufigkeit der Klasse $k$. Durch Zusammenfassen gleicher Ziele wird der mittlere Verlust zu $-\sum_k q_k\ln p_k$. Er wird bei $p=q$ minimal. Wer das ohne Ableitungen prüfen möchte, zieht den Wert bei $p=q$ ab und verwendet $-\ln u\geq 1-u$. Für positive $p,q$ folgt

$$
\mathcal L(p)-\mathcal L(q)
=-\sum_k q_k\ln(p_k/q_k)
\geq \sum_k(q_k-p_k)=0.
$$

Damit ist auch verständlich, was die Zielfunktion beim Lernen belohnt. Bei unsicheren Ausgängen ist eine zutreffende Verteilung die beste Prognose. Das Training eines Decoders stellt zusätzlich die Forderung, dass *dieselben Gewichte* diese Verteilungen für viele verschiedene Kontexte berechnen können.

## Wie aus einer Verteilung ein Text wird

Am Ende eines Prompts gibt es mehrere Möglichkeiten, ein Token auszuwählen. Angenommen, zwei mögliche Ausgaben haben die Scores $z_G=\ln 3$ und $z_H=0$. Wer jeweils das Maximum wählt, schreibt immer `G`. Wer gemäß der Verteilung zieht, schreibt im Mittel dreimal so oft `G` wie `H`. In beiden Fällen wird das gewählte Token Teil des nächsten Kontexts. Durch Wiederholung dieser einfachen Operation entsteht eine ganze Antwort.

Der Parameter **Temperatur** $\tau>0$ teilt die Scores vor der Auswahl durch $\tau$. Für zwei im Auswahlpool verbleibende Tokens folgt aus der Softmax sofort $p_\tau(G)/p_\tau(H)=\exp[(z_G-z_H)/\tau]=3^{1/\tau}$. Bei $\tau=2$ sinkt das Verhältnis von $3$ auf $\sqrt 3$. Bei sehr kleiner positiver Temperatur konzentriert sich die Masse auf das Maximum. Bei `--temperature 0` nimmt der Byte-Sampler das größte zulässige Logit direkt. `--top-k` kann den Pool zusätzlich auf die $k$ höchstbewerteten Kandidaten beschränken.

Diese Wahl gehört zum **Erzeugen**. Beim Messen der Textwahrscheinlichkeit wertet `eval` alle 258 Logits aus. `sample` filtert beim Schreiben unter anderem nach im Training gesehenen Bytes und gültigen UTF-8-Fortsetzungen. `recall` wählt das Maximum über alle 258 IDs, bei Gleichstand die kleinste ID. Weicht ein erzeugtes Byte von der Sollantwort ab, zählt die gesamte Antwort als falsch. So haben freier Beispieltext, exakte Antwortquote und Perplexität jeweils eigene Versuchsbedingungen. [Auswahl in `choose` und `sample`](../../src/main.c) · [Exakter Antworttest](../../src/recall.c)

## Zwei Aufgaben vor dem Weitergehen

1. Die Zeile `Aä` wird mit Kontextlänge zwei gelesen. Schreibe beide Blöcke mit ihren Eingaben und Zielen auf. Wie viele Terme hat ihre Verlustsumme?
2. Eine Antwort besteht aus zehn Bytes. Ein hypothetisches Modell gibt den ersten neun richtigen Bytes unter der jeweils richtigen Vorgeschichte die Wahrscheinlichkeit $0{,}99$, dem letzten richtigen Byte $0{,}1$. Ein falsches Schlussbyte erhält $0{,}9$. Berechne die Perplexität dieser zehn Vorhersagen und entscheide, ob gieriges Dekodieren die ganze Antwort trifft.

<details>
<summary>Lösungen und Kontrollen</summary>

1. Die Folge ist `[256,65,195,164,257]`. Die erste Eingabe `[256,65]` hat die Ziele `[65,195]`, die zweite Eingabe `[195,164]` hat die Ziele `[164,257]`. Das sind vier Verlustterme. Die Grenze eines Rechenblocks darf zwischen den beiden Bytes des `ä` liegen.
2. Der Mittelwert ist $-(9\ln 0{,}99+\ln 0{,}1)/10 \simeq 0{,}2393$ Nats. Das ergibt eine PPL von rund $1{,}27$. Trotzdem wählt das Modell am Ende das falsche Byte und erhält für diese Antwort null exakte Treffer. Dieses konstruierte Beispiel erklärt, *wie* die beiden Messungen auseinanderfallen können. Wo das gespeicherte Modell seine Fehler macht, muss der Versuch mit echten Prompts zeigen.

</details>

Im nächsten Kapitel ersetzen wir die einzelne Bigramm-Tabellenzeile durch eine Funktion, die weiter zurückliegende Positionen gewichten kann. [Weiter zu Kapitel 2](02-transformer.md) · [Übersicht](README.md)
