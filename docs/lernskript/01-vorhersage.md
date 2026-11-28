# 1 · Vom Text zur messbaren Vorhersage

Auf die Eingabe `Fach A: Glas. Fach B: Harz. In B liegt ` folgt im Datensatz das Wort `Harz`. Wie lässt sich die Vorhersage unmittelbar nach dem Leerzeichen bewerten? Für das erste gesuchte Byte `H` genügt eine einzige Zahl, die Wahrscheinlichkeit, die das Modell diesem Byte zuweist. Für das ganze Wort braucht man weitere Vorhersagen, jeweils unter einer um ein Byte verlängerten Vorgeschichte.

## Aus einer Zeile werden Beobachtungen

Die Zeile `Ab` enthält zwei Bytes. Das Programm behandelt jede nichtleere UTF-8-Zeile als einen Datensatz. Es ergänzt ein Startsymbol (BOS, ID 256) und ein Endsymbol (EOS, ID 257). Aus `Ab` wird so die Folge `[256, 65, 98, 257]`. Bei einer Fensterlänge von mindestens drei ergeben sich die folgenden Vorhersagen.

| Position $t$ | Sichtbare Eingabe bis $t$ | Gesuchtes Token $y_t$ |
| ---: | --- | --- |
| 0 | BOS | `A` (65) |
| 1 | BOS, `A` | `b` (98) |
| 2 | BOS, `A`, `b` | EOS (257) |

Die Eingabe `[256, 65, 98]` und die Ziele `[65, 98, 257]` liegen um einen Eintrag versetzt im selben Speicher. Die Tabelle unterscheidet, was dem Programm für die Verlustrechnung vorliegt und was der Decoder an jeder Position lesen darf. Beim Training und beim Messen der Textwahrscheinlichkeit enthält seine Vorgeschichte die tatsächlichen Bytes der Zeile (*teacher forcing*). Bei der freien Erzeugung geht das selbst gewählte Byte in die nächste Eingabe ein. [Datenaufbereitung und Selbsttest](../../src/data.c) · [Trainingsschleife und Textgenerierung](../../src/main.c)

**Was ändert ein kürzeres Fenster?** Bei Fensterlänge zwei enthält der erste Block die Eingaben `[BOS,A]` und Ziele `[A,b]`. Im zweiten Block ist nur `[b]` sichtbar, das Ziel ist `[EOS]`. Alle drei Ziele gehen einmal in die Auswertung ein. Bei der eigenen Zeile `ä` sind es ebenfalls drei Ziele, denn das Schriftzeichen besteht in UTF-8 aus den Bytes 195 und 164. Die Einheit des Verlusts ist hier ein Byte oder EOS. [Definition der 258 IDs](../../src/data.h) · [GPT-2-Tokenisierung im Projekt](../../tools/prepare_124m.py)

**Welche Einheit verwendet GPT-2?** `Hello, world!` belegt 13 ASCII-Bytes. Der GPT-2-Tokenizer kodiert denselben Text als `[15496, 11, 995, 0]`, also vier Tokens. Beim *Byte Pair Encoding* (BPE) erhält eine wiederkehrende Folge benachbarter Symbole eine gemeinsame ID. Die beim Erstellen des Vokabulars gelernten Zusammenführungen und ihre Rangfolge stehen beim Kodieren fest. Für die Shakespeare-Daten fügt das Projekt jeder Rede das Endsymbol 50256 an. Ein Kontext von 128 GPT-2-Tokens kann daher unterschiedlich viele Bytes umfassen. [Funktionierendes Beispiel mit genau diesen IDs](../gpt2-small.md) · [Kodierung der Reden](../../tools/prepare_124m.py)

## Warum ein Produkt von Wahrscheinlichkeiten?

Aus der Definition $P(B\mid A)=P(A,B)/P(A)$ für $P(A)>0$ folgt $P(A,B)=P(A)P(B\mid A)$. Für drei Tokens ergibt sich so $P(x_1,x_2,x_3)=P(x_1)P(x_2\mid x_1)P(x_3\mid x_1,x_2)$. Setzt man den Startmarker $x_0=\mathrm{BOS}$ voraus und führt die Zerlegung bis $x_n=\mathrm{EOS}$ fort, erhält man

$$
P(x_1,\ldots,x_n\mid x_0)
=\prod_{i=1}^n P(x_i\mid x_0,\ldots,x_{i-1}).
$$

Diese Kettenregel verlangt keine Unabhängigkeit der Tokens. Eine Modellannahme kommt erst hinzu, wenn eine Funktion $p_\theta(\,\cdot\mid c_i)$ die Faktoren berechnet. Der sichtbare Kontext $c_i$ kann kürzer sein als die ganze Vorgeschichte. Bei festen Gewichten $\theta$ und gleichen Positionen im Fenster erzeugen zwei identische sichtbare Folgen dieselbe Prognose. Frühere Unterschiede außerhalb des Fensters bleiben dem Modell verborgen.

## Wie viel kostet eine falsche Prognose?

Für die drei Ziele der Zeile `Ab` seien die Modellwahrscheinlichkeiten $1/2$, $1/4$ und $1/8$. Damit beträgt die Wahrscheinlichkeit der ganzen Folge $1/64$. Gesucht sind Gewichte, die den beobachteten Folgen eine möglichst hohe Wahrscheinlichkeit zuweisen. Der Logarithmus erhält die Rangfolge dieser Produkte und verwandelt die Multiplikationen in Additionen. Mit negativem Vorzeichen ergibt sich ein zu minimierender Verlust. Sein Mittel über $N$ Zielpositionen beträgt

$$
\mathcal L(\theta)=-\frac{1}{N}\sum_{i=1}^{N}\ln p_\theta(y_i\mid c_i).
$$

Die Messdateien bezeichnen diesen Mittelwert als **NLL** (*negative log-likelihood*). Der natürliche Logarithmus ergibt die Einheit **Nats pro Zieltoken**. Wahrscheinlichkeit eins trägt null bei, jede Halbierung der Zielwahrscheinlichkeit erhöht den einzelnen Verlust um $\ln 2$. Für `Ab` ergibt sich $\mathcal L=\ln(64)/3=\ln 4$. Derselbe Wert ergäbe sich, wenn alle drei Ziele die Wahrscheinlichkeit $1/4$ hätten. Die **Perplexität** ist der Exponentialwert dieses mittleren Verlusts.

$$
\mathrm{PPL}=e^{\mathcal L}
=\frac{1}{\left(\prod_{i=1}^{N} p_\theta(y_i\mid c_i)\right)^{1/N}}.
$$

Der Nenner ist das geometrische Mittel der Wahrscheinlichkeiten der beobachteten Ziele. Im Rechenbeispiel beträgt die PPL 4, bei einer Gleichverteilung über die 258 Symbole wäre sie 258. Für das gespeicherte Fachmodell ergibt der Test eine PPL von rund 1,270. Das entspricht einem geometrischen Mittel von rund 0,787. Für die Trefferquote wäre eine andere Frage zu beantworten. An wie vielen Positionen hatte das richtige Token den höchsten Score? Für den logarithmischen Verlust macht es beispielsweise einen Unterschied von $\ln(1000)$, ob ein Ziel die Wahrscheinlichkeit $0{,}4$ oder $0{,}0004$ erhält. [Berechnung in `token_objective`](../../src/gpt.c) · [Testwerte des Fachmodells](../../results/test.csv)

**Wie werden Blöcke unterschiedlicher Länge gemittelt?** Ein Block mit einem Ziel habe Verlust 2, ein Block mit drei Zielen habe mittleren Verlust 0. Über alle vier Ziele ergibt sich $(1\cdot2+3\cdot0)/4=0{,}5$. Wer nur die beiden Blockmittel mittelt, erhält 1 und gewichtet damit das einzelne Ziel dreimal so stark wie jedes Ziel im langen Block. Die Funktion `evaluate` multipliziert jeden Blockmittelwert mit der Zahl seiner Ziele und teilt die Summe durch die Gesamtzahl der Ziele. [Auswertung in C](../../src/main.c)

## Ein Bigramm-Modell aus Übergangszahlen

Ein **Bigramm-Modell** verwendet als Kontext nur das unmittelbar vorausgehende Token $a$. Angenommen, auf ein bestimmtes $a$ folgen in vier Beobachtungen dreimal `G` und einmal `H`. Im Rechenbeispiel gibt es nur diese beiden möglichen Nachfolger. Mit $p=P(G\mid a)$ hat die beobachtete Folge unter dem Modell die Likelihood $p^3(1-p)$. Das Maximum im Intervall $0<p<1$ erfüllt

$$
\frac{d}{dp}\ln[p^3(1-p)] = \frac{3}{p}-\frac{1}{1-p}=0
\quad\Longrightarrow\quad p=\frac34.
$$

Wegen $-3/p^2-1/(1-p)^2<0$ ist die logarithmische Likelihood für $0<p<1$ streng konkav. Für ein ganzes Alphabet seien $n_{ab}$ die Anzahlen der Übergänge $a\to b$ und $n_a=\sum_b n_{ab}>0$. Zu maximieren ist $\sum_b n_{ab}\ln p_b$ unter $\sum_b p_b=1$. Für beobachtete Nachfolger liefert der Lagrange-Multiplikator $n_{ab}/p_b=\lambda$. Unbeobachtete Nachfolger erhalten am Maximum keine Masse. Aus der Normierung folgt $\lambda=n_a$ und damit $p_b=n_{ab}/n_a$. Tritt im Test ein solcher unbeobachteter Übergang auf, wird $-\ln p_b$ unendlich.

Der Bigramm-Vergleich des Projekts glättet die Zählungen. Er addiert zu jeder Zelle $\alpha=0{,}1$ und verwendet die $V=258$ Ausgabesymbole des Byte-Modells. Damit gilt

$$
p_{\rm bi}(b\mid a)=\frac{n_{ab}+\alpha}{n_a+\alpha V}.
$$

Insgesamt kommen zum Nenner $V$ Beiträge der Größe $\alpha$ hinzu. Für einen im Training nie gesehenen Vorgänger ($n_a=0$) ergibt sich $p_{\rm bi}=1/V$. Die Übergänge werden ausschließlich in den Trainingsdaten gezählt. Für die Bewertung auf neuen Texten bleiben diese Zahlen fest. [`bigram_nll`](../../src/data.c)

Welche Verteilung verwendet das Bigramm nach `In A liegt ` und `In B liegt `? Beide Eingaben enden mit einem Leerzeichen. Deshalb entnimmt das Modell die Wahrscheinlichkeiten für `G` und `H` derselben Tabellenzeile. Sind bei festen Fachinhalten beide Fragen gleich häufig und nur diese beiden Anfangsbytes möglich, ist $p_G=p_H=1/2$ die beste gemeinsame Prognose. Sie kostet an dieser Position im Mittel $\ln 2$ Nats. Ein längerer Kontext enthält die Fachbezeichnung. Ordnet ein Modell die Frage dem richtigen Inhalt zu, kann es den Verlust an dieser Position beliebig nahe an null bringen.

## Vom Tabellenwert zum trainierbaren Score

Eine Funktion des Kontexts muss für $V$ Ausgänge nichtnegative Zahlen liefern, die sich zu eins summieren. Für eine strikt positive Verteilung lassen sich zunächst die Logarithmen der Wahrscheinlichkeitsverhältnisse zu einer Referenzklasse $r$ bilden, $z_k=\ln(p_k/p_r)$. Aus $p_k=p_r e^{z_k}$ folgt durch Summieren $1=p_r\sum_j e^{z_j}$. Einsetzen ergibt

$$
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}.
$$

Diese Abbildung heißt **Softmax**. Sie liefert für beliebige reelle Scores $z_k$ (*Logits*) eine normierte Verteilung. Für $z_G=\ln 3$ und $z_H=0$ ergeben sich $3/4$ und $1/4$. Wie ändern sich diese Werte, wenn beide Scores um 100 steigen? Der gemeinsame Faktor $e^{100}$ kürzt sich aus Zähler und Nenner. Der Code nutzt diese Freiheit und zieht vor dem Exponenzieren den größten Logit ab.

Für ein beobachtetes Ziel $y$ folgt mit $m=\max_j z_j$

$$
\ell=-\ln p_y
= (m-z_y)+\ln\sum_j e^{z_j-m}.
$$

Die Ableitung lässt sich aus der algebraisch gleichen Form $\ln\sum_j e^{z_j}-z_y$ gewinnen. Der erste Term liefert $e^{z_k}/\sum_j e^{z_j}=p_k$, der zweite liefert $-1$, falls $k=y$, sonst null. Somit ist $\partial\ell/\partial z_k=p_k-\mathbf 1_{k=y}$. Bei $p_G=3/4$ und beobachtetem `H` hat der Gradient die Komponenten $(+3/4,-3/4)$. Ein kleiner Schritt entgegen dem Gradienten senkt den Score für `G` und hebt den für `H`. Die Komponenten summieren sich zu null, wie es die Invarianz unter einer gemeinsamen Verschiebung verlangt. [Stabiler Verlust und Stabilitätstest](../../src/gpt.c)

Für einen festen Kontext bezeichne $q_k$ die beobachtete relative Häufigkeit der Klasse $k$. Fasst man die Verlustterme gleicher Ziele zusammen, erhält man die **Kreuzentropie** $-\sum_k q_k\ln p_k$. Warum liegt ihr Minimum bei $p=q$? Für positive $p_k,q_k$ folgt aus $-\ln u\geq 1-u$

$$
\mathcal L(p)-\mathcal L(q)
=-\sum_k q_k\ln(p_k/q_k)
\geq \sum_k(q_k-p_k)=0.
$$

Gleichheit gilt bei $p=q$. Jede Abweichung von den beobachteten Häufigkeiten erhöht bei diesem Kontext die Kreuzentropie. Ein Decoder berechnet die Verteilungen für viele Kontexte mit denselben Gewichten. Sein Trainingsverlust mittelt über alle zugehörigen Zielpositionen.

## Wie aus einer Verteilung ein Text wird

Bei den Scores $z_G=\ln 3$ und $z_H=0$ wählt die gierige Dekodierung stets `G`. Bei einer Zufallsziehung aus der Softmax-Verteilung wird an dieser Stelle im Mittel dreimal so häufig `G` wie `H` gezogen. Anschließend verlängert das gewählte Token die Eingabe. Jede weitere Ziehung hängt damit auch von den früheren Ziehungen ab.

Der Parameter **Temperatur** $\tau>0$ teilt die Scores vor der Auswahl durch $\tau$. Für zwei im Auswahlpool verbleibende Tokens folgt aus der Softmax sofort $p_\tau(G)/p_\tau(H)=\exp[(z_G-z_H)/\tau]=3^{1/\tau}$. Bei $\tau=2$ sinkt das Verhältnis von $3$ auf $\sqrt 3$. Bei sehr kleiner positiver Temperatur konzentriert sich die Masse auf das Maximum. Bei `--temperature 0` nimmt der Byte-Sampler das größte zulässige Logit direkt. `--top-k` kann den Pool zusätzlich auf die $k$ höchstbewerteten Kandidaten beschränken.

Für die Verlustmessung wertet `eval` alle 258 Logits mit der ursprünglichen Softmax aus. Bei der Texterzeugung lässt `sample` unter anderem nur im Training beobachtete Bytes und gültige UTF-8-Fortsetzungen zu. Der Antworttest `recall` nimmt an jeder Position das Maximum über alle 258 IDs und bei Gleichstand die kleinste ID. Bereits ein abweichendes Byte macht die ganze Sollantwort falsch. [Auswahl in `choose` und `sample`](../../src/main.c) · [Exakter Antworttest](../../src/recall.c)

## Rechenaufgaben

1. Die Zeile `Aä` wird mit Kontextlänge zwei gelesen. Schreibe beide Blöcke mit ihren Eingaben und Zielen auf. Wie viele Terme hat ihre Verlustsumme?
2. Eine Antwort besteht aus zehn Bytes. Ein hypothetisches Modell gibt den ersten neun richtigen Bytes unter der jeweils richtigen Vorgeschichte die Wahrscheinlichkeit $0{,}99$, dem letzten richtigen Byte $0{,}1$. Ein falsches Schlussbyte erhält $0{,}9$. Berechne die Perplexität dieser zehn Vorhersagen und entscheide, ob gieriges Dekodieren die ganze Antwort trifft.

<details>
<summary>Lösungen und Kontrollen</summary>

1. Die Folge ist `[256,65,195,164,257]`. Die erste Eingabe `[256,65]` hat die Ziele `[65,195]`, die zweite Eingabe `[195,164]` hat die Ziele `[164,257]`. Das sind vier Verlustterme. Die Grenze eines Rechenblocks darf zwischen den beiden Bytes des `ä` liegen.
2. Der mittlere Verlust beträgt $-(9\ln 0{,}99+\ln 0{,}1)/10 \simeq 0{,}2393$ Nats pro Zielbyte. Die PPL beträgt rund $1{,}27$. An den ersten neun Stellen hat das richtige Byte die höchste Wahrscheinlichkeit. Am Ende wählt der Decoder das falsche Byte, weil $0{,}9>0{,}1$ ist. Die Antwort zählt als falsch.

</details>

Wie kann eine Vorhersage das frühere `B` im Kontext nutzen, obwohl auf `In A liegt ` und `In B liegt ` dasselbe Leerzeichen folgt? Kapitel 2 verfolgt den Beitrag einer früheren Position bis zum aktuellen Score. [Weiter zu Kapitel 2](02-transformer.md) · [Übersicht](README.md)
