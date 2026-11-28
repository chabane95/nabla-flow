# 2 · Wie greift eine Vorhersage auf frühere Tokens zu?

In der ersten Zeile steht `Fach A: Glas. Fach B: Harz. In B liegt `, in der zweiten sind Glas und Harz vertauscht. Auf dieselbe Frage müsste einmal `H`, einmal `G` folgen. Das Bigramm verwendet in beiden Fällen die Tabellenzeile für das letzte Leerzeichen. Wie kann der frühere Eintrag über Fach B die Verteilung an dieser Stelle verändern?

Im Decoder erhält jede Position einen Zustandsvektor. Die **kausale Selbstaufmerksamkeit** (*causal self-attention*) mischt an der fraglichen Position Beiträge aus dem bisher gelesenen Text. Für die Rechnung sind zwei Entscheidungen auseinanderzuhalten. Welche Beiträge überhaupt zulässig sind, bestimmt ihre Position. Wie stark ein zulässiger Beitrag eingeht, hängt von den gelernten Gewichten und vom Text ab.

## Vom Token zum Zustand

Ein Fenster enthält die Token-IDs $u_0,\ldots,u_{T-1}$. An Position $t$ soll der Decoder eine Verteilung für $u_{t+1}$ liefern. $V$ ist die Vokabulargröße, $C$ die maximale Fensterlänge, $d$ die Breite eines Zustands. Es gilt $T\le C$. Alle Zustände einer einzelnen Position schreiben wir als Spaltenvektoren. Eine Gewichtsmatrix hat eine Zeile je Ausgabekomponente, genau wie bei der C-Funktion `dense`.

Die ID 76 bezeichnet das Byte `L`. Der Decoder verwendet diese Zahl als Index in der gelernten Tokentabelle $E\in\mathbb R^{V\times d}$. Algebraisch wählt der Einheitsvektor $e_{u_t}\in\mathbb R^V$ genau die Zeile mit Index $u_t$ aus. Zu dieser Zeile kommt die Zeile $t$ der gelernten Positionstabelle $P\in\mathbb R^{C\times d}$. Als Spaltenvektor geschrieben ergibt das

$$
x_t^{(0)}=E^\top e_{u_t}+P^\top e_t.
$$

Der Einheitsvektor $e_t$ hat hier $C$ Komponenten. Zwei Vorkommen von `B` teilen sich den Tokenanteil und erhalten an unterschiedlichen Stellen verschiedene Positionsanteile. Die Addition ist eine Architekturentscheidung. Sie lässt die Breite $d$ unverändert. [Einbettung in `gpt_forward`](../../src/gpt.c#L274-L293)

## Welche Position erhält welches Gewicht?

Ein ungewichteter Mittelwert gäbe jedem bisherigen Eintrag denselben Koeffizienten. Um bei verschiedenen Fragen andere Fundstellen gewichten zu können, berechnet ein Attention-Kopf drei Projektionen. Dabei bezeichnet $r_j$ den normierten Zustand an Position $j$. Die Rechnung für diese Normierung steht im Abschnitt über den Decoderblock.

$$
q_t=W_Qr_t+b_Q,\qquad
k_j=W_Kr_j+b_K,\qquad
v_j=W_Vr_j+b_V.
$$

Die **Query** $q_t$ und der **Key** $k_j$ bestimmen gemeinsam das Gewicht für die Verbindung von $j$ nach $t$. Der **Value** $v_j$ ist der Vektor, der mit diesem Gewicht übertragen wird. So können verschiedene Komponenten für die Wahl der Fundstelle und für die von dort übernommene Information zuständig sein. Alle Positionen eines Kopfes verwenden dieselben drei Matrizen und Biasvektoren. Ob sich ein Kopf im Fachversuch tatsächlich an einer bestimmten Fachbezeichnung orientiert, ist eine empirische Frage.

Für $H$ Köpfe hat jede Projektion eines Kopfes $d_h=d/H$ Ausgänge und somit eine Matrix der Größe $d_h\times d$. Das Skalarprodukt $q_t^\top k_j$ vergleicht Query und Key. Ohne Bias lautet es $r_t^\top W_Q^\top W_K r_j$. Die Matrix $W_Q^\top W_K$ kann asymmetrisch sein. Das Vertauschen der beiden Positionen kann daher sogar vor Anwendung der Maske einen anderen Score ergeben.

![Trainierte Gewichte des Fachmodells. Oben stehen die 32 Koordinaten für acht Byte-IDs, unten die acht Zeilen der Query-Matrix des ersten Kopfes im ersten Block.](../../figures/byte-weights.png)

Die Abbildung zeigt Parameter des Fachmodells beim Validierungsminimum des 1×-Laufs (Update 1.000). Oben stehen die 32 Koordinaten von acht ausgewählten Byte-IDs, unten die acht Zeilen der Query-Matrix des ersten Kopfes im ersten Block. Jede Zelle ist ein gespeicherter, dimensionsloser FP32-Parameter. Beide Matrizen verwenden die Farbskala von $-0{,}55$ bis $+0{,}55$. [Vektoransicht](../../figures/byte-weights.svg) · [Druckfassung](../../figures/byte-weights.pdf) · [Plot aus dem Checkpoint erzeugen](../../tools/plot_byte_weights.py)

Für `G` beträgt die erste Koordinate in der Tokentabelle rund $-0{,}065$. Die untere Matrix dient einer anderen Rechnung. Das Skalarprodukt ihrer ersten Zeile mit $r_t$, zuzüglich des ersten Biaswerts, liefert die erste Query-Komponente. Auch wenn `G` an zwei Stellen denselben Tabelleneintrag hat, können sich die Queries unterscheiden, weil die Positionsvektoren in $r_t$ eingehen. [Gewichte des Fachmodells](../../results/model.nfg)

Warum steht im Score der Faktor $1/\sqrt{d_h}$? Als Überschlagsrechnung seien die Query- und Key-Komponenten unabhängig, zentriert und von Varianz eins. Dann hat jedes Produkt $q_{t,i}k_{j,i}$ Erwartungswert null und Varianz eins. Die Summe über $d_h$ unabhängige Produkte hat Varianz $d_h$. Division durch ihre Standardabweichung $\sqrt{d_h}$ hält die typische Scoregröße beim Ändern der Kopfbreite konstant. Die Annahme der Unabhängigkeit dient dieser Abschätzung. Im trainierten Modell können die Komponenten korreliert sein.

$$
s_{tj}=\frac{q_t^\top k_j}{\sqrt{d_h}},\qquad 0\le j\le t.
$$

Die Beschränkung $j\leq t$ ist die kausale Maske. Das Token $u_t$ ist bei der Vorhersage bekannt. Die Position $t+1$ enthält bereits das gesuchte Ziel. In der C-Implementierung endet die Schleife über die Keys bei `j <= t`. [Projektion und kausale Attention im CPU-Code](../../src/gpt.c#L245-L301)

## Wie werden aus Scores Mischungsgewichte?

Der Kopf liefert $o_t=\sum_{j=0}^{t}a_{tj}v_j$. Sind alle $v_j$ gleich, soll $o_t$ ihren gemeinsamen Wert wiedergeben. Dafür müssen sich die Gewichte zu eins summieren. Mit $a_{tj}\geq0$ bleibt $o_t$ in der konvexen Hülle der Values. Diese Bedingungen allein lassen viele Verteilungen zu.

Eine Herleitung der Softmax gewichtet zwei Größen. Für die Zeile $t$ sei $a_j=a_{tj}$. Der erste Term ist der mittlere Score $\sum_j a_j s_{tj}$. Er wäre maximal, wenn alle Masse auf einer Position mit höchstem Score läge. Der zweite Term, die Entropie $-\sum_j a_j\ln a_j$, erreicht ihr Maximum bei Gleichverteilung. Bei gleichem Vorfaktor beider Terme und der Nebenbedingung $\sum_j a_j=1$ lautet die zu maximierende Funktion

$$
F(a,\lambda)=\sum_j a_j s_{tj}-\sum_j a_j\ln a_j
                +\lambda\left(1-\sum_j a_j\right).
$$

Im Inneren des zulässigen Bereichs gilt $0=\partial F/\partial a_j=s_{tj}-\ln a_j-1-\lambda$ und damit $a_j\propto\exp(s_{tj})$. Der Entropieterm ist streng konkav ($\partial^2 F/\partial a_j^2=-1/a_j$). Die normierte Lösung ist folglich das eindeutige Maximum. Für die Rechnung wird das Zeilenmaximum $M_t$ abgezogen.

$$
a_{tj}=\frac{\exp(s_{tj}-M_t)}{\sum_{i=0}^{t}\exp(s_{ti}-M_t)},
\qquad M_t=\max_{0\le i\le t}s_{ti}.
$$

Das Abziehen von $M_t$ kürzt denselben Faktor aus Zähler und Nenner. Der größte Exponentialterm ist jetzt eins, auch wenn ein Score etwa 1.000 beträgt. Gleiche Scores liefern gleiche Gewichte. Eine Differenz von $\ln 3$ ergibt ein Gewichtsverhältnis von $3:1$. Für $j>t$ gilt $a_{tj}=0$.

### Eine Zeile von Hand

An Position $t=1$ eines Fensters mit drei Tokens sei $d_h=2$ und $q_1=(\sqrt{2},0)^\top$. Die Tabelle enthält konstruierte Keys und Values nach der Projektion. Welche Zeilen gehen in den Nenner der Softmax ein?

| Position $j$ | Key $k_j$ | Value $v_j$ | Score für $q_1$ | Gewicht |
| --- | --- | --- | --- | --- |
| 0 | $(0,1)^\top$ | $(2,0)^\top$ | $0$ | $1/4$ |
| 1 | $(\ln 3,1)^\top$ | $(0,4)^\top$ | $\ln 3$ | $3/4$ |
| 2 | $(10,0)^\top$ | $(9,9)^\top$ | durch die Maske ausgeschlossen | $0$ |

Das zweite Skalarprodukt ist $\sqrt{2}\ln 3$. Nach Division durch $\sqrt{2}$ und Subtraktion des Zeilenmaximums bleiben die Scores $(-\ln 3,0)$. Ihre Exponentialwerte $(1/3,1)$ summieren sich zu $4/3$. Der Kopf liefert

$$
o_1=\tfrac14(2,0)^\top+\tfrac34(0,4)^\top=(1/2,3)^\top.
$$

Was geschieht, wenn die erste Komponente von $k_2$ auf eine Million steigt? Das Ergebnis bleibt $(1/2,3)^\top$, denn in dieser Zeile fehlt $j=2$ bereits in der Summe. Dasselbe Argument gilt für einen ganzen Decoder. Ändert sich nur der Text hinter einem festen Präfix, bleiben die Eingangsvektoren des Präfixes gleich. Jede seiner Attention-Zeilen liest ausschließlich aus diesem Bereich. Die übrigen Operationen wirken an jeder Position einzeln. Damit gilt die Aussage nach jedem weiteren Block. Der CPU-Selbsttest ersetzt in einer Fünferfolge die letzten zwei IDs und vergleicht die ersten drei Logit-Vektoren bitweise. Er wiederholt den Vergleich nach Kürzen auf die ersten drei IDs. [Kausalitätstest](../../src/gpt.c#L577-L602)

## Der Decoderblock

Jeder Kopf berechnet für die Position $t$ seine eigene Verteilung über die lesbaren Stellen. Das Aneinanderfügen der Kopfvektoren liefert $Hd_h=d$ Komponenten. Die Ausgangsmatrix $W_O\in\mathbb R^{d\times d}$ mischt diese Komponenten. Im C-Code sind die Q-, K- und V-Matrizen aller Köpfe zu einer Projektion $d\to 3d$ zusammengefasst. Jeder Kopf liest daraus seinen Ausschnitt. [Gewichtsanordnung](../../src/gpt.c#L41-L61) · [Attention-Schleifen](../../src/gpt.c#L245-L272)

Vor den Projektionen nimmt der Decoder für jede Position getrennt eine **LayerNorm** vor. Sie bestimmt Mittelwert und Varianz über die $d$ Komponenten des dortigen Zustandsvektors $x$. Es gilt

$$
\mu=\frac1d\sum_i x_i,\qquad
\sigma^2=\frac1d\sum_i(x_i-\mu)^2,\qquad
\operatorname{LN}(x)_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+10^{-5}}}+\beta_i.
$$

Eine gemeinsame additive Verschiebung aller Komponenten fällt durch die Subtraktion von $\mu$ weg. Bei $x=(1,3)^\top$ erhält man $\mu=2$ und $\sigma^2=1$. Mit $\gamma=(1,1)^\top$ und $\beta=(0,0)^\top$ ergibt sich näherungsweise $(-1,1)^\top$. Ein konstanter Eingangsvektor hat Varianz null und liefert dank des kleinen positiven Terms genau $\beta$. Die lernbaren Parameter $\gamma,\beta$ geben jeder Komponente wieder einen eigenen Maßstab und Nullpunkt. [LayerNorm auf der CPU](../../src/gpt.c#L216-L233)

Für einen Block mit Eingang $x_t$ ergeben sich zwei aufeinanderfolgende Änderungen. Zunächst geht die Mischung der Values aus dem Präfix in $m_t$ ein. Anschließend verarbeitet ein zweiter Zweig die Komponenten von $m_t$ an derselben Position.

$$
\begin{aligned}
r_t &= \operatorname{LN}_1(x_t),\\
m_t &= x_t + W_O [o_t^{(1)};\dots;o_t^{(H)}]+b_O,\\
n_t &= \operatorname{LN}_2(m_t),\\
f_t &= \operatorname{GELU}(W_1n_t+b_1),\\
x'_t &= m_t + W_2 f_t+b_2.
\end{aligned}
$$

Die eckige Klammer hängt die Kopfvektoren aneinander. Die Normalisierung steht jeweils **vor** der Verarbeitung durch den Zweig. Daher heißt diese Bauweise *Pre-LayerNorm*. Die beiden Additionen sind **Residualverbindungen**. Gibt ein Zweig den Nullvektor aus, steht sein Eingang für den nächsten Schritt weiterhin zur Verfügung.

Im zweiten Zweig erweitert $W_1$ den Vektor von $d$ auf $4d$ Komponenten. $W_2$ führt ihn auf $d$ Komponenten zurück. Zwei affine Abbildungen unmittelbar hintereinander ließen sich zu einer einzigen zusammenfassen. Zwischen ihnen wirkt deshalb eine nichtlineare Funktion auf jede Komponente des Zwischenvektors. GPT-2 verwendet GELU. Ihre Form lässt sich aus einem Schwellenversuch ablesen. Sei $\xi$ standardnormalverteilt. Wird $a$ für $\xi\le a$ und sonst null ausgegeben, beträgt der Erwartungswert $a\Pr(\xi\le a)=a\Phi(a)$. Der Code berechnet diesen Wert mit der Näherung

$$
\operatorname{GELU}(a)\approx \frac a2
\left[1+\tanh\left(\sqrt{2/\pi}\,(a+0.044715a^3)\right)\right].
$$

Für $a=0$ erhält man null. Für große positive Argumente geht der Faktor in der eckigen Klammer gegen zwei, für große negative gegen null. Sowohl GELU als auch die Breite $4d$ sind Entscheidungen dieser Architektur. [Beide Residualzweige und GELU in `gpt_forward`](../../src/gpt.c#L290-L308)

## Vom Zustand zur Vorhersage

Nach $L$ Blöcken normalisiert der Decoder den letzten Zustand nochmals. Mit $h_t=\operatorname{LN}_f(x_t^{(L)})$ erhält jedes mögliche nächste Token $k$ einen **Logit** $z_{t,k}$. Dazu wird dieselbe Tabellenzeile $E_{k,:}$ benutzt, aus der am Eingang der Vektor für Token $k$ stammt.

$$
z_{t,k}=\sum_{i=1}^d E_{k,i}\,h_{t,i},\qquad
p(u_{t+1}=k\mid u_0,\ldots,u_t)=
\frac{\exp(z_{t,k})}{\sum_{v=0}^{V-1}\exp(z_{t,v})}.
$$

In diesem Decoder treten zwei verschiedene Softmax-Verteilungen auf. Die Attention verteilt für jede Query das Gewicht über die **lesbaren Positionen**. Die Ausgabe verteilt für jede Position die Wahrscheinlichkeit über die **möglichen nächsten Token-IDs**. Ein Attention-Gewicht von $0{,}75$ ist damit ein Koeffizient eines Value-Vektors. Bis zur Tokenwahrscheinlichkeit folgen unter anderem die Ausgangsprojektion, der zweite Residualzweig und die Multiplikation mit der Tokentabelle.

Im CPU-Code erzeugt `gpt_forward` die Logits. `token_objective` wertet daraus die Kreuzentropie mit abgezogenem Zeilenmaximum aus. Die Ausgabeschicht hat keinen eigenen Bias. Die gemeinsame Matrix $E$ erhält beim Lernen Beiträge über ihren Eingangspfad und ihren Ausgangspfad. [Finale Normierung und Ausgabe](../../src/gpt.c#L307-L315) · [Berechnung der Zielfunktion](../../src/gpt.c#L410-L447)

### Zählen als unabhängige Kontrolle

| Bestandteil | Zahl der gelernten Skalare |
| --- | ---: |
| Einmalig für Token- und Positionstabelle | $(V+C)d$ |
| Q, K, V und Ausgangsprojektion pro Block, mit Bias | $4d^2+4d$ |
| MLP pro Block, mit Bias | $8d^2+5d$ |
| Zwei LayerNorms pro Block, je Skala und Offset | $4d$ |
| Finale LayerNorm | $2d$ |

Die Summe lautet $N=(V+C)d+L(12d^2+13d)+2d$. Im Fachmodell mit $V=258$, $C=64$, $d=32$, $L=2$ entspricht das $10\,304+2\cdot12\,704+64=35\,776$ Parametern. Für GPT-2 Small mit $V=50\,257$, $C=1\,024$, $d=768$, $L=12$ ergeben sich **124.439.808**. Das Trainingsfenster des Small-Laufs umfasste nur $T=128$ Positionen. Für die Parameterzahl zählt die volle Tabelle mit $C=1\,024$ Zeilen. Bei festem $d$ beeinflusst die Kopfzahl nur die Aufteilung $Hd_h=d$. [Parameterzählung in C](../../src/gpt.c#L101-L113) · [Messprotokoll des Bytemodells](../../results/summary.csv) · [Konfiguration des großen Laufs](../training-124m.md)

## Messung am gespeicherten Fachmodell

Welche Werte sind in den ersten beiden Zeilen für `Links: rot.` bereits durch die Maske festgelegt? Der Decoder fügt zuerst den Startmarker ein. Der folgende Befehl gibt aus dem zweiten Block, ersten Kopf die ersten beiden Query-Zeilen bis einschließlich Key-Position 2 aus. Die Indizes im CPU-Programm beginnen bei null. Der Befehl läuft im Hauptverzeichnis des Repositorys. [Bauanleitung für das CPU-Programm](../byte-models.md)

~~~powershell
.\nablaflow.exe inspect --model results/model.nfg --prompt "Links: rot." --layer 1 --head 0 |
    ConvertFrom-Csv | Where-Object { [int]$_.query_pos -lt 2 -and [int]$_.key_pos -lt 3 } |
    Format-Table query_pos,key_pos,weight
~~~

In Zeile 0 liegt das Gewicht eins auf dem Startmarker. Alle späteren Keys sind gesperrt. In Zeile 1 sind BOS und `L` lesbar. Die Messung ergibt $a_{1,0}=0{,}412516$ und $a_{1,1}=0{,}587484$ (gerundet). Ihr Verhältnis liefert direkt die Differenz der Scores, $s_{1,1}-s_{1,0}=\ln(a_{1,1}/a_{1,0})\approx0{,}354$. Eine beiden Scores gemeinsame Konstante kürzt sich heraus. [Vollständige Attention-Zeilen](../../results/attention.csv) · [Grafische Ansicht](../../figures/attention.svg)

[![Attention für den Prompt „Links: rot.“. Jede Zeile zeigt die Gewichte einer Query über die erlaubten früheren und aktuellen Positionen.](../../figures/attention.gif)](../../figures/attention.svg)

Die Animation ergänzt für jedes weitere Byte eine gemessene Zeile. Die Schraffur beginnt in Zeile $t$ stets bei Spalte $t+1$. Die Felder davor zeigen, wie der Kopf sein Gewicht innerhalb des zulässigen Bereichs verteilt. [Standbild als PDF](../../figures/attention.pdf) · [Messung](../../results/attention.csv)

Beim `t` von `rot` fällt das größte Gewicht auf das unmittelbar vorausgehende `o` ($a_{10,9}=0{,}711$). Der Value-Vektor am `o` geht also mit diesem Koeffizienten in die Summe $o_{10}$ ein. Um daraus eine Prognose für das nächste Byte zu berechnen, braucht man die Komponenten der Value-Vektoren und die nachfolgenden Schichten. [Messdaten für diese Abbildung](../../results/attention.csv)

**Rechenaufgabe.** Bei fester Breite $d$ verdopple sich die Zahl der Köpfe von $H$ auf $2H$. Voraussetzung ist, dass $2H$ die Breite $d$ teilt. Wie ändern sich Kopfbreite, Parameterzahl und Zahl der zulässigen Scores je Block bei Fensterlänge $T$?

**Lösung.** Die Kopfbreite halbiert sich. Jede der drei Projektionen behält über alle Köpfe hinweg $d$ Ausgänge, daher bleibt $N$ gleich. Ein Kopf benötigt $1+2+\cdots+T=T(T+1)/2$ zulässige Scores. Über alle Köpfe verdoppelt sich ihre Anzahl auf $2H\cdot T(T+1)/2$. Jedes einzelne Skalarprodukt verarbeitet dann halb so viele Komponenten.

Auf der GPU führt `forward` dieselben Operationen aus. Die dichten Matrixprodukte laufen über cuBLAS. Im CUDA-Kernel `attention` bearbeitet ein Threadblock eine Kombination aus Sequenz, Query-Position und Kopf. Jedes Beispiel im Batch liest damit ausschließlich aus der eigenen Sequenz. CPU- und GPU-Summen können sich wegen ihrer Summationsreihenfolge geringfügig unterscheiden. Die Paritätsprüfung misst unter anderem die Abweichungen der Logits und Gradienten. [GPU-Vorwärtspfad](../../src/gpu.c#L376-L400) · [CUDA-Attention](../../src/gpu_kernels.cu#L108-L133) · [CPU/GPU-Paritätsprüfung](../../src/gpu.c#L479-L547)

In der Handrechnung ist die erste Komponente von $o_1$ gleich $2a_{1,0}$. Wie müssen sich die beiden zulässigen Scores relativ zueinander ändern, damit diese Komponente kleiner wird? Kapitel 3 berechnet, wie ein Vorhersagefehler solche Änderungen über Gradienten auslöst. [Weiter zu Kapitel 3](03-lernen.md) · [Übersicht](README.md)
