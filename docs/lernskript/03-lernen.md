# 3 · Wie ein Vorhersagefehler ein Gewicht verändert

Im [Vorwärtspass](02-transformer.md) ergab eine Zeile der Attention den Vektor $(1/2,3)$. Als Probe nehmen wir die erste Komponente und verändern einen der beiden Scores um einen kleinen Betrag $h$. Die Ableitung liefert eine Vorhersage für diese Änderung. Eine zweite Auswertung der Vorwärtsrechnung prüft sie.

## Eine Störung als Gedankenexperiment

Bei festem Eingabetext und festen Ziel-IDs hängt der mittlere Tokenverlust $L$ nur von den Gewichten ab. Ändert sich allein das Gewicht $\theta_i$ um $h$, gilt in erster Ordnung

$$
L(\theta+h e_i)-L(\theta)=h\,\frac{\partial L}{\partial\theta_i}+O(h^2).
$$

Dabei bezeichnet $e_i$ den Einheitsvektor in Richtung des Gewichts. Für einen Zwischenvektor $u$ schreiben wir $\bar u$ für den Gradienten des Verlusts nach $u$. Sein Beitrag zum Differential des Verlusts ist $\bar u^\top\mathrm du$. Für $v=f(u)$ gilt $\mathrm dv=J_f\,\mathrm du$. Einsetzen in $\bar v^\top\mathrm dv$ ergibt $\bar u=J_f^\top\bar v$. Das ist der Rückwärtsschritt der Kettenregel. Geht $u$ in zwei Rechnungen ein, tritt dieselbe Variation $\mathrm du$ in beiden auf. Ihre Koeffizienten addieren sich.

Die Balken beziehen sich im Folgenden auf die Summe $S=\sum_t\ell_t$ über die Zielpositionen. Für den Mittelwert $L=S/N$ teilt der Trainer am Ende durch die Anzahl $N$ der Ziele. Vektoren sind Spalten. Eine Gewichtsmatrix hat wie in der C-Funktion `dense` eine Zeile je Ausgang und eine Spalte je Eingang. [Herleitung am einzelnen Neuron](../../latex/backprop_first_principles.tex) · [Konvention für die Gradientensumme](../../src/gpt.h)

## Vom beobachteten Token zur letzten Modellschicht

An Position $t$ hat das tatsächlich folgende Token die ID $y_t$. Aus dem Logit $z_{tj}$ ergibt sich die Wahrscheinlichkeit $p_{tj}=\exp z_{tj}/\sum_k\exp z_{tk}$ für Klasse $j$. Beim Ableiten von $\ell_t=-\ln p_{t y_t}$ liefert der Nenner einen Beitrag für jede Klasse, der Zähler nur für die beobachtete.

$$
\ell_t=\ln\sum_k e^{z_{tk}}-z_{t y_t},
\qquad
\mathrm d\ell_t=\sum_j p_{tj}\,\mathrm dz_{tj}-\mathrm dz_{t y_t},
\qquad
\bar z_{tj}=p_{tj}-\mathbf 1_{j=y_t}.
$$

Erhält das Ziel die Wahrscheinlichkeit $1/4$, beträgt die Ableitung nach seinem Logit $-3/4$. Erhöht man ihn um $h$, sinkt der Verlust in erster Ordnung um $3h/4$. Die Summe der Ableitungen verschwindet, weil eine gemeinsame Verschiebung aller Logits die Wahrscheinlichkeiten unverändert lässt. Der Code nutzt diese Symmetrie und zieht vor dem Exponenzieren das Zeilenmaximum ab. Der logarithmische Ausdruck für $\ell_t$ bleibt selbst dann auswertbar, wenn die Zielwahrscheinlichkeit in der Gleitkommarechnung auf null rundet. [`token_objective`](../../src/gpt.c#L410)

Für die Ausgabe gilt $z_t=E h_t$, mit dem abschließend normalisierten Zustand $h_t\in\mathbb R^d$ und der Tokenmatrix $E\in\mathbb R^{V\times d}$. Ändern sich beide, ergibt die Produktregel zwei Beiträge.

$$
\mathrm dz_t=(\mathrm dE)h_t+E\,\mathrm dh_t,
\qquad
\bar E_{\rm Ausgabe}=\sum_t\bar z_t h_t^\top,
\qquad
\bar h_t=E^\top\bar z_t .
$$

Die beiden Gradienten sind die Koeffizienten von $\mathrm dE_{ji}$ und $\mathrm dh_{ti}$ im Produkt $\bar z_t^\top\mathrm dz_t$. Für jede affine Schicht $v=Wu+b$ folgt entsprechend der Beitrag $\bar v u^\top$ zur Matrixableitung, $\bar v$ zur Biasableitung und $W^\top\bar v$ am Eingang. Die Beiträge verschiedener Positionen summieren sich, weil dort dieselbe Matrix wirkt. [`dense_backward`](../../src/gpt.c#L336)

## Durch eine LayerNorm zurück

Bei der LayerNorm beeinflusst jede Komponente auch die anderen. Eine Änderung von $x_i$ verschiebt den Mittelwert, der von allen Komponenten abgezogen wird. Mit $\langle a\rangle=d^{-1}\sum_i a_i$ lautet die Vorwärtsrechnung

$$
\mu=\langle x\rangle,\quad c_i=x_i-\mu,\quad
r=(\langle c^2\rangle+\varepsilon)^{-1/2},\quad
u_i=r c_i,\quad y_i=\gamma_i u_i+\beta_i,
\qquad \varepsilon=10^{-5}.
$$

Weil $\langle c\rangle=0$, gilt $\mathrm d\langle c^2\rangle=2\langle c\,\mathrm dx\rangle$. Die Kettenregel für die inverse Quadratwurzel liefert $\mathrm dr=-r^3\langle c\,\mathrm dx\rangle$. Setzt man das in $\mathrm du_i=c_i\,\mathrm dr+r(\mathrm dx_i-\mathrm d\mu)$ ein, erhält man

$$
\mathrm du_i=r\big(\mathrm dx_i-\langle\mathrm dx\rangle-u_i\langle u\,\mathrm dx\rangle\big).
$$

Mit $g_i=\gamma_i\bar y_i$ ist der Beitrag der normierten Komponenten zum Differential des Verlusts $\sum_i g_i\,\mathrm du_i$. Der Koeffizient von $\mathrm dx_i$ liefert den Eingangsgradienten. Die Ableitungen nach $\gamma_i$ und $\beta_i$ folgen aus $y_i=\gamma_i u_i+\beta_i$.

$$
\bar x_i^{\,\rm LN}=r\big(g_i-\langle g\rangle-u_i\langle gu\rangle\big),
\qquad
\bar\gamma_i\mathrel{+}=\bar y_i u_i,\qquad
\bar\beta_i\mathrel{+}=\bar y_i .
$$

**Symmetrieprobe.** Für $x_i\mapsto x_i+a$ bleibt $c_i$ für jedes $i$ gleich. Die Richtungsableitung entlang $(1,\ldots,1)$ muss also verschwinden. Tatsächlich summieren sich die $\bar x_i^{\,\rm LN}$ zu null, da $\langle u\rangle=0$. Im C-Code heißen die zwei Mittelwerte des Rückwegs `avg` und `projected`. `normalize_backward` addiert seinen Beitrag zu einem bereits vorhandenen Eingangsgradienten. [`normalize_backward`](../../src/gpt.c#L355)

## Durch Residuum und MLP

Der zweite Halbblock bildet $b=a+F(a)$ aus dem Ergebnis $a$ des ersten. Aus $\mathrm db=\mathrm da+J_F\,\mathrm da$ folgt $\bar a=\bar b+J_F^\top\bar b$. Der erste Summand stammt aus der direkten Verbindung. Der zweite durchläuft die Ausgabematrix des MLP, die Aktivierung, die Eingangsmatrix und die LayerNorm in umgekehrter Reihenfolge zur Vorwärtsrechnung.

Zwischen den beiden Matrizen wirkt die tanh-Näherung der GELU auf die $4d$ Komponenten. Mit $a_0=\sqrt{2/\pi}$, $b_0=0{,}044715$ und $u=\tanh(a_0(x+b_0 x^3))$ folgt aus $\phi(x)=x(1+u)/2$ die Ableitung

$$
\phi'(x)=\tfrac12(1+u)+\tfrac12 x(1-u^2)\,a_0(1+3b_0x^2),
\qquad
\bar x=\bar\phi\,\phi'(x).
$$

Die Formel gilt komponentenweise. Bei $x=0$ geht also die Hälfte des ankommenden Gradienten weiter. Für stark positive $x$ nähert sich die Steigung eins. Im Rückweg des ersten Halbblocks tritt die gleiche Residualaddition auf. Dessen Zweig führt über die Ausgangsprojektion zu den Kopfvektoren der Attention. [`gpt_backward`](../../src/gpt.c#L454-L513)

## Durch eine kausale Attention-Zeile

In einem Kopf der Breite $d_h$ hat Position $t$ Zugriff auf die Positionen $j\leq t$. Ihre Scores sind $s_{tj}=q_t^\top k_j/\sqrt{d_h}$, die Gewichte $a_{tj}$ deren Softmax. Im Kopfvektor $o_t=\sum_{j\le t}a_{tj}v_j$ können sich sowohl die gelesenen Werte als auch ihre Gewichte ändern.

$$
\mathrm do_t=\sum_{j\le t} a_{tj}\,\mathrm dv_j+\sum_{j\le t}v_j\,\mathrm da_{tj}.
$$

Der Wert $v_j$ erhält den Beitrag $a_{tj}\bar o_t$. Für das Gewicht gilt $\bar a_{tj}=\bar o_t^\top v_j$. Um daraus die Score-Ableitung zu erhalten, differenziert man den Quotienten im Softmax. Innerhalb der Zeile ist $\mathrm da_j=a_j(\mathrm ds_j-\sum_m a_m\,\mathrm ds_m)$. Der Koeffizient von $\mathrm ds_j$ in $\sum_m\bar a_m\,\mathrm da_m$ ist deshalb $a_j(\bar a_j-\sum_m a_m\bar a_m)$. Der gewichtete Mittelwert in der Klammer ist $\bar o_t^\top o_t$. Es folgt

$$
\bar s_{tj}=a_{tj}\,\bar o_t^\top(v_j-o_t).
$$

Die Produktregel für $q_t^\top k_j$ liefert die Gradienten nach Query und Key. Die Summen laufen jeweils über die von der Maske zugelassenen Positionen.

$$
\bar q_t=\sum_{j\le t}\frac{\bar s_{tj}k_j}{\sqrt{d_h}},
\qquad
\bar k_j=\sum_{t\ge j}\frac{\bar s_{tj}q_t}{\sqrt{d_h}},
\qquad
\bar v_j=\sum_{t\ge j}a_{tj}\bar o_t .
$$

Ein früher Wert kann also Fehlerbeiträge von mehreren späteren Vorhersagen erhalten. Für $j>t$ ist der Beitrag der Zeile $t$ genau null. In der CPU-Schleife enthält `ds` bereits den Faktor $1/\sqrt{d_h}$. Die drei Gradienten gehen danach durch die gemeinsame affine QKV-Projektion zurück. [`attention_backward`](../../src/gpt.c#L377)

**Zahlenprobe.** In Kapitel 2 waren $a_{10}=1/4$, $a_{11}=3/4$, $v_0=(2,0)$, $v_1=(0,4)$ und $o_1=(1/2,3)$. Für die zu Beginn gewählte erste Komponente gilt $\bar o_1=(1,0)$. Ihre beiden Steigungen sind

$$
\bar s_{10}=\tfrac14(2-\tfrac12)=\tfrac38,\qquad
\bar s_{11}=\tfrac34(0-\tfrac12)=-\tfrac38.
$$

Erhöht sich nur $s_{10}=0$ um $h$, wächst die erste Komponente um $3h/8+O(h^2)$. Zur direkten Kontrolle bleibt $s_{11}=\ln 3$ fest. Die Vorwärtsrechnung ergibt dann $2e^h/(e^h+3)$, mit der Steigung $3/8$ bei $h=0$. Eine gemeinsame Verschiebung beider Scores muss die Mischung erhalten. Dazu passt $3/8-3/8=0$.

## Ein Gewicht mit zwei Wirkungen

Der Rückweg endet an $x_t=E_{i_t,:}^\top+P_{t,:}^\top$, der Summe der Token- und Positionsvektoren. Die Eingabe-ID $i_t$ wählt eine Zeile aus. Ändert sich diese Zeile, ändern sich **alle** Eingangspositionen mit dieser ID. Außerdem liest die Ausgangsprojektion dieselbe Zeile für jedes vorhergesagte Token. Für die einzelne Komponente $k$ einer Zeile $j$ lautet die gesamte Tokenableitung

$$
\bar E_{jk}=\underbrace{\sum_t \bar z_{tj}h_{tk}}_{\text{Ausgabe}}
            +\underbrace{\sum_{t:\,i_t=j}\bar x_{tk}}_{\text{Eingabe}}.
$$

Der Positionsvektor $P_{t,:}$ erhält den Eingangsgradienten seiner Position, bei mehreren Fenstern summiert über deren Beiträge. Für das Tokengewicht schreibt `token_objective` zuerst den Ausgabebeitrag. Am Ende von `gpt_backward_weighted` addiert `dtoken[i] += ...` den Eingabebeitrag an dieselbe Adresse. [Ausgabeprojektion](../../src/gpt.c#L410) · [Eingangs- und Positionsgradient](../../src/gpt.c#L503)

**Kontrollmodell mit einem Gewicht.** Zwei Token haben die skalaren Einträge $(w,0)$. Das erste ist Eingabe, das zweite Ziel. Ohne Blöcke und Normierung bleibt der Zustand $w$, die beiden Logits sind $(w^2,0)$. Der Verlust beträgt $\ln(1+e^{w^2})$. Bei $w=1$ ist $p_0=e/(e+1)$. Die Ausgabe liefert zur Ableitung den Beitrag $p_0 w$, die Eingabe einen weiteren Beitrag $p_0 w$. Die Gesamtsteigung ist also $2p_0\approx1{,}46212$.

## Einen Gradienten messen

Eine unabhängige Messung benutzt nur die Vorwärtsrechnung mit identischen Daten bei $\theta_i+h$ und $\theta_i-h$. Die beiden Taylorentwicklungen um $\theta_i$ liefern

$$
\frac{L(\theta+h e_i)-L(\theta-h e_i)}{2h}
=\partial_i L(\theta)+O(h^2).
$$

Der gerade Term fällt beim Subtrahieren weg. Bei zu kleinem $h$ werden die beiden Verlustwerte im Rechner fast ununterscheidbar. Im Kontrollmodell liefert $h=0{,}001$ in doppelter Genauigkeit $1{,}46211743$ für die Steigung bei $w=1$, gegenüber $1{,}46211716$ aus der Ableitung.

`.\nablaflow.exe check` unter PowerShell bzw. `./nablaflow check` unter Linux und macOS wiederholt diese Probe für alle 1 128 Parameter eines kleinen Decoders. Die Eingabe enthält eine ID zweimal, sodass beide Beiträge zum Eingabegewicht eingehen müssen. Der nominelle Schritt beträgt $h=10^{-3}$. Als Nenner verwendet `test_gradients` den tatsächlichen Abstand der gespeicherten FP32-Werte. Bei fünf Zielpositionen ist `s->grad[i]/5` der Vergleichswert für den mittleren Verlust. Die größte absolute Abweichung im Referenzlauf betrug $6{,}78\cdot10^{-5}$. [Bauanleitung und Referenzwert](../byte-models.md) · [`test_gradients`](../../src/gpt.c#L604)

## Vom Gradienten zum Optimierungsschritt

Für eine kleine Änderung $\Delta\theta$ ist $\Delta L\approx g^\top\Delta\theta$, mit $g=\nabla L$. Unter der Nebenbedingung $\|\Delta\theta\|_2=\rho$ ist der kleinste lineare Beitrag $-\rho\|g\|_2$. Er wird bei $g\ne0$ in Richtung $-g$ erreicht. Das motiviert einen Abstiegsschritt. Das Projekt wählt AdamW, der jede Koordinate anhand der bisher beobachteten Gradienten skaliert.

1. **Über Ziele mitteln.** Aus Fenstern mit zusammen $N$ vorhergesagten Tokens erhält man $g=(\sum_t\nabla\ell_t)/N$. Sonst würde ein kurzes Restfenster beim Mitteln über Fenster pro Token stärker zählen.
2. **Die globale Norm begrenzen.** Bei $\|g\|_2>c>0$ ersetzt man $g$ durch $\tilde g=cg/\|g\|_2$, sonst gilt $\tilde g=g$. Die Richtung bleibt erhalten. `grad_norm` protokolliert $\|g\|_2$ vor dem Begrenzen. Der erste 124M-Batch liefert $26{,}7278843$. Bei $c=1$ gehen alle Komponenten mit dem gemeinsamen Faktor $0{,}0374141$ in die Momentenschätzung ein. [Erste Batchmessung](../../results/gpt2-124m/batches.csv)
3. **Zwei zeitliche Mittel bilden.** Für Schritt $n$ und eine einzelne Koordinate gelten mit $m_0=v_0=0$

   $$
   m_n=\beta_1 m_{n-1}+(1-\beta_1)\tilde g_n,\qquad
   v_n=\beta_2 v_{n-1}+(1-\beta_2)\tilde g_n^2 .
   $$

   $m$ hält die mittlere Richtung fest, $v$ erfasst die Größenordnung der quadrierten Ableitungen. Ein konstantes Signal $a$ ergibt für das erste Mittel $m_n=a(1-\beta_1^n)$. Dies erklärt die Anfangskorrekturen $\hat m_n=m_n/(1-\beta_1^n)$ und $\hat v_n=v_n/(1-\beta_2^n)$. Das Projekt verwendet $\beta_1=0{,}9$ und $\beta_2=0{,}999$.
4. **Gewicht aktualisieren.** Für die Lernrate $\eta_n$ gilt pro Parameter

   $$
   \theta_n=(1-\eta_n\lambda)\theta_{n-1}
       -\eta_n\,\frac{\hat m_n}{\sqrt{\hat v_n}+10^{-8}} .
   $$

   Der Faktor $1-\eta_n\lambda$ verkleinert Matrixgewichte separat von den Momenten. Dazu zählen auch Token- und Positionsmatrizen. Für Biasvektoren sowie $\gamma,\beta$ der Normierung setzt der Code $\lambda=0$. Der Trainer für die Byte-Modelle verwendet für die Matrizen $\lambda=0{,}01$. Die GPT-2-Small-Trainer auf CPU und GPU verwenden $\lambda=0{,}1$. Alle drei begrenzen die Gradientennorm auf eins. [Update](../../src/gpt.c#L529) · [Byte-Training](../../src/main.c#L165) · [GPT-2 auf der CPU](../../src/scale.c#L84) · [GPT-2 auf der GPU](../../tools/nfgpu.c#L192)

## Vergleich zwischen CPU und GPU

Auf der CPU summiert `gpt_backward` die Beiträge eines Fensters, der Trainer ruft die Funktion für weitere Fenster wiederholt auf. Die GPU verarbeitet mehrere Fenster eines Batches gemeinsam. Der CUDA-Kernel `cross_entropy` überschreibt im Rückwärtsmodus die gespeicherten Logits mit $p-\mathrm{onehot}(y)$. cuBLAS berechnet die affinen Matrixprodukte. `embedding_back` addiert Beiträge parallel mit `atomicAdd`, etwa wenn mehrere Positionen dieselbe ID enthalten. Ab Breite 64 erlaubt der Host für Matrixprodukte TF32 bei FP32-Speicherung. Rundung und Reihenfolge der Additionen beeinflussen daher den numerischen Vergleich. [GPU-Rückweg](../../src/gpu.c#L402) · [CUDA-Kernel](../../src/gpu_kernels.cu#L40)

`nfgpu check` vergleicht CPU und GPU bei Breiten 12 und 64. Neben den Logits prüft er den akkumulierten Gradienten, die Norm vor dem Begrenzen und die Wirkung zweier AdamW-Schritte. Die CPU-Differenzenprobe stellt eine andere Frage. Sie misst, wie sich die **Vorwärtsrechnung** beim Verändern eines Gewichts verhält. Zwei übereinstimmende Rückwärtsimplementierungen könnten denselben Ableitungsfehler enthalten. [Paritätstest](../../src/gpu.c#L479) · [Berichtete Messwerte](../training-124m.md#numerical-checks)

## Drei Rechenproben

**1. Gleiche Werte, unterschiedliche Scores.** Eine Attention-Zeile sieht zwei identische Vektoren $v_0=v_1=v$. Kann ein Fehlergradient aus ihrem Kopfvektor die beiden Scores auseinanderziehen? Was erhält jeder Wert?

<details>
<summary>Lösungsweg</summary>

Da die Gewichte zusammen eins ergeben, gilt $o_t=v$ für jede mögliche Score-Wahl. Daher verschwinden beide $\bar s_{tj}$. Die Werte erhalten für diese Zeile weiterhin $a_{t0}\bar o_t$ und $a_{t1}\bar o_t$. Bei der ersten Position eines kausalen Fensters tritt derselbe Score-Grenzfall schon mit nur einem erlaubten Wert auf.

</details>

**2. Zwei Fenster mit verschiedenen Längen.** Ein einzelnes Gewicht hat an den fünf Zielen des ersten Fensters jeweils die Ableitung $+1$, an den zwei Zielen des zweiten jeweils $-1$. Wie groß ist der korrekte Gradient? Welche Zahl bekäme man durch gleiches Gewichten der Fenstermittel?

<details>
<summary>Lösungsweg</summary>

Die Gradientensumme ist $5-2=3$ und der Mittelwert $3/7$. Mit gleichem Gewicht für beide Fenstermittel ergäbe sich $(1-1)/2=0$. Genau diese Unterscheidung prüft `test_optimizer` mit fünf und zwei Zielen. [Optimierertest](../../src/gpt.c#L637)

</details>

**3. Erster AdamW-Schritt von Hand.** Eine Matrixkomponente hat den Wert $\theta=2$ und nach Clipping die Ableitung $\tilde g=-0{,}5$. Berechne aus dem Ruhezustand den ersten Schritt bei $\eta=0{,}01$, $\lambda=0{,}1$. Was entscheidet über das Vorzeichen der Änderung?

<details>
<summary>Lösungsweg</summary>

$m_1=-0{,}05$ und $v_1=0{,}00025$. Nach Anfangskorrektur sind $\hat m_1=-0{,}5$ und $\hat v_1=0{,}25$. Der Adam-Term erhöht das Gewicht um fast $0{,}01$, der Zerfall verringert es um $0{,}002$. Damit ergibt sich $\theta_1\approx2{,}008$. Das Vorzeichen ergibt sich aus der Summe beider Beiträge.

</details>

[Weiter zum Labor](04-labor.md) · [Übersicht](README.md)
