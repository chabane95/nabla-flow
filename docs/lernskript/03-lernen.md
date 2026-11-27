# 3 · Wie ein Vorhersagefehler ein Gewicht verändert

Im [Vorwärtspass](02-transformer.md) liefert eine Attention-Zeile den Vektor $(1/2,3)$. Verwende zur Probe seine erste Komponente als Verlust. Du darfst einen der beiden Attention-Scores ein wenig verkleinern. Welchen wählst du, damit der Verlust sinkt? Am Ende des Rückwegs können wir ausrechnen, um wie viel er sich dabei ändert.

Du brauchst die Produkt- und Kettenregel und das Skalarprodukt. Halte beim Lesen einen Unterschied fest. Die Ableitung sagt, wie ein vorgegebener Verlust auf eine kleine Störung reagiert. Die Regel, nach der wir anschließend tatsächlich die Gewichte ändern, ist eine eigene Modellierungsentscheidung.

## Eine Störung als Gedankenexperiment

Halte Eingabetext und Ziel-IDs fest. Verändere genau ein Gewicht $\theta_i$ um ein kleines $h$. Die Vorhersage für die Änderung des mittleren Tokenverlusts $L$ lautet

$$
L(\theta+h e_i)-L(\theta)=h\,\frac{\partial L}{\partial\theta_i}+O(h^2).
$$

Hier ist $e_i$ der Einheitsvektor in der gewählten Parameterrichtung. Wie berechnet man alle diese Steigungen mit einem einzigen Rückweg? Sei $u$ ein Zwischenvektor und $\bar u$ der Vektor der partiellen Ableitungen des betrachteten skalaren Verlusts nach $u$. Eine beliebige kleine Änderung $\mathrm du$ trägt dann $\bar u^\top\mathrm du$ zum Differential des Verlusts bei. Für $v=f(u)$ gilt $\mathrm dv=J_f\,\mathrm du$. Einsetzen liefert $\bar v^\top J_f\,\mathrm du$, also $\bar u=J_f^\top\bar v$. Man braucht nur dieses Produkt mit der transponierten Jacobi-Matrix. Wird $u$ an zwei Stellen gelesen, erzeugt dasselbe $\mathrm du$ zwei Beiträge. Ihre Summe ist der Gradient an $u$.

Im Folgenden beziehen sich die Balken auf die **Summe** $S=\sum_t\ell_t$ der Verluste aller betrachteten Zielpositionen. Für den Mittelwert $L=S/N$ teilen wir am Schluss durch die Anzahl $N$ der Ziele. Vektoren einer einzelnen Position sind Spalten, eine Gewichtsmatrix hat die Form $(\text{Ausgänge},\text{Eingänge})$. So speichert sie auch die C-Funktion `dense`. [Herleitung am einzelnen Neuron](../../latex/backprop_first_principles.tex) · [Konvention für die Gradientensumme](../../src/gpt.h)

## Vom beobachteten Token zur letzten Modellschicht

An Position $t$ ist $y_t$ die ID des tatsächlich folgenden Tokens, $z_t$ der Vektor der Logits und $p_{tj}=\exp z_{tj}/\sum_k\exp z_{tk}$. Schreiben wir $\ell_t=-\ln p_{t y_t}$ als Differenz, liefert der erste Term beim Ableiten die Wahrscheinlichkeit jeder Klasse und der zweite markiert die beobachtete Klasse.

$$
\ell_t=\ln\sum_k e^{z_{tk}}-z_{t y_t},
\qquad
\mathrm d\ell_t=\sum_j p_{tj}\,\mathrm dz_{tj}-\mathrm dz_{t y_t},
\qquad
\bar z_{tj}=p_{tj}-\mathbf 1_{j=y_t}.
$$

Wenn das Modell dem Ziel die Wahrscheinlichkeit $1/4$ gibt, ist dessen Logit-Ableitung $-3/4$. Eine kleine Erhöhung dieses Logits senkt den Verlust. Die Ableitungen aller Logits einer Position summieren sich zu null. Das folgt auch ohne Rechnung aus der Invarianz des Softmax unter einer gemeinsamen Verschiebung aller Logits. Im Code wird das größte Logit vor dem Exponenzieren abgezogen. Für den Verlust wird die Log-Summe direkt ausgewertet. Dadurch bleibt selbst bei numerisch auf null gerundeter Zielwahrscheinlichkeit ein endlicher Verlust berechenbar. [`token_objective`](../../src/gpt.c#L410)

Für die Ausgabe gilt $z_t=E h_t$, mit dem abschließend normalisierten Zustand $h_t\in\mathbb R^d$ und der Tokenmatrix $E\in\mathbb R^{V\times d}$. Ändern sich beide, ergibt die Produktregel zwei Beiträge.

$$
\mathrm dz_t=(\mathrm dE)h_t+E\,\mathrm dh_t,
\qquad
\bar E_{\rm Ausgabe}=\sum_t\bar z_t h_t^\top,
\qquad
\bar h_t=E^\top\bar z_t .
$$

Man liest die beiden letzten Ausdrücke ab, indem man $\bar z_t^\top\mathrm dz_t$ ausmultipliziert und die Koeffizienten von $\mathrm dE_{ji}$ und $\mathrm dh_{ti}$ sammelt. Dieselbe Rechnung gilt für jede affine Schicht $v=Wu+b$. Zu $\bar W$ kommt $\bar v u^\top$, zu $\bar b$ kommt $\bar v$, am Eingang kommt $W^\top\bar v$ an. Weil eine Schicht an vielen Positionen dieselben Gewichte verwendet, werden deren Beiträge summiert. [`dense_backward`](../../src/gpt.c#L336)

## Durch eine LayerNorm zurück

Die letzte Operation vor der Ausgabe und beide Zweige in jedem Block enthalten eine LayerNorm. Schon das Ändern einer einzigen Vektorkomponente verschiebt ihren Mittelwert. Wie viel von der Änderung erreicht die nächste Schicht?

Betrachte eine Position. Schreibe $\langle a\rangle=d^{-1}\sum_i a_i$ für den Mittelwert über deren $d$ Komponenten. Die Vorwärtsrechnung ist

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

Mit $g_i=\gamma_i\bar y_i$ ist der eingangsabhängige Verlustbeitrag $\sum_i g_i\,\mathrm du_i$. Sortieren nach $\mathrm dx_i$ ergibt den Beitrag der LayerNorm zum Eingangsgradienten. Die beiden Parameterableitungen folgen direkt aus der affinen Abbildung $y_i=\gamma_i u_i+\beta_i$.

$$
\bar x_i^{\,\rm LN}=r\big(g_i-\langle g\rangle-u_i\langle gu\rangle\big),
\qquad
\bar\gamma_i\mathrel{+}=\bar y_i u_i,\qquad
\bar\beta_i\mathrel{+}=\bar y_i .
$$

**Symmetrieprobe.** Für $x_i\mapsto x_i+a$ bleibt $c_i$ für jedes $i$ gleich. Die Richtungsableitung entlang $(1,\ldots,1)$ muss also verschwinden. Tatsächlich summieren sich die $\bar x_i^{\,\rm LN}$ zu null, da $\langle u\rangle=0$. Im C-Code heißen die zwei Mittelwerte des Rückwegs `avg` und `projected`. `normalize_backward` addiert seinen Beitrag zu einem bereits vorhandenen Eingangsgradienten. [`normalize_backward`](../../src/gpt.c#L355)

## Durch Residuum und MLP

Für den zweiten Halbblock gilt $b=a+F(a)$, wobei $a$ das Ergebnis des ersten Halbblocks ist. Da $\mathrm db=\mathrm da+J_F\,\mathrm da$, erreicht $\bar a=\bar b+J_F^\top\bar b$ den Knoten $a$. Der erste Summand läuft direkt durch die Addition. Beim zweiten geht es nacheinander durch die Ausgabematrix des MLP, seine Aktivierung, die Eingangsmatrix und die LayerNorm. Im MLP wird die Dimension von $d$ auf $4d$ erweitert und dann wieder auf $d$ verkleinert.

Für die Aktivierung benutzt das Projekt die tanh-Näherung der GELU. Setze $a_0=\sqrt{2/\pi}$, $b_0=0{,}044715$ und $u=\tanh(a_0(x+b_0 x^3))$. Aus $\phi(x)=x(1+u)/2$ folgen mit der Produktregel

$$
\phi'(x)=\tfrac12(1+u)+\tfrac12 x(1-u^2)\,a_0(1+3b_0x^2),
\qquad
\bar x=\bar\phi\,\phi'(x).
$$

Die Formel gilt komponentenweise. Bei $x=0$ geht also die Hälfte des ankommenden Gradienten weiter. Für stark positive $x$ nähert sich die Steigung eins. Im Rückweg des ersten Halbblocks tritt die gleiche Residualaddition auf. Dessen Zweig führt über die Ausgangsprojektion zu den Kopfvektoren der Attention. [`gpt_backward`](../../src/gpt.c#L454-L513)

## Durch eine kausale Attention-Zeile

Für einen Kopf der Breite $d_h$ liest Position $t$ die Positionen $j\leq t$. Ihre Scores sind $s_{tj}=q_t^\top k_j/\sqrt{d_h}$, die Gewichte $a_{tj}$ deren Softmax und der Kopfvektor ist $o_t=\sum_{j\le t}a_{tj}v_j$. Fixiere $t$. Die Produktregel erfasst Änderungen der gelesenen Werte und ihrer Gewichtung.

$$
\mathrm do_t=\sum_{j\le t} a_{tj}\,\mathrm dv_j+\sum_{j\le t}v_j\,\mathrm da_{tj}.
$$

Der Wert $v_j$ erhält den Beitrag $a_{tj}\bar o_t$. Für das Gewicht gilt $\bar a_{tj}=\bar o_t^\top v_j$. Um daraus die Score-Ableitung zu erhalten, differenziert man den Quotienten im Softmax. Innerhalb der Zeile ist $\mathrm da_j=a_j(\mathrm ds_j-\sum_m a_m\,\mathrm ds_m)$. Der Koeffizient von $\mathrm ds_j$ in $\sum_m\bar a_m\,\mathrm da_m$ ist deshalb $a_j(\bar a_j-\sum_m a_m\bar a_m)$. Der gewichtete Mittelwert in der Klammer ist $\bar o_t^\top o_t$. Es folgt

$$
\bar s_{tj}=a_{tj}\,\bar o_t^\top(v_j-o_t).
$$

Nun nochmals die Produktregel, diesmal für $q_t^\top k_j$. Über alle erlaubten Kanten ergeben sich

$$
\bar q_t=\sum_{j\le t}\frac{\bar s_{tj}k_j}{\sqrt{d_h}},
\qquad
\bar k_j=\sum_{t\ge j}\frac{\bar s_{tj}q_t}{\sqrt{d_h}},
\qquad
\bar v_j=\sum_{t\ge j}a_{tj}\bar o_t .
$$

Ein früher Wert kann also Fehlerbeiträge von mehreren späteren Vorhersagen erhalten. Für $j>t$ ist der Beitrag der Zeile $t$ genau null. In der CPU-Schleife enthält `ds` bereits den Faktor $1/\sqrt{d_h}$. Die drei Gradienten gehen danach durch die gemeinsame affine QKV-Projektion zurück. [`attention_backward`](../../src/gpt.c#L377)

**Handrechnung am Beispiel aus Kapitel 2.** Es gilt $a_{10}=1/4$, $a_{11}=3/4$, $v_0=(2,0)$, $v_1=(0,4)$ und $o_1=(1/2,3)$. Wähle die erste Komponente von $o_1$ als skalaren Verlust für diese Probe. Dann ist $\bar o_1=(1,0)$. Erhält der erste Wert mehr Gewicht, wächst die erste Komponente. Einsetzen liefert die Steigungen bezüglich beider Scores.

$$
\bar s_{10}=\tfrac14(2-\tfrac12)=\tfrac38,\qquad
\bar s_{11}=\tfrac34(0-\tfrac12)=-\tfrac38.
$$

Eine kleine Senkung des **ersten** Scores senkt diesen Verlust. Eine gemeinsame Verschiebung beider Scores hat wegen der entgegengesetzten Ableitungen keinen Effekt. Noch direkter lässt sich das erste Vorzeichen prüfen. Erhöhe $s_{10}=0$ um $h$ und halte $s_{11}=\ln 3$ fest. Die erste Komponente ist dann $2e^h/(e^h+3)$. Ihre Steigung bei null ist $3/8$.

## Ein Gewicht mit zwei Wirkungen

Der Rückweg endet an $x_t=E_{i_t,:}^\top+P_{t,:}^\top$, der Summe der Token- und Positionsvektoren. Die Eingabe-ID $i_t$ wählt eine Zeile aus. Ändert sich diese Zeile, ändern sich **alle** Eingangspositionen mit dieser ID. Außerdem liest die Ausgangsprojektion dieselbe Zeile für jedes vorhergesagte Token. Für die einzelne Komponente $k$ einer Zeile $j$ lautet die gesamte Tokenableitung

$$
\bar E_{jk}=\underbrace{\sum_t \bar z_{tj}h_{tk}}_{\text{Ausgabe}}
            +\underbrace{\sum_{t:\,i_t=j}\bar x_{tk}}_{\text{Eingabe}}.
$$

Der Positionsvektor $P_{t,:}$ erhält den Eingangsgradienten seiner Position, bei mehreren Fenstern summiert über deren Beiträge. Man kann den Lebensweg eines Eintrags im Speicher verfolgen. `token_objective` schreibt zuerst den Ausgabebeitrag, am Ende von `gpt_backward` addiert `dtoken[i] += ...` den Eingabebeitrag an dieselbe Adresse. [Ausgabeprojektion](../../src/gpt.c#L410) · [Eingangs- und Positionsgradient](../../src/gpt.c#L503)

**Kontrollmodell mit nur einem freien Gewicht.** Streiche für dieses Handexperiment die Blöcke und die Normierung. Zwei Token haben die skalaren Einträge $(w,0)$. Das erste ist Eingabe, das zweite Ziel. Der Zustand ist $w$, die Logits sind $(w^2,0)$ und der Verlust ist $\ln(1+e^{w^2})$. Bei $w=1$ ist $p_0=e/(e+1)$. Die Ausgabe liefert $p_0 w$, der Rückweg zum gelesenen Eingabevektor nochmals $p_0 w$. Zusammen ergeben sie $2p_0\approx1{,}46212$. Wer einen Pfad vergisst, erhält bei dieser Probe genau die halbe Steigung.

## Einen Gradienten messen

Für ein Gewicht $\theta_i$ werte mit identischen Daten den mittleren Verlust bei $\theta_i+h$ und $\theta_i-h$ aus. Taylorentwicklung an beiden Stellen und Subtraktion liefern

$$
\frac{L(\theta+h e_i)-L(\theta-h e_i)}{2h}
=\partial_i L(\theta)+O(h^2).
$$

Der gerade Term fällt beim Subtrahieren weg. Bei zu kleinem $h$ werden die beiden Verlustwerte im Rechner fast ununterscheidbar. Im Kontrollmodell liefert $h=0{,}001$ in doppelter Genauigkeit $1{,}46211743$ für die Steigung bei $w=1$, gegenüber $1{,}46211716$ aus der Ableitung.

**Experiment im Repository.** Führe aus dem Projektverzeichnis unter PowerShell `.\nablaflow.exe check` aus, unter Linux oder macOS `./nablaflow check`. Die Bauanleitung steht bei den Byte-Modellen. `test_gradients` verwendet fünf Zielpositionen und variiert jeden der 1 128 Parameter eines kleinen Decoders um nominell $10^{-3}$. Die Eingabe enthält dieselbe ID zweimal. Eine falsche Zuweisung anstelle einer Addition in der Embedding-Ableitung wird dadurch messbar. Im Nenner verwendet der Test den tatsächlichen Abstand der beiden gespeicherten FP32-Gewichte. Verglichen wird mit `s->grad[i]/5`, weil `gpt_loss` einen Mittelwert liefert. Die größte absolute Abweichung im dokumentierten Referenzlauf war $6{,}78\cdot10^{-5}$. [Bauanleitung und Referenzwert](../byte-models.md) · [`test_gradients`](../../src/gpt.c#L604)

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

## Welche Rechnung überprüft die GPU?

Auf der CPU summiert `gpt_backward` die Beiträge eines Fensters, der Trainer ruft die Funktion für weitere Fenster wiederholt auf. Die GPU verarbeitet mehrere Fenster eines Batches gemeinsam. Der CUDA-Kernel `cross_entropy` überschreibt im Rückwärtsmodus die gespeicherten Logits mit $p-\mathrm{onehot}(y)$. cuBLAS berechnet die affinen Matrixprodukte. `embedding_back` addiert Beiträge parallel mit `atomicAdd`, etwa wenn mehrere Positionen dieselbe ID enthalten. Ab Breite 64 erlaubt der Host für Matrixprodukte TF32 bei FP32-Speicherung. Rundung und Reihenfolge der Additionen beeinflussen daher den numerischen Vergleich. [GPU-Rückweg](../../src/gpu.c#L402) · [CUDA-Kernel](../../src/gpu_kernels.cu#L40)

`nfgpu check` vergleicht CPU und GPU bei Breiten 12 und 64. Neben den Logits prüft er den akkumulierten Gradienten, die Norm vor dem Begrenzen und die Wirkung zweier AdamW-Schritte. Die CPU-Differenzenprobe stellt eine andere Frage. Sie misst, wie sich die **Vorwärtsrechnung** beim Verändern eines Gewichts verhält. Zwei übereinstimmende Rückwärtsimplementierungen könnten denselben Ableitungsfehler enthalten. [Paritätstest](../../src/gpu.c#L479) · [Berichtete Messwerte](../training-124m.md#numerical-checks)

## Aufgaben zur Selbstkontrolle

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

Wie gut diese Schritte die nächste Vorhersage und die Antwort auf eine ganze Frage verbessern, lässt sich an den gespeicherten Gewichten messen. [Weiter zum Labor](04-labor.md) · [Übersicht](README.md)
