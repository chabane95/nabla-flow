# nabla-flow
Kompakte, mathematisch saubere Implementierung von Backpropagation (Reverse-Mode Automatic Differentiation) auf einem Computational Graph.

## Vorwissen (minimal, aber notwendig)
- Grenzwerte und Stetigkeit.
- Ableitung einer Funktion $f:\mathbb{R}\to\mathbb{R}$ aus der Definition.
- Partielle Ableitungen und Kettenregel für mehrere Variablen.
- Grundidee von Gradientenabstieg.

## 1. Was bei ML-Ableitungen wirklich passiert
Die mathematische Ableitung ist immer dieselbe Größe. Unterschiedlich ist nur die Rechenmethode:
- Numerische Differentiation: Approximation, z.B. $\frac{f(x+h)-f(x)}{h}$.
- Symbolische Differentiation: algebraische Umformung zu geschlossener Formel.
- Automatic Differentiation (AD): exakte Anwendung der Kettenregel auf elementare Operationen (bis auf Floating-Point-Rundung).

Backpropagation ist Reverse-Mode AD für eine skalare Loss-Funktion $L$.

## 2. First Principles
Für $f:\mathbb{R}\to\mathbb{R}$:
$$
f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}.
$$

Für $f:\mathbb{R}^n\to\mathbb{R}$:
$$
\frac{\partial f}{\partial x_i}(x)=\lim_{h\to 0}\frac{f(x_1,\dots,x_i+h,\dots,x_n)-f(x)}{h}.
$$

Der Gradient ist
$$
\nabla f(x)=\left(\frac{\partial f}{\partial x_1},\dots,\frac{\partial f}{\partial x_n}\right).
$$

## 3. Computational Graph und Backprop-Gleichung
Jeder Zwischenwert ist ein Knoten $v$ mit:
- `data`: numerischer Vorwärtswert.
- `grad`: adjungierte Größe $\bar v:=\frac{\partial L}{\partial v}$.

Wenn ein Knoten $u$ mehrere direkte Nachfolger $w_1,\dots,w_m$ hat, gilt aus der Kettenregel:
$$
\bar u=\frac{\partial L}{\partial u}
=\sum_{k=1}^{m}\frac{\partial L}{\partial w_k}\frac{\partial w_k}{\partial u}
=\sum_{k=1}^{m}\bar w_k\frac{\partial w_k}{\partial u}.
$$
Das ist die zentrale Reverse-Accumulation-Regel.

## 4. Lokale Herleitungen elementarer Operationen
### Addition
Sei $z=x+y$.
$$
\frac{\partial z}{\partial x}
=\lim_{h\to 0}\frac{(x+h)+y-(x+y)}{h}=1,\quad
\frac{\partial z}{\partial y}=1.
$$
Backprop:
$$
\bar x{+}= \bar z,\quad \bar y{+}= \bar z.
$$

### Multiplikation
Sei $z=xy$.
$$
\frac{\partial z}{\partial x}
=\lim_{h\to 0}\frac{(x+h)y-xy}{h}=y,\quad
\frac{\partial z}{\partial y}=x.
$$
Backprop:
$$
\bar x{+}= \bar z\,y,\quad \bar y{+}= \bar z\,x.
$$

### Potenz (ganzzahliger Exponent $n\in\mathbb{N}$)
Sei $z=x^n$. Mit Binomialsatz:
$$
\frac{(x+h)^n-x^n}{h}
=\frac{\sum_{k=0}^{n}\binom{n}{k}x^{n-k}h^k-x^n}{h}
=n x^{n-1}+\sum_{k=2}^{n}\binom{n}{k}x^{n-k}h^{k-1}.
$$
Für $h\to 0$ verschwinden alle Terme ab $k\ge 2$, also:
$$
\frac{\partial z}{\partial x}=n x^{n-1},\qquad
\bar x{+}= \bar z\,n x^{n-1}.
$$

## 5. Beispiel: vollständige Kettenregel
Sei
$$
L=(ab+c)d.
$$
Setze $u=ab$, $v=u+c$, $L=vd$.

Vorwärts:
$$
u=ab,\quad v=u+c,\quad L=vd.
$$

Rückwärts (Start $\bar L=1$):
$$
\bar v=\bar L\frac{\partial L}{\partial v}=d,\quad
\bar d=\bar L\frac{\partial L}{\partial d}=v,
$$
$$
\bar u=\bar v\frac{\partial v}{\partial u}=d,\quad
\bar c=\bar v\frac{\partial v}{\partial c}=d,
$$
$$
\bar a=\bar u\frac{\partial u}{\partial a}=bd,\quad
\bar b=\bar u\frac{\partial u}{\partial b}=ad.
$$
Damit:
$$
\frac{\partial L}{\partial a}=bd,\quad
\frac{\partial L}{\partial b}=ad,\quad
\frac{\partial L}{\partial c}=d,\quad
\frac{\partial L}{\partial d}=ab+c.
$$

## 6. Numerischer Gradient-Check (nur zur Validierung)
Zentrale Differenz:
$$
\frac{\partial L}{\partial x_i}\approx
\frac{L(x_i+\varepsilon)-L(x_i-\varepsilon)}{2\varepsilon}
\quad (\varepsilon\ll 1).
$$
Das ist ein Testwerkzeug, nicht der eigentliche Trainingsalgorithmus.

## 7. Referenzen (Standard-Lehrbücher)
- Michael Spivak, *Calculus*.
- Walter Rudin, *Principles of Mathematical Analysis*.
- Christopher M. Bishop, *Pattern Recognition and Machine Learning*.
- Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning*.
- Jorge Nocedal, Stephen J. Wright, *Numerical Optimization*.
