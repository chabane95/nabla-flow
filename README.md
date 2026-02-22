# nabla-flow
# Whitebox-Grad 🧠

Eine pädagogische Implementierung von Backpropagation "from scratch".

## 1. Das Konzept: Computational Graph

Jeder Rechenschritt (z.B. `a * b`) erzeugt einen Knoten im Graphen.
Wir speichern für jeden Wert $v$:
1. `data`: Den numerischen Wert (Forward Pass).
2. `grad`: Die lokale Ableitung $\frac{\partial L}{\partial v}$ (Backward Pass).

## 2. Die Mathematik der Multiplikation

Sei $z = x \cdot y$.
Nach der Produktregel gilt für die partiellen Ableitungen:

$$\frac{\partial z}{\partial x} = y \quad \text{und} \quad \frac{\partial z}{\partial y} = x$$

Im Code (`engine.py`) sieht das so aus:

```python
def _backward():
    # Anwendung der Kettenregel:
    # global_grad * local_grad
    self.grad += out.grad * other.data  # dz/dx * dL/dz
    other.grad += out.grad * self.data  # dz/dy * dL/dz