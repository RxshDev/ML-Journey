import torch

# Einfache lineare Regression mit PyTorch
x = torch.tensor(1.0)

# Parameter mit Gradientenberechnung aktivieren
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Vorhersage berechnen
y_pred = w * x + b

# Wahre Zielwerte
y_true = torch.tensor(2.0)

# Berechnung des Verlusts (Mean Squared Error)
loss = (y_pred - y_true) ** 2

# Ausgabe der Vorhersage und des Verlusts
print(f"Vorhersage: {y_pred.item()}")
print(f"Verlust: {loss.item()}")

# Berechnung der Gradienten
loss.backward()

# Gradienten ausgeben
print(f"Gradient von w: {w.grad.item()}")
print(f"Gradient von b: {b.grad.item()}")

# Aktualisierung der Parameter mit einem einfachen Gradient Descent Schritt
learning_rate = 0.1

# Aktualisierung der Parameter ohne die Gradienten zu verfolgen
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad


print(f"Neues w: {w.item()}")
print(f"Neues b: {b.item()}")


y_pred_new = x * w + b
loss_new = (y_pred_new - y_true) ** 2
print(f"Neue Vorhersage: {y_pred_new.item()}")
print(f"Neuer Verlust: {loss_new.item()}")

