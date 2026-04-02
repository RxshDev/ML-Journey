import torch

# Simple linear regression with PyTorch
x = torch.tensor(1.0)

# Enable gradient tracking for parameters
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Compute prediction
y_pred = w * x + b

# True target value
y_true = torch.tensor(2.0)

# Compute loss (Mean Squared Error)
loss = (y_pred - y_true) ** 2

# Print prediction and loss
print(f"Vorhersage: {y_pred.item()}")
print(f"Verlust: {loss.item()}")

# Compute gradients
loss.backward()

# Print gradients
print(f"Gradient von w: {w.grad.item()}")
print(f"Gradient von b: {b.grad.item()}")

# Update parameters with a simple gradient descent step
learning_rate = 0.1

# Update parameters without tracking gradients
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad


print(f"Neues w: {w.item()}")
print(f"Neues b: {b.item()}")

# new prediction and loss after parameter update
y_pred_new = x * w + b
loss_new = (y_pred_new - y_true) ** 2

print(f"Neue Vorhersage: {y_pred_new.item()}")
print(f"Neuer Verlust: {loss_new.item()}")
