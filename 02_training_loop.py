import torch

def training_loop(n):
    
    print("This is the training loop.")
    
    x = torch.tensor(1.0)
    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    y_true = torch.tensor(2.0)
    
    for i in range(1, n):
        
        y_pred = x * w + b
        
        loss = (y_pred - y_true) ** 2
        
        loss.backward()
        
        learning_rate = 0.1
        
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
        
        w.grad.zero_()
        b.grad.zero_()
        
        if i % 10 == 0:
            print(f"\nIteration: {i}")
            print(f"Vorhersage: {y_pred.item()}")
            print(f"Verlust: {loss.item()}")
        
            
training_loop(101)

    
      