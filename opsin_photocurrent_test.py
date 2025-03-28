import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -----------------------------
# 1) Define the single-cluster opsin model (with full A_o)
# -----------------------------
class SingleClusterOpsinModel(nn.Module):
    def __init__(self, No):
        super().__init__()
        # Learnable parameters: A_o, B_o, C_o, beta
        self.A_o = nn.Parameter(torch.randn(No, No) * 0.1)  # Full matrix
        self.B_o = nn.Parameter(torch.randn(No, 1) * 0.1)
        self.C_o = nn.Parameter(torch.randn(1, No) * 0.1)
        self.beta = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, u):
        """
        u: Tensor of shape [T] representing the scalar light input over T time steps.
        Returns the photocurrent y of shape [T].
        """
        T = u.shape[0]
        x = torch.zeros((self.A_o.size(0),), dtype=u.dtype, device=u.device)
        y = torch.zeros((T,), dtype=u.dtype, device=u.device)

        for t in range(T):
            x = self.A_o @ x + self.B_o.view(-1) * (self.beta * u[t])
            y[t] = self.C_o @ x
        return y


torch.manual_seed(42)
No = 2

true_A_o = torch.tensor([[0.8, 0.0],
                         [0.0, 0.95]])
true_B_o = torch.tensor([[0.5],
                         [0.2]])
true_C_o = torch.tensor([[1.0, -0.5]])
beta_true = 1.0

def simulate_photocurrent(A_o, B_o, C_o, beta, u):
    T = u.shape[0]
    x = torch.zeros((A_o.shape[0],))
    y = torch.zeros((T,))
    for t in range(T):
        x = A_o @ x + B_o.view(-1) * (beta * u[t])
        y[t] = C_o @ x
    return y

# Create a pulse input: light is on for t < 10, off for t >= 10
T = 50
u = torch.zeros(T)
u[:10] = 1.0

with torch.no_grad():
    y_target = simulate_photocurrent(true_A_o, true_B_o, true_C_o, beta_true, u)

# -----------------------------
# 2) Instantiate our model, optimizer, and loss function
# -----------------------------
model = SingleClusterOpsinModel(No)
optimizer = optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

# -----------------------------
# 3) Prepare for interactive plotting
# -----------------------------
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Setup loss plot
losses = []
loss_line, = ax1.plot([], [], label="Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss")
ax1.legend()
ax1.grid(True)

# Setup photocurrent plot: target and current prediction
time = range(T)
target_line, = ax2.plot(time, y_target.numpy(), label="True Photocurrent", marker='o')
pred_line, = ax2.plot(time, torch.zeros(T).numpy(), label="Predicted Photocurrent", marker='x', linestyle='--')
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Photocurrent")
ax2.set_title("Photocurrent Fitting")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# 4) Training loop with visualization
# -----------------------------
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(u)
    loss = loss_fn(y_pred, y_target)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    # Update plots every 10 epochs
    if epoch % 10 == 0:
        loss_line.set_data(range(len(losses)), losses)
        ax1.relim()
        ax1.autoscale_view()

        pred_line.set_ydata(y_pred.detach().numpy())
        ax2.relim()
        ax2.autoscale_view()

        plt.pause(0.01)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

plt.ioff()
plt.show()

# -----------------------------
# 6) Print learned parameters for the best model
# -----------------------------
print("Learned parameters for the best model:")
print("A_o:\n", best_model.A_o.data)
print("B_o:\n", best_model.B_o.data)
print("C_o:\n", best_model.C_o.data)
print("beta:\n", best_model.beta.data)

# -----------------------------
# 7) Compare model output and target with metrics
# -----------------------------
mse = nn.MSELoss()(y_pred, y_target).item()
mae = nn.L1Loss()(y_pred, y_target).item()
r2 = 1 - torch.sum((y_target - y_pred)**2) / torch.sum((y_target - y_target.mean())**2)

print("\nPerformance Metrics:")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R-squared: {r2:.6f}")
