# Standalone Flow Matching Code

# %% 导入模块
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_moons
from torch import Tensor, nn
from tqdm import tqdm


# %% 定义模型
class Flow(nn.Module):

    def __init__(self, dim: int=2, h: int=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim+1, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, dim),
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), dim=-1))

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        # 简单期间, 使用 Midpoint ODE Solver
        # d psi_t/dt = u_t(psi_t) -> (psi_1-psi_0)/(t_1-t_0) = u(psi_0.5, t_0.5)
        # psi_1 = psi_0 + (t_1-t_0) * u(psi_0.5, t_0.5)
        # psi_0.5 = psi_0 + (t_0.5-t_0) * u(psi_0, t_0)
        return x_t + (t_end-t_start) * self.forward(x_t+self.forward(x_t, t_start) * (t_end-t_start)/2, t_start+(t_end-t_start)/2)

# %% 训练模型
flow = Flow()
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

for _ in tqdm(range(10000)):
    x_1  = Tensor(make_moons(n_samples=256, noise=0.05)[0])
    x_0  = torch.randn_like(x_1)
    t    = torch.rand(len(x_1), 1)
    x_t  = (1-t) * x_0 + t * x_1
    dx_t = x_1 - x_0
    optimizer.zero_grad()
    loss_fn(flow(x_t, t), dx_t).backward()
    optimizer.step()

# %% 采样
x = torch.randn(300, 2)
n_steps = 8
time_steps = torch.linspace(0, 1.0, n_steps+1)

fig, axes = plt.subplots(1, n_steps+1, figsize=(30, 4), sharex=True, sharey=True)

axes[0].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
axes[0].set_title(f"t={time_steps[0]:.2f}")
axes[0].set_xlim(-3.0, 3.0)
axes[0].set_ylim(-3.0, 3.0)

for i in range(n_steps):
    x = flow.step(x, time_steps[i], time_steps[i+1])
    axes[i+1].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
    axes[i+1].set_title(f"t={time_steps[i+1]:.2f}")

plt.tight_layout()
plt.show()

# %%