# %%
# flow_matching/solver/ode_solver.py

from abc import ABC, abstractmethod
# %% 引入模块
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torchdiffeq import odeint


def gradient(
    output: Tensor,
    x: Tensor,
    grad_outputs: Optional[Tensor] = None,
    create_graph: bool = False) -> Tensor:
    if grad_outputs is None:
        grad_outputs = torch.ones_like(output).detach()
    grad = torch.autograd.grad(
        output, x, grad_outputs=grad_outputs, create_graph=create_graph
    )[0]
    return grad

# %%
class ModelWrapper(ABC, nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        return self.model(x=x, t=t, **extras)

class Solver(ABC, nn.Module):

    @abstractmethod
    def sample(self, x_0: Tensor=None) -> Tensor:
        ...

class ODESolver(Solver):

    def __init__(self, velocity_model: Union[ModelWrapper, Callable]):
        super().__init__()
        self.velocity_model = velocity_model

    def sample(
        self,
        x_init: Tensor, #初始条件 X_0~p
        step_size: Optional[float], #步长
        method: str = "euler", #torchdiffeq 库支持的方法
        atol: float = 1e-5, #绝对容差, 用于自适应步长求解器
        rtol: float = 1e-5, #相对容差, 用于自适应步长求解器
        time_grid: Tensor = torch.tensor([0.0, 1.0]), #时间区间
        return_intermediates: bool = False, #返回中间时间步
        enable_grad: bool = False, #采样时是否计算梯度
        **model_extras) -> Union[Tensor, Sequence[Tensor]]:

        time_grid = time_grid.to(x_init.device)

        def ode_func(t, x):
            return self.velocity_model(x=x, t=t, **model_extras)

        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            sol = odeint(
                ode_func,
                x_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        if return_intermediates:
            return sol
        else:
            return sol[-1]

    def compute_likelihood(
            self,
            x_1: Tensor,
            log_p0: Callable[[Tensor], Tensor],
            step_size: Optional[float],
            method: str = "euler",
            atol: float = 1e-5,
            rtol: float = 1e-5,
            time_grid: Tensor = torch.tensor([0.0, 1.0]),
            return_intermediates: bool = False,
            exact_divergence: bool = False,
            enable_grad: bool = False,
            **model_extras) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]:
        assert (
            time_grid[0] == 1.0 and time_grid[-1] == 0.0
        ), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"

        # Fix the random projection for the Hutchinson divergence estimator
        if not exact_divergence:
            z = (torch.randn_like(x_1).to(x_1.device) < 0) * 2.0 - 1.0

        def ode_func(x, t):
            return self.velocity_model(x=x, t=t, **model_extras)

        def dynamics_func(t, states):
            xt = states[0]
            with torch.set_grad_enabled(True):
                xt.requires_grad_()
                ut = ode_func(xt, t)

                if exact_divergence:
                    # Compute exact divergence
                    div = 0
                    for i in range(ut.flatten(1).shape[1]):
                        div += gradient(ut[:, i], xt, create_graph=True)[:, i]
                else:
                    # Compute Hutchinson divergence estimator E[z^T D_x(ut) z]
                    ut_dot_z = torch.einsum(
                        "ij,ij->i", ut.flatten(start_dim=1), z.flatten(start_dim=1)
                    )
                    grad_ut_dot_z = gradient(ut_dot_z, xt)
                    div = torch.einsum(
                        "ij,ij->i",
                        grad_ut_dot_z.flatten(start_dim=1),
                        z.flatten(start_dim=1),
                    )

            return ut.detach(), div.detach()

        y_init = (x_1, torch.zeros(x_1.shape[0], device=x_1.device))
        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            sol, log_det = odeint(
                dynamics_func,
                y_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        x_source = sol[-1]
        source_log_p = log_p0(x_source)

        if return_intermediates:
            return sol, source_log_p + log_det[-1]
        else:
            return sol[-1], source_log_p + log_det[-1]

# %%
class Flow(nn.Module):

    def __init__(self, dim=2, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim+1, h),
            nn.ReLU(),
            nn.Linear(h, dim))

    def forward(self, x, t):
        t = t.view(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([t, x], dim=-1))

# %%
velocity_model = Flow()
x_0 = torch.randn(300, 2)

solver = ODESolver(velocity_model)
num_steps = 100

x_1 = solver.sample(
    x_init=x_0,
    method="midpoint",
    step_size=1.0 / num_steps,
)
print(f"{x_1=}")
# %%