import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
from matplotlib.lines import Line2D 

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "mathtext.fontset": "cm",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.55,
        "lines.linewidth": 1.6,
    }
)

@dataclass(frozen=True)
class RheologyParameters:
    """Dimensionless Carreau-Yasuda (generalized Newtonian) parameters.

    The apparent viscosity is modelled as

        mu(gamma) = mu_inf + (mu_0 - mu_inf) * [1 + (lambda*gamma)^a]^((n-1)/a),

    where gamma is the scalar shear-rate invariant. For n < 1 the fluid is
    shear-thinning: mu -> mu_0 as gamma -> 0 and mu -> mu_inf as
    gamma -> infinity. The viscosity is scaled with mu_0, hence mu_0 = 1.
    """

    mu_0: float = 1.0                     
    mu_inf: float = 1.0e-2                 
    relaxation_time: float = 0.5           
    power_law_index: float = 0.5          
    yasuda_exponent: float = 2.0           
    gamma_regularization: float = 1.0e-6  

    def __post_init__(self) -> None:
        if not 0.0 < self.mu_inf < self.mu_0:
            raise ValueError("RheologyParameters requires 0 < mu_inf < mu_0.")
        if not 0.0 < self.power_law_index < 1.0:
            raise ValueError("This benchmark considers shear-thinning fluids: require 0 < n < 1.")
        if self.relaxation_time <= 0.0 or self.yasuda_exponent <= 0.0:
            raise ValueError("RheologyParameters requires lambda > 0 and a > 0.")
        if self.gamma_regularization <= 0.0:
            raise ValueError("The shear-rate regularisation epsilon must be strictly positive.")


@dataclass(frozen=True)
class FlowParameters:
    """Dimensionless flow parameters of the plane channel.

    NOTE: the Brinkman number is intentionally ABSENT. Under the
    theta = T / Br formulation the training problem is Br-invariant; Br
    enters only as a post hoc reporting scale
    (see StudyParameters.brinkman_reporting_values).
    """

    half_channel_height: float = 1.0   
    pressure_gradient: float = -1.0    
    thermal_conductivity: float = 0.1  

    def __post_init__(self) -> None:
        if self.half_channel_height <= 0.0:
            raise ValueError("The half-channel height H must be positive.")
        if self.thermal_conductivity <= 0.0:
            raise ValueError("The thermal conductivity k must be positive.")


@dataclass(frozen=True)
class TrainingParameters:
    """Network architecture, collocation and optimisation settings.

    Wall conditions are enforced by the output transform (hard constraints),
    so the loss contains exactly the two interior residuals
    [momentum, energy] and no boundary collocation points are required.

    The configuration is deliberately FIXED across the entire (lambda, n)
    grid: one architecture, one budget, one collocation set.
    """

    float_dtype: str = "float64"
    layer_sizes: Tuple[int, ...] = (1, 50, 50, 50, 2)
    activation: str = "tanh"           
    kernel_initializer: str = "Glorot normal"
    num_domain: int = 600             
    num_boundary: int = 0               
    num_test: int = 1000                
    train_distribution: str = "Hammersley"
    adam_iterations: int = 5000
    adam_learning_rate: float = 1.0e-3
    adam_display_every: int = 500      
    lbfgs_max_iterations: int = 5000
    lbfgs_restarts: int = 1             
    loss_weights: Tuple[float, ...] = (1.0, 1.0)   
    def __post_init__(self) -> None:
        if len(self.loss_weights) != 2:
            raise ValueError("v2.x uses exactly two loss terms [momentum, energy].")
        if self.adam_iterations <= 0 or self.adam_learning_rate <= 0.0:
            raise ValueError("Adam iterations and learning rate must be positive.")
        if self.lbfgs_restarts < 1:
            raise ValueError("lbfgs_restarts must be >= 1.")


@dataclass(frozen=True)
class StudyParameters:
    """Experimental design of the numerical study.

    Training grid : relaxation_times x power_law_indices (two-factor design),
                    replicated over ALL `seeds` (mean +/- std everywhere).
    Execution     : relaxation_times is listed strongest-FIRST so that the
                    stiffest corner (lambda_max, n_min) is run 1 - an early
                    warning. The order has no effect on results or figures.
    Reporting     : the Brinkman number is a reporting axis only - the
                    physical temperature is exported for every value of
                    `brinkman_reporting_values` via the exact scaling
                    T(Br) = Br * theta.
    """

    relaxation_times: Tuple[float, ...] = (5.0, 2.0, 0.5)   
    power_law_indices: Tuple[float, ...] = (0.3, 0.5, 0.9)
    seeds: Tuple[int, ...] = (42, 0, 1, 2, 3)
    brinkman_reporting_values: Tuple[float, ...] = (0.1, 1.0)
    quadrature_points: int = 20_001     
    validation_points: int = 401     
    output_directory: str = "output_results"

    def __post_init__(self) -> None:
        if not self.relaxation_times or not self.power_law_indices:
            raise ValueError("The training grid must be non-empty.")
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        if not self.brinkman_reporting_values:
            raise ValueError("At least one Brinkman reporting value is required.")
        if self.quadrature_points < 1000 or self.validation_points < 101:
            raise ValueError("Quadrature/validation resolutions are too coarse.")


@dataclass(frozen=True)
class CaseConfig:
    """A single rheological configuration of the training grid."""

    tag: str
    rheology: RheologyParameters
    flow: FlowParameters


DEFAULT_RHEO = RheologyParameters()
DEFAULT_FLOW = FlowParameters()
TRAIN = TrainingParameters()
STUDY = StudyParameters()
OUTPUT_DIR = Path(STUDY.output_directory)


LOSS_COMPONENT_NAMES: Tuple[str, ...] = (
    "loss_pde_momentum",
    "loss_pde_energy",
)


VALIDATION_METRIC_KEYS: Tuple[str, ...] = ("rmse", "mae", "relative_l2", "max_abs_error", "r2")


def collect_version_metadata() -> Dict[str, str]:
    """Return the software stack versions for the reproducibility record."""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pytorch": torch.__version__,
        "deepxde": dde.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }


def accelerator_summary() -> Dict[str, object]:
    """Return accelerator metadata (availability and device name)."""
    available = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if available else None
    return {"cuda_available": available, "cuda_device_name": name}


def enforce_reproducibility(seed: int, float_dtype: str) -> None:
    """Seed every stochastic source (NumPy, PyTorch, DeepXDE) and fix the
    DeepXDE floating-point precision *before* any network or dataset is built.

    For bit-exact cross-platform reproduction one may additionally pin
    ``torch.set_num_threads(1)`` (thread-count-dependent reduction order is
    the only remaining nondeterminism on CPU).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dde.config.set_random_seed(seed)
    dde.config.set_default_float(float_dtype)


def print_configuration_banner() -> None:
    """Display the full configuration metadata (versions, devices, design)."""
    versions = collect_version_metadata()
    accel = accelerator_summary()
    rule = "=" * 79
    print(rule)
    print(" COUPLED NON-NEWTONIAN FLOW / THERMAL PINN STUDY - v2.2 (DeepXDE + PyTorch)")
    print(" theta-formulation | hard wall constraints | 2-term loss | (lambda, n) grid")
    print(rule)
    print(f" Python             : {versions['python']}")
    print(f" NumPy              : {versions['numpy']}")
    print(f" PyTorch            : {versions['pytorch']}"
          + (f" (CUDA {torch.version.cuda})" if bool(accel["cuda_available"]) else ""))
    print(f" DeepXDE            : {versions['deepxde']}")
    print(f" pandas             : {versions['pandas']}")
    print(f" matplotlib         : {versions['matplotlib']}")
    if bool(accel["cuda_available"]):
        print(f" Accelerator        : CUDA available - {accel['cuda_device_name']}")
    else:
        print(" Accelerator        : CPU only")
    print(" Execution device   : PyTorch default device (full-batch, deterministic)")
    print(f" Floating-point     : {TRAIN.float_dtype}")
    print(f" Rheology (CY)      : mu_0={DEFAULT_RHEO.mu_0:g}, mu_inf={DEFAULT_RHEO.mu_inf:g}, "
          f"a={DEFAULT_RHEO.yasuda_exponent:g}, "
          f"lambda in {list(STUDY.relaxation_times)} (execution order), "
          f"n in {list(STUDY.power_law_indices)}")
    print(f" Flow               : dp/dx={DEFAULT_FLOW.pressure_gradient:g}, "
          f"H={DEFAULT_FLOW.half_channel_height:g}, k={DEFAULT_FLOW.thermal_conductivity:g} "
          f"(Br = reporting axis only: {list(STUDY.brinkman_reporting_values)})")
    print(f" Network            : {list(TRAIN.layer_sizes)} | {TRAIN.activation} | "
          f"{TRAIN.kernel_initializer} | output transform: (1 - (y/H)^2) hard walls")
    print(f" Collocation        : {TRAIN.num_domain} domain / {TRAIN.num_test} test points "
          f"({TRAIN.train_distribution}; walls enforced by construction)")
    print(f" Optimisation       : Adam ({TRAIN.adam_iterations} it, lr={TRAIN.adam_learning_rate:g}) "
          f"-> L-BFGS ({TRAIN.lbfgs_restarts} pass)")
    print(f" Seeds              : {list(STUDY.seeds)} (applied to every case)")
    print(rule)

def carreau_yasuda_viscosity(gamma_dot: np.ndarray, rheo: RheologyParameters) -> np.ndarray:
    """Carreau-Yasuda apparent viscosity, NumPy implementation (benchmark path)."""
    scaled = (rheo.relaxation_time * np.asarray(gamma_dot, dtype=np.float64)) ** rheo.yasuda_exponent
    exponent = (rheo.power_law_index - 1.0) / rheo.yasuda_exponent
    return rheo.mu_inf + (rheo.mu_0 - rheo.mu_inf) * (1.0 + scaled) ** exponent


def carreau_yasuda_viscosity_torch(gamma_dot: torch.Tensor, rheo: RheologyParameters) -> torch.Tensor:
    """Differentiable Carreau-Yasuda apparent viscosity (PyTorch / AD path).

    Implements exactly the same law as :func:`carreau_yasuda_viscosity`, in a
    form that is fully differentiable within the PINN computational graph.
    The regularised shear rate (gamma >= epsilon > 0) guarantees that both
    `torch.pow` calls remain in a smooth, strictly positive regime, so no
    numerical singularity can arise at low shear rates.
    """
    scaled = torch.pow(rheo.relaxation_time * gamma_dot, rheo.yasuda_exponent)
    exponent = (rheo.power_law_index - 1.0) / rheo.yasuda_exponent
    return rheo.mu_inf + (rheo.mu_0 - rheo.mu_inf) * torch.pow(1.0 + scaled, exponent)


def invert_shear_rate_from_stress(
    tau_mag: np.ndarray,
    rheo: RheologyParameters,
    max_iter: int = 500,
    tol: float = 1.0e-15,
) -> np.ndarray:
    """Invert the implicit constitutive relation gamma = |tau| / mu(gamma).

    The Picard map f(gamma) = |tau| / mu(gamma) is a contraction for
    shear-thinning Carreau-Yasda fluids (|f'| <= 1 - n < 1 asymptotically;
    1 - n = 0.7 at the strongest index n = 0.3 of the grid), hence the
    iteration converges geometrically to machine precision. The achieved
    closure residual is asserted to guarantee benchmark integrity for every
    parameter set of the study.
    """
    tau = np.asarray(tau_mag, dtype=np.float64)
    gamma = tau / rheo.mu_0  
    for _ in range(max_iter):
        gamma_updated = tau / carreau_yasuda_viscosity(gamma, rheo)
        if np.max(np.abs(gamma_updated - gamma), initial=0.0) < tol:
            gamma = gamma_updated
            break
        gamma = gamma_updated
    closure = np.max(
        np.abs(carreau_yasuda_viscosity(gamma, rheo) * gamma - tau), initial=0.0
    )
    if closure > 1.0e-8:
        raise RuntimeError(
            "Rheological Picard inversion did not converge "
            f"(closure residual {closure:.3e})."
        )
    return gamma

@dataclass(frozen=True)
class ReferenceSolution:
    """Semi-analytical benchmark fields evaluated on a query grid.

    `theta` is the Br-invariant thermal field (theta = T/Br); the physical
    temperature at a Brinkman number `br` is recovered exactly as `br*theta`.
    """

    y: np.ndarray         
    u: np.ndarray         
    theta: np.ndarray     
    gamma_dot: np.ndarray  
    mu_app: np.ndarray     


def _cumulative_trapezoid(f: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of f over x, prefixed with 0 at x[0].

    Implemented explicitly (instead of relying on SciPy or the deprecated
    `np.trapz` cumulative companions) to keep the dependency footprint minimal
    and version-stable.
    """
    dx = np.diff(x)
    increments = 0.5 * (f[1:] + f[:-1]) * dx
    return np.concatenate(([0.0], np.cumsum(increments)))


def compute_reference_solution(
    y_query: np.ndarray,
    rheo: RheologyParameters,
    flow: FlowParameters,
    quadrature_points: int,
    verbose: bool = False,
) -> ReferenceSolution:
    """Semi-analytical benchmark solution of the coupled reduced problem.

    Exploiting the mirror symmetry of the solution about the centreline
    y = 0, all fields are constructed on the radial grid s = |y| in [0, H]
    and then interpolated onto the (arbitrary) query grid.

    Pipeline
    --------
    1. Exact first integral of momentum:  tau_xy(s) = (dp/dx) * s,
       hence |tau_xy| = G * s with G = |dp/dx|.
    2. Picard inversion of the Carreau-Yasuda law: gamma = G*s / mu(gamma).
    3. Velocity quadrature:   u(s) = Integral_s^H gamma(xi) d(xi).
    4. Thermal quadrature (theta-form):  theta(s) = (1/k) Integral_s^H
       Integral_0^z mu(t) gamma(t)^2 dt dz,  with T(s; Br) = Br * theta(s).

    The construction is finally self-verified by finite differences using
    explicit centred stencils restricted to interior nodes.
    """
    H = flow.half_channel_height
    G = abs(flow.pressure_gradient) 
    inv_k = 1.0 / flow.thermal_conductivity


    s = np.linspace(0.0, H, quadrature_points)


    tau_mag = G * s
    gamma = invert_shear_rate_from_stress(tau_mag, rheo)
    mu = carreau_yasuda_viscosity(gamma, rheo)


    cumulative_gamma = _cumulative_trapezoid(gamma, s)  
    u_of_s = cumulative_gamma[-1] - cumulative_gamma    


    dissipation = mu * gamma**2                       
    inner = _cumulative_trapezoid(dissipation, s)      
    outer = _cumulative_trapezoid(inner, s)            
    theta_of_s = inv_k * (outer[-1] - outer)          

    h_step = float(s[1] - s[0])


    fd_du = (u_of_s[2:] - u_of_s[:-2]) / (2.0 * h_step)             
    velocity_check = float(np.max(np.abs(fd_du + gamma[1:-1])))

    fd_d2theta = (theta_of_s[2:] - 2.0 * theta_of_s[1:-1] + theta_of_s[:-2]) / h_step**2
    energy_check = float(np.max(np.abs(fd_d2theta + inv_k * dissipation[1:-1])))

    if max(velocity_check, energy_check) > 1.0e-5:
        raise RuntimeError(
            "Semi-analytical benchmark failed its finite-difference "
            f"self-verification (velocity: {velocity_check:.2e}, "
            f"energy: {energy_check:.2e})."
        )
    if verbose:
        print(f"[Benchmark self-check] max|d/ds u + gamma|            = {velocity_check:.3e}")
        print(f"[Benchmark self-check] max|d2/ds2 theta + Phi/k|      = {energy_check:.3e}")


    y_abs = np.abs(np.asarray(y_query, dtype=np.float64))
    return ReferenceSolution(
        y=np.asarray(y_query, dtype=np.float64),
        u=np.interp(y_abs, s, u_of_s),
        theta=np.interp(y_abs, s, theta_of_s),
        gamma_dot=np.interp(y_abs, s, gamma),
        mu_app=np.interp(y_abs, s, mu),
    )

def make_pde_residuals(rheo: RheologyParameters, flow: FlowParameters):
    """Factory returning the case-parameterised PDE residual function.

    The network output is y = (u, theta) AFTER the hard-wall output transform
    (DeepXDE applies the transform inside the forward pass, so this function
    and every prediction see the physical u and theta). Both residuals are
    assembled entirely from automatic differentiation:

      * dde.grad.jacobian  -> du/dy and the transversal viscosity gradient,
      * dde.grad.hessian   -> d2u/dy2 and d2theta/dy2,

    so no analytical differentiation of the (non-linear) rheological law is
    ever required. The energy residual is the physical equation divided by
    Br (theta-form): all terms are O(1) at every Brinkman number, which
    removes the loss-gradient imbalance diagnosed in v2.0.
    """

    def pde_residuals(x: torch.Tensor, y: torch.Tensor) -> List[torch.Tensor]:

        du_dy = dde.grad.jacobian(y, x, i=0, j=0)                    
        d2u_dy2 = dde.grad.hessian(y, x, component=0, i=0, j=0)      
        d2theta_dy2 = dde.grad.hessian(y, x, component=1, i=0, j=0) 


        gamma_dot = torch.sqrt(du_dy**2 + rheo.gamma_regularization**2)


        mu_app = carreau_yasuda_viscosity_torch(gamma_dot, rheo)


        dmu_dy = dde.grad.jacobian(mu_app, x, i=0, j=0)


        residual_momentum = mu_app * d2u_dy2 + dmu_dy * du_dy - flow.pressure_gradient


        residual_energy = (
            flow.thermal_conductivity * d2theta_dy2 + mu_app * du_dy**2
        )

        return [residual_momentum, residual_energy]

    return pde_residuals


def make_hard_wall_transform(half_channel_height: float):
    """Factory returning the multiplicative hard-constraint output transform.

        u(y)     = (1 - (y/H)^2) * N1(y),
        theta(y) = (1 - (y/H)^2) * N2(y),

    so u(+/-H) = theta(+/-H) = 0 hold to machine precision by construction
    and no boundary loss terms are required. Because u and theta have simple
    zeros at the walls (u'(H) = -gamma(H) != 0), the raw network outputs
    N1, N2 remain finite and smooth there - the parameterisation is
    well-conditioned (on the stiff corner the raw amplitudes reach ~10 for
    N1 and ~25 for N2: large but perfectly representable).
    """
    H = half_channel_height

    def transform(inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        wall = 1.0 - torch.square(inputs / H)   
        return torch.cat((outputs[:, 0:1] * wall, outputs[:, 1:2] * wall), dim=1)

    return transform


def make_rheology_operator(rheo: RheologyParameters):
    """Factory returning a post-processing operator [du/dy, gamma, mu(gamma)].

    Used with ``model.predict(..., operator=...)`` to extract the network's
    rheological state for independent validation of the apparent-viscosity
    field. It mirrors exactly the regularisation adopted in the PDE residual
    and only involves the velocity component (Br-invariant).
    """

    def rheology_operator(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        du_dy = dde.grad.jacobian(y, x, i=0, j=0)
        gamma_dot = torch.sqrt(du_dy**2 + rheo.gamma_regularization**2)
        mu_app = carreau_yasuda_viscosity_torch(gamma_dot, rheo)
        return torch.cat((du_dy, gamma_dot, mu_app), dim=1)

    return rheology_operator


def build_training_data(case: CaseConfig, train: TrainingParameters) -> dde.data.PDE:
    """Assemble the collocation dataset (geometry + residuals; no soft BCs)."""
    H = case.flow.half_channel_height
    geometry = dde.geometry.Interval(-H, H)

    return dde.data.PDE(
        geometry,
        make_pde_residuals(case.rheology, case.flow),
        [],
        num_domain=train.num_domain,
        num_boundary=train.num_boundary,
        train_distribution=train.train_distribution,
        num_test=train.num_test,  
    )


def build_model(case: CaseConfig, train: TrainingParameters, prefix: str = "") -> dde.Model:
    """Construct the PINN: FNN + hard-wall output transform."""
    data = build_training_data(case, train)
    network = dde.nn.FNN(list(train.layer_sizes), train.activation, train.kernel_initializer)
    network.apply_output_transform(make_hard_wall_transform(case.flow.half_channel_height))
    model = dde.Model(data, network)
    n_parameters = sum(p.numel() for p in network.parameters())
    print(f"{prefix}[Model] FNN {list(train.layer_sizes)} | {train.activation} | "
          f"{train.kernel_initializer} | {n_parameters} trainable parameters | "
          f"hard walls (1 - (y/H)^2) | collocation {train.num_domain} domain / "
          f"{train.num_test} test")
    return model

def _merge_lbfgs_histories(histories: List[Any]) -> Any:
    """Concatenate L-BFGS restart passes with renumbered steps (monotone)."""
    steps: List[int] = []
    losses: List[List[float]] = []
    offset = 0
    for history in histories:
        steps.extend(int(s) + offset for s in history.steps)
        losses.extend([float(v) for v in row] for row in history.loss_train)
        if steps:
            offset = steps[-1] + 1
    return SimpleNamespace(steps=steps, loss_train=losses)


def train_model(
    model: dde.Model,
    train: TrainingParameters,
    prefix: str = "",
    fine_history: bool = True,
) -> Tuple[Any, Any, Dict[str, float]]:
    """Two-phase hybrid optimisation: Adam (exploration) then L-BFGS (refinement).

    Phase I performs robust first-order exploration. Phase II applies a
    curvature-aware L-BFGS optimiser with strong-Wolfe line search; restarts
    are supported but disabled by default (v2.0 restarts gained < 1 %, and a
    restart is abandoned as soon as the final loss of a pass fails to improve
    by at least 5 % - line-search exhaustion / stagnation criterion).

    NOTE (DeepXDE 1.15): the `iterations` argument is ignored for L-BFGS;
    each pass runs the internal default budget (~15000 iterations) and
    self-terminates on line-search failure.

    ``fine_history`` selects the loss-history granularity: fine for primary
    (detailed) runs, coarse for statistical replicates.
    """
    weights = list(train.loss_weights)
    timings: Dict[str, float] = {}

    adam_display = train.adam_display_every if fine_history else max(1, train.adam_iterations)
    lbfgs_display = 500 if fine_history else 2000


    print(f"{prefix} Phase I  - Adam ({train.adam_iterations} iterations, "
          f"lr={train.adam_learning_rate:g})")
    model.compile("adam", lr=train.adam_learning_rate, loss_weights=weights)
    t0 = time.perf_counter()
    history_adam, _ = model.train(
        iterations=train.adam_iterations, display_every=adam_display
    )
    timings["adam_s"] = time.perf_counter() - t0


    print(f"{prefix} Phase II - L-BFGS (up to {train.lbfgs_restarts} pass(es))")
    t0 = time.perf_counter()
    passes: List[Any] = []
    best_total = float("inf")
    for attempt in range(train.lbfgs_restarts):
        model.compile("L-BFGS", loss_weights=weights)
        history, _ = model.train(
            iterations=train.lbfgs_max_iterations, display_every=lbfgs_display
        )
        if len(history.steps) == 0:
            break
        passes.append(history)
        final_total = float(np.sum(np.asarray(history.loss_train[-1], dtype=np.float64)))
        print(f"{prefix}   L-BFGS pass {attempt + 1}: final total loss = {final_total:.3e}")
        if not (final_total < 0.95 * best_total):
            break  
        best_total = min(best_total, final_total)
    timings["lbfgs_s"] = time.perf_counter() - t0

    history_lbfgs = _merge_lbfgs_histories(passes)
    print(f"{prefix} [Training] Adam {timings['adam_s']:.1f} s | "
          f"L-BFGS {timings['lbfgs_s']:.1f} s ({len(passes)} pass(es))")
    return history_adam, history_lbfgs, timings


def _rebase_steps(history_df: pd.DataFrame) -> np.ndarray:
    """Re-base per-phase iteration counters onto a single monotone axis."""
    global_steps = np.zeros(len(history_df), dtype=np.float64)
    offset = 0.0
    for phase_name, phase_frame in history_df.groupby("phase", sort=False):
        local = phase_frame["step"].to_numpy(dtype=np.float64)
        rebased = offset + (local - local[0])
        mask = history_df["phase"].to_numpy() == phase_name
        global_steps[mask] = rebased
        offset = rebased[-1] + 1.0
    return global_steps


def export_loss_history(history_adam: Any, history_lbfgs: Any, run_dir: Path,
                        prefix: str = "") -> pd.DataFrame:
    """Consolidate the phase-wise DeepXDE loss histories into one tagged frame."""
    frames: List[pd.DataFrame] = []
    for phase_name, history in (("Adam", history_adam), ("L-BFGS", history_lbfgs)):
        steps = np.asarray(history.steps, dtype=np.int64)
        losses = np.asarray(history.loss_train, dtype=np.float64)
        if steps.size == 0:
            continue
        if losses.ndim != 2 or losses.shape[1] != len(LOSS_COMPONENT_NAMES):
            raise RuntimeError(
                f"Unexpected loss-history layout in phase {phase_name}: {losses.shape}"
            )
        frame = pd.DataFrame(losses, columns=list(LOSS_COMPONENT_NAMES))
        frame["loss_total"] = frame[list(LOSS_COMPONENT_NAMES)].sum(axis=1)
        frame.insert(0, "step", steps)
        frame.insert(0, "phase", phase_name)
        frames.append(frame)
    if not frames:
        raise RuntimeError("Both training phases returned empty loss histories.")

    history_df = pd.concat(frames, ignore_index=True)
    history_df["global_step"] = _rebase_steps(history_df)

    csv_path = run_dir / "training_loss_history.csv"
    history_df.to_csv(csv_path, index=False)
    print(f"{prefix}[Export] Loss history written to {csv_path}")
    return history_df


def plot_convergence_history(history_df: pd.DataFrame, save_path: Path,
                             prefix: str = "") -> None:
    """Publication-grade convergence history (2 PDE residuals, log scale)."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)
    x = history_df["global_step"].to_numpy(dtype=np.float64)

    curve_specs = (
        ("loss_pde_momentum", "Momentum residual (PDE)", "tab:blue", 1.4),
        ("loss_pde_energy", "Energy residual (PDE, theta-form)", "tab:orange", 1.4),
        ("loss_total", "Total training loss", "black", 2.0),
    )
    y_min = np.inf
    for column, label, color, lw in curve_specs:
        values = np.maximum(history_df[column].to_numpy(dtype=np.float64), 1.0e-30)
        ax.plot(x, values, color=color, lw=lw, label=label, alpha=0.9)
        y_min = min(y_min, float(values.min()))

    lbfgs_mask = history_df["phase"].to_numpy() == "L-BFGS"
    if lbfgs_mask.any():
        boundary = float(x[lbfgs_mask].min())
        ax.axvline(boundary, color="grey", ls="--", lw=1.0)
        ax.axvspan(boundary, float(x.max()), color="grey", alpha=0.08)
    ax.text(0.02, 0.94, "Phase I: Adam", transform=ax.transAxes,
            fontsize=9, style="italic")
    ax.text(0.98, 0.94, "Phase II: L-BFGS", transform=ax.transAxes,
            fontsize=9, style="italic", ha="right")
    ax.text(0.98, 0.87, "walls: hard-enforced (no BC loss)",
            transform=ax.transAxes, fontsize=8, style="italic", ha="right",
            color="dimgray")


    ax.axhline(1.0e-5, color="dimgray", ls=":", lw=1.0)
    ax.annotate(r"residual target $10^{-5}$",
                xy=(float(np.median(x)), 1.0e-5), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=8,
                color="dimgray")
    ax.set_ylim(bottom=min(y_min * 0.5, 1.0e-6))

    ax.set_yscale("log")
    ax.set_xlabel("Training step (phase-rebased)")
    ax.set_ylabel("Loss magnitude")
    ax.set_title("PINN convergence: Adam exploration and L-BFGS refinement")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"{prefix}[Figure] Convergence history written to {save_path}")

def compute_validation_metrics(prediction: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    """Quantitative agreement metrics between a PINN field and the benchmark.

    Returns RMSE, MAE, relative L2 norm, max absolute (L-infinity) error and
    the coefficient of determination R^2 (reported downstream as 1 - R^2 in
    scientific notation, which resolves the quality of near-perfect fits).
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    error = prediction - reference
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((reference - reference.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "relative_l2": float(np.linalg.norm(error) / (np.linalg.norm(reference) + 1.0e-30)),
        "max_abs_error": float(np.max(np.abs(error))),
        "r2": float(1.0 - ss_res / (ss_tot + 1.0e-30)),
    }


def print_metric_table(records: List[Dict[str, Any]], title: str = "") -> None:
    """Console rendering of the per-run validation metrics (1 - R^2 form)."""
    if title:
        print(title)
    rule = "-" * 92
    print(rule)
    print(f"{'Field':<12}{'Br':>7}{'RMSE':>13}{'MAE':>13}{'Rel. L2':>13}"
          f"{'Max |err|':>13}{'1 - R^2':>13}")
    print(rule)
    for m in records:
        br = m["brinkman_number"]
        br_str = f"{br:g}" if isinstance(br, float) else "-"
        print(f"{m['field']:<12}{br_str:>7}{m['rmse']:>13.3e}{m['mae']:>13.3e}"
              f"{m['relative_l2']:>13.3e}{m['max_abs_error']:>13.3e}"
              f"{1.0 - m['r2']:>13.3e}")
    print(rule)


def plot_validation_profiles(
    y: np.ndarray,
    u_pinn: np.ndarray,
    T_pinn: np.ndarray,
    u_ref: np.ndarray,
    T_ref: np.ndarray,
    metrics: Dict[str, Dict[str, float]],
    save_path: Path,
    prefix: str = "",
    br_label: str = "",
) -> None:
    """Four-panel validation figure: profiles, local error map and parity plot."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), dpi=300)
    (ax_u, ax_T), (ax_err, ax_parity) = axes


    ax_u.plot(y, u_ref, color="black", lw=2.2, label="Semi-analytical reference")
    ax_u.plot(y, u_pinn, color="crimson", ls="--", lw=1.8, marker="o",
              markersize=2.6, markevery=20,
              label=f"PINN (RMSE = {metrics['u(y)']['rmse']:.2e})")
    ax_u.set_title("(a) Axial velocity profile")
    ax_u.set_xlabel(r"$y/H$")
    ax_u.set_ylabel(r"$u(y)$")
    ax_u.legend(loc="lower center")


    ax_T.plot(y, T_ref, color="black", lw=2.2, label="Semi-analytical reference")
    ax_T.plot(y, T_pinn, color="navy", ls="--", lw=1.8, marker="s",
              markersize=2.6, markevery=20,
              label=f"PINN (RMSE = {metrics['T(y)']['rmse']:.2e})")
    ax_T.set_title(f"(b) Temperature profile (Br = {br_label})" if br_label
                   else "(b) Temperature profile")
    ax_T.set_xlabel(r"$y/H$")
    ax_T.set_ylabel(r"$T(y)$")
    ax_T.legend(loc="lower center")


    err_u = np.maximum(np.abs(u_pinn - u_ref), 1.0e-16)
    err_T = np.maximum(np.abs(T_pinn - T_ref), 1.0e-16)
    ax_err.semilogy(y, err_u, color="crimson", lw=1.6,
                    label=r"$|u_{\mathrm{PINN}} - u_{\mathrm{ref}}|$")
    ax_err.semilogy(y, err_T, color="navy", lw=1.6,
                    label=r"$|T_{\mathrm{PINN}} - T_{\mathrm{ref}}|$")
    ax_err.set_title("(c) Local absolute error distribution")
    ax_err.set_xlabel(r"$y/H$")
    ax_err.set_ylabel("Absolute error")
    ax_err.legend(loc="upper center")


    upper = 1.08 * max(float(np.max(u_ref)), float(np.max(T_ref)))
    ax_parity.plot([0.0, upper], [0.0, upper], color="black", ls="--", lw=1.0,
                   label="Identity")
    ax_parity.scatter(u_ref, u_pinn, s=14, color="crimson", marker="o",
                      label=rf"$u$: $1-R^2$ = {1.0 - metrics['u(y)']['r2']:.2e}")
    ax_parity.scatter(T_ref, T_pinn, s=14, color="navy", marker="s",
                      label=rf"$T$: $1-R^2$ = {1.0 - metrics['T(y)']['r2']:.2e}")
    ax_parity.set_xlim(0.0, upper)
    ax_parity.set_ylim(0.0, upper)
    ax_parity.set_aspect("equal", adjustable="box")
    ax_parity.set_title("(d) Parity: PINN vs semi-analytical")
    ax_parity.set_xlabel("Reference value")
    ax_parity.set_ylabel("PINN prediction")
    ax_parity.legend(loc="upper left")

    fig.suptitle("Coupled non-Newtonian flow / heat-transfer PINN validation "
                 "(theta-formulation, hard walls)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"{prefix}[Figure] Validation figure written to {save_path}")


def plot_rheology_validation(
    y: np.ndarray,
    gamma_pinn: np.ndarray,
    mu_pinn: np.ndarray,
    gamma_ref: np.ndarray,
    mu_ref: np.ndarray,
    metrics_mu: Dict[str, float],
    rheo: RheologyParameters,
    save_path: Path,
    prefix: str = "",
) -> None:
    """Rheological cross-validation: shear-rate invariant and apparent viscosity."""
    fig, (ax_gamma, ax_mu) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)

    ax_gamma.plot(y, gamma_ref, color="black", lw=2.2,
                  label="Semi-analytical reference")
    ax_gamma.plot(y, gamma_pinn, color="purple", ls="--", lw=1.8, marker="o",
                  markersize=2.6, markevery=20, label="PINN")
    ax_gamma.set_title("(a) Shear-rate invariant")
    ax_gamma.set_xlabel(r"$y/H$")
    ax_gamma.set_ylabel(r"$\dot{\gamma}(y)$")
    ax_gamma.legend(loc="center")

    ax_mu.plot(y, mu_ref, color="black", lw=2.2,
               label="Semi-analytical reference")
    ax_mu.plot(y, mu_pinn, color="seagreen", ls="--", lw=1.8, marker="s",
               markersize=2.6, markevery=20,
               label=f"PINN (RMSE = {metrics_mu['rmse']:.2e})")

    ax_mu.axhline(rheo.mu_0, color="grey", ls=":", lw=1.0)
    ax_mu.text(0.02, 0.935, r"$\mu_0$ (zero-shear)", transform=ax_mu.transAxes,
               fontsize=8, color="grey")
    ax_mu.axhline(rheo.mu_inf, color="grey", ls=":", lw=1.0)
    ax_mu.text(0.02, 0.055, r"$\mu_\infty$ (infinite-shear)", transform=ax_mu.transAxes,
               fontsize=8, color="grey")
    ax_mu.set_ylim(0.0, 1.08 * rheo.mu_0)
    ax_mu.set_title("(b) Apparent viscosity (shear-thinning)")
    ax_mu.set_xlabel(r"$y/H$")
    ax_mu.set_ylabel(r"$\mu_{\mathrm{app}}(y)$")
    ax_mu.legend(loc="lower center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"{prefix}[Figure] Rheology validation figure written to {save_path}")


def _get_cmap(name: str):
    """Version-robust colormap accessor (matplotlib >= 3.5 / older fallback)."""
    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError): 
        return plt.cm.get_cmap(name) 


def plot_parametric_profiles(records: List["RunRecord"], cases: List[CaseConfig],
                             study: StudyParameters, save_path: Path) -> None:
    """Physics figure of the two-factor design: 1-D slices of the (lambda, n) grid.

    Panel (a): velocity across lambda at the baseline power-law index.
    Panels (b)-(d): velocity, apparent viscosity and temperature (primary Br)
    across the power-law index at the strongest relaxation time - the slice
    where the (lambda, n) interaction is most visible. With a single-factor
    grid the slices degenerate gracefully to the v2.1 figure. Solid lines:
    semi-analytical benchmark; dashed lines: PINN (primary seed of each case).
    """
    by_tag = {r.tag: r for r in records if r.profiles is not None}
    br_primary = study.brinkman_reporting_values[0]
    lam_values = sorted(study.relaxation_times)
    n_values = sorted(study.power_law_indices)
    n_base = n_values[len(n_values) // 2]  
    lam_top = lam_values[-1]               
    cmap = _get_cmap("viridis")

    def _color(index: int, count: int):
        return cmap(0.15 + 0.7 * index / max(1, count - 1))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), dpi=300)
    (ax_u_lam, ax_u_n), (ax_mu_n, ax_T_n) = axes

    for i, lam in enumerate(lam_values):
        record = by_tag.get(f"lam{lam:g}_n{n_base:g}")
        if record is None:
            continue
        p = record.profiles
        color = _color(i, len(lam_values))
        ax_u_lam.plot(p["y"], p["u_ref"], color=color, lw=2.0,
                      label=rf"$\lambda={lam:g}$")
        ax_u_lam.plot(p["y"], p["u_pinn"], color=color, lw=1.5, ls="--")
    ax_u_lam.set_title(rf"(a) Velocity across $\lambda$ (n = {n_base:g})")
    ax_u_lam.set_xlabel(r"$y/H$")
    ax_u_lam.set_ylabel(r"$u(y)$")
    proxy = [Line2D([0], [0], color="k", lw=2.0, ls="-", label="Semi-analytical"),
             Line2D([0], [0], color="k", lw=1.5, ls="--", label="PINN")]
    handles, _ = ax_u_lam.get_legend_handles_labels()
    ax_u_lam.legend(handles=proxy + handles, loc="lower center")

    for i, n in enumerate(n_values):
        record = by_tag.get(f"lam{lam_top:g}_n{n:g}")
        if record is None:
            continue
        p = record.profiles
        color = _color(i, len(n_values))
        label = rf"$n={n:g}$"
        ax_u_n.plot(p["y"], p["u_ref"], color=color, lw=2.0, label=label)
        ax_u_n.plot(p["y"], p["u_pinn"], color=color, lw=1.5, ls="--")
        ax_mu_n.plot(p["y"], p["mu_ref"], color=color, lw=2.0, label=label)
        if "mu_pinn" in p:
            ax_mu_n.plot(p["y"], p["mu_pinn"], color=color, lw=1.5, ls="--")
        ax_T_n.plot(p["y"], br_primary * p["theta_ref"], color=color, lw=2.0,
                    label=label)
        ax_T_n.plot(p["y"], br_primary * p["theta_pinn"], color=color, lw=1.5, ls="--")

    ax_u_n.set_title(rf"(b) Velocity across $n$ ($\lambda = {lam_top:g}$)")
    ax_u_n.set_xlabel(r"$y/H$")
    ax_u_n.set_ylabel(r"$u(y)$")
    ax_u_n.legend(loc="lower center")

    ax_mu_n.axhline(DEFAULT_RHEO.mu_0, color="grey", ls=":", lw=1.0)
    ax_mu_n.axhline(DEFAULT_RHEO.mu_inf, color="grey", ls=":", lw=1.0)
    ax_mu_n.set_ylim(0.0, 1.08 * DEFAULT_RHEO.mu_0)
    ax_mu_n.set_title(rf"(c) Apparent viscosity across $n$ ($\lambda = {lam_top:g}$)")
    ax_mu_n.set_xlabel(r"$y/H$")
    ax_mu_n.set_ylabel(r"$\mu_{\mathrm{app}}(y)$")
    ax_mu_n.legend(loc="lower center")

    ax_T_n.set_title(rf"(d) Temperature at Br = {br_primary:g} across $n$ "
                     rf"($\lambda = {lam_top:g}$)")
    ax_T_n.set_xlabel(r"$y/H$")
    ax_T_n.set_ylabel(r"$T(y)$")
    ax_T_n.legend(loc="lower center")

    fig.suptitle("Parametric study across the (lambda, n) rheological grid "
                 "(solid: semi-analytical benchmark; dashed: PINN)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Figure] Parametric profiles figure written to {save_path}")


def _rel_l2_samples(records: List["RunRecord"], tag: str, field: str,
                    br: Optional[float] = None) -> List[float]:
    """Collect the relative-L2 samples of one (case, field[, Br]) across seeds."""
    values = []
    for record in records:
        if record.tag != tag:
            continue
        for m in record.metrics_records:
            if m["field"] != field:
                continue
            if br is None or m["brinkman_number"] == br:
                values.append(m["relative_l2"])
    return values


def plot_study_summary(records: List["RunRecord"], cases: List[CaseConfig],
                       study: StudyParameters, save_path: Path) -> None:
    """Accuracy overview: relative L2 error per case and field (log scale).

    Bars are the mean over seeds; error bars show +/- one standard deviation.
    T is reported at the primary Br (relative metrics are Br-invariant by
    construction under the theta-formulation).
    """
    br_primary = study.brinkman_reporting_values[0]
    fields = (("u(y)", None, "crimson", r"$u(y)$"),
              ("T(y)", br_primary, "navy", r"$T(y)$"),
              ("mu_app(y)", None, "seagreen", r"$\mu(y)$"))

    x = np.arange(len(cases))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=300)
    for k, (field, br, color, label) in enumerate(fields):
        means: List[float] = []
        stds: List[float] = []
        for case in cases:
            values = _rel_l2_samples(records, case.tag, field, br)
            means.append(float(np.mean(values)) if values else float("nan"))
            stds.append(float(np.std(values)) if len(values) > 1 else 0.0)
        ax.bar(x + (k - 1) * width, means, width, yerr=stds, capsize=2.5,
               color=color, label=label, alpha=0.9)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([case.tag for case in cases], rotation=25, fontsize=8.0)
    ax.set_ylabel("Relative L2 error")
    ax.set_title(r"Study-level accuracy across the (lambda, n) grid "
                 rf"(T at Br = {br_primary:g}; Br-invariant by formulation)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Figure] Study summary figure written to {save_path}")

@dataclass
class RunRecord:
    """Complete record of one training run (aggregated at study level)."""

    tag: str
    relaxation_time: float
    power_law_index: float
    seed: int
    detailed: bool
    metrics_records: List[Dict[str, Any]]
    timings: Dict[str, float]
    final_losses: Dict[str, Dict[str, float]]
    final_total_loss: float
    profiles: Optional[Dict[str, np.ndarray]] = None
    loss_history: Optional[pd.DataFrame] = None


def build_study_cases(study: StudyParameters, default_rheo: RheologyParameters,
                      default_flow: FlowParameters) -> List[CaseConfig]:
    """Build the (lambda, n) training grid (Br is a reporting axis only).

    The iteration order follows `study.relaxation_times` (strongest first by
    default) crossed with `study.power_law_indices`, so the stiffest corner
    is executed first.
    """
    cases: List[CaseConfig] = []
    for lam in study.relaxation_times:
        for n in study.power_law_indices:
            tag = f"lam{lam:g}_n{n:g}"
            cases.append(CaseConfig(
                tag=tag,
                rheology=replace(default_rheo, relaxation_time=lam, power_law_index=n),
                flow=default_flow,
            ))
    return cases


def build_run_plan(cases: List[CaseConfig],
                   study: StudyParameters) -> List[Tuple[CaseConfig, int, bool]]:
    """Execution plan: (case, seed, detailed_artifacts).

    EVERY case is replicated over ALL seeds (full-grid statistics); the first
    seed of each case produces the detailed artefact set.
    """
    plan: List[Tuple[CaseConfig, int, bool]] = []
    for case in cases:
        for index, seed in enumerate(study.seeds):
            plan.append((case, seed, index == 0))
    return plan


def execute_run(case: CaseConfig, train: TrainingParameters, study: StudyParameters,
                seed: int, detailed: bool, run_dir: Path,
                reference: ReferenceSolution) -> RunRecord:
    """Execute one full pipeline run: training -> validation -> export."""
    prefix = f"[{case.tag} | seed {seed}]"
    print(f"\n{'-' * 79}\n{prefix} run start (detailed artifacts: {detailed})\n{'-' * 79}")
    t_start = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)


    enforce_reproducibility(seed, train.float_dtype)


    H = case.flow.half_channel_height
    y_grid = np.linspace(-H, H, study.validation_points)
    print(f"{prefix} Benchmark (cached): u(0)={reference.u.max():.6f} | "
          f"theta(0)={reference.theta.max():.6f} | "
          f"mu(H)={reference.mu_app[-1]:.6f} | "
          f"gamma(H)={reference.gamma_dot[-1]:.6f}")


    model = build_model(case, train, prefix)
    history_adam, history_lbfgs, timings = train_model(
        model, train, prefix, fine_history=detailed
    )


    t_post = time.perf_counter()
    history_df = export_loss_history(history_adam, history_lbfgs, run_dir, prefix)
    if detailed:
        plot_convergence_history(history_df, run_dir / "convergence_loss_history.png", prefix)

    y_column = y_grid.reshape(-1, 1)
    prediction = np.asarray(model.predict(y_column), dtype=np.float64)
    u_pinn = prediction[:, 0]
    theta_pinn = prediction[:, 1]        
    
    T_pinn = {br: br * theta_pinn for br in study.brinkman_reporting_values}
    T_ref = {br: br * reference.theta for br in study.brinkman_reporting_values}

    metrics_u = compute_validation_metrics(u_pinn, reference.u)
    metrics_theta = compute_validation_metrics(theta_pinn, reference.theta)
    metrics_T = {br: compute_validation_metrics(T_pinn[br], T_ref[br])
                 for br in study.brinkman_reporting_values}


    gamma_pinn: Optional[np.ndarray] = None
    mu_pinn: Optional[np.ndarray] = None
    try:
        rheo_pred = np.asarray(
            model.predict(y_column, operator=make_rheology_operator(case.rheology)),
            dtype=np.float64,
        )
        gamma_pinn, mu_pinn = rheo_pred[:, 1], rheo_pred[:, 2]
        metrics_mu = compute_validation_metrics(mu_pinn, reference.mu_app)
    except Exception as exc:  
        print(f"{prefix} [WARNING] Operator-based rheology extraction unavailable ({exc}).")
        metrics_mu = None


    metrics_records: List[Dict[str, Any]] = [
        {"field": "u(y)", "brinkman_number": "", **metrics_u},
        {"field": "theta(y)", "brinkman_number": "", **metrics_theta},
    ]
    for br in study.brinkman_reporting_values:
        metrics_records.append({"field": "T(y)", "brinkman_number": br, **metrics_T[br]})
    if metrics_mu is not None:
        metrics_records.append({"field": "mu_app(y)", "brinkman_number": "", **metrics_mu})

    metrics_df = pd.DataFrame(metrics_records)[
        ["field", "brinkman_number", *VALIDATION_METRIC_KEYS]
    ]
    metrics_df.to_csv(run_dir / "validation_metrics.csv", index=False)


    final_losses: Dict[str, Dict[str, float]] = {}
    for phase_name, phase_frame in history_df.groupby("phase", sort=False):
        last_row = phase_frame.iloc[-1]
        entry = {"loss_total": float(last_row["loss_total"])}
        entry.update({name: float(last_row[name]) for name in LOSS_COMPONENT_NAMES})
        final_losses[phase_name] = entry
    final_total = final_losses.get("L-BFGS", {}).get("loss_total", float("nan"))


    profiles: Optional[Dict[str, np.ndarray]] = None
    if detailed:
        br_primary = study.brinkman_reporting_values[0]
        profiles = {
            "y": y_grid,
            "u_ref": reference.u, "u_pinn": u_pinn,
            "theta_ref": reference.theta, "theta_pinn": theta_pinn,
            "gamma_ref": reference.gamma_dot, "mu_ref": reference.mu_app,
        }
        if mu_pinn is not None:
            profiles["gamma_pinn"] = gamma_pinn
            profiles["mu_pinn"] = mu_pinn

        profiles_frame: Dict[str, Any] = {
            "case": case.tag, "seed": seed, "y": y_grid,
            "u_pinn": u_pinn, "u_reference": reference.u,
            "absolute_error_u": np.abs(u_pinn - reference.u),
            "theta_pinn": theta_pinn, "theta_reference": reference.theta,
            "absolute_error_theta": np.abs(theta_pinn - reference.theta),
            "gamma_dot_reference": reference.gamma_dot,
            "mu_app_reference": reference.mu_app,
        }
        for br in study.brinkman_reporting_values:
            profiles_frame[f"T_pinn_Br{br:g}"] = T_pinn[br]
            profiles_frame[f"T_reference_Br{br:g}"] = T_ref[br]
            profiles_frame[f"absolute_error_T_Br{br:g}"] = np.abs(T_pinn[br] - T_ref[br])
        if mu_pinn is not None:
            profiles_frame["gamma_dot_pinn"] = gamma_pinn
            profiles_frame["mu_app_pinn"] = mu_pinn
            profiles_frame["absolute_error_mu"] = np.abs(mu_pinn - reference.mu_app)
        pd.DataFrame(profiles_frame).to_csv(
            run_dir / "validation_profiles_data.csv", index=False
        )
        print(f"{prefix}[Export] Validation profiles written to "
              f"{run_dir / 'validation_profiles_data.csv'}")

        figure_metrics = {"u(y)": metrics_u, "T(y)": metrics_T[br_primary]}
        plot_validation_profiles(
            y_grid, u_pinn, T_pinn[br_primary], reference.u, T_ref[br_primary],
            figure_metrics, run_dir / "pinn_non_newtonian_validation.png",
            prefix, br_label=f"{br_primary:g}",
        )
        print_metric_table(metrics_records, title=f"{prefix} validation metrics:")
        if mu_pinn is not None:
            plot_rheology_validation(
                y_grid, gamma_pinn, mu_pinn, reference.gamma_dot, reference.mu_app,
                metrics_mu, case.rheology,
                run_dir / "apparent_viscosity_validation.png", prefix,
            )

    timings["post_processing_s"] = time.perf_counter() - t_post
    timings["total_s"] = time.perf_counter() - t_start
    print(f"{prefix} completed in {timings['total_s']:.1f} s | "
          f"final total loss = {final_total:.3e}")

    return RunRecord(
        tag=case.tag,
        relaxation_time=case.rheology.relaxation_time,
        power_law_index=case.rheology.power_law_index,
        seed=seed,
        detailed=detailed,
        metrics_records=metrics_records,
        timings=timings,
        final_losses=final_losses,
        final_total_loss=final_total,
        profiles=profiles,
        loss_history=history_df,
    )


def build_raw_metrics_table(records: List[RunRecord]) -> pd.DataFrame:
    """One row per (run x field [x Br]) validation record."""
    rows = []
    for record in records:
        for m in record.metrics_records:
            rows.append({
                "case": record.tag,
                "relaxation_time": record.relaxation_time,
                "power_law_index": record.power_law_index,
                "seed": record.seed,
                "field": m["field"],
                "brinkman_number": m["brinkman_number"],
                **{key: m[key] for key in VALIDATION_METRIC_KEYS},
            })
    return pd.DataFrame(rows)


def build_summary_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(case, field, Br) mean / std / count aggregation over seeds."""
    group_keys = ["case", "relaxation_time", "power_law_index", "field",
                  "brinkman_number"]
    grouped = raw_df.groupby(group_keys, sort=False)[list(VALIDATION_METRIC_KEYS)]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        c if isinstance(c, str) else (f"{c[0]}_{c[1]}" if c[1] else c[0])
        for c in summary.columns
    ]
    return summary


def _format_mean_std(mean: float, std: float, count: float) -> str:
    """Compact 'mean +/- std' formatting for the console summary table."""
    if count > 1 and np.isfinite(std) and std > 0.0:
        return f"{mean:.2e} +/- {std:.1e}"
    return f"{mean:.2e}"


def print_study_summary(summary_df: pd.DataFrame, cases: List[CaseConfig],
                        study: StudyParameters) -> None:
    """Console rendering of the study-level aggregated accuracy."""
    lookup = summary_df.set_index(["case", "field", "brinkman_number"])
    br_primary = study.brinkman_reporting_values[0]
    rule = "-" * 96
    print("\n" + rule)
    print(" STUDY SUMMARY - relative L2 error vs the semi-analytical benchmark "
          "(mean +/- std over seeds)")
    print(" NOTE: T relative metrics are Br-invariant by construction (theta = T/Br);")
    print("       physical RMSE scales linearly: RMSE_T(Br) = Br * RMSE_theta "
          "(per-Br values in study_summary.csv).")
    print(rule)
    print(f"{'Case':<16}{'n seeds':>8}{'u(y)':>24}{'T(y)':>24}{'mu_app(y)':>24}")
    print(rule)
    for case in cases:
        cells = []
        n_seeds = ""
        for field, br_key in (("u(y)", ""), ("T(y)", br_primary), ("mu_app(y)", "")):
            try:
                row = lookup.loc[(case.tag, field, br_key)]
                cells.append(_format_mean_std(
                    float(row["relative_l2_mean"]),
                    float(row["relative_l2_std"]),
                    float(row["relative_l2_count"]),
                ))
                n_seeds = str(int(row["relative_l2_count"]))
            except KeyError:
                cells.append("--")
        print(f"{case.tag:<16}{n_seeds:>8}{cells[0]:>24}{cells[1]:>24}{cells[2]:>24}")
    print(rule)


def export_study_loss_history(records: List[RunRecord], save_path: Path) -> None:
    """Concatenate the convergence history of ALL runs into one CSV."""
    frames = []
    for record in records:
        if record.loss_history is None:
            continue
        frame = record.loss_history.copy()
        frame.insert(0, "seed", record.seed)
        frame.insert(0, "case", record.tag)
        frames.append(frame)
    if not frames:
        return
    pd.concat(frames, ignore_index=True).to_csv(save_path, index=False)
    print(f"[Export] Study-level loss history written to {save_path}")


def export_benchmark_profiles(cases: List[CaseConfig],
                              references: Dict[str, ReferenceSolution],
                              study: StudyParameters, save_path: Path) -> None:
    """Export the exact benchmark fields of every case (with T per Br)."""
    frames = []
    for case in cases:
        ref = references[case.tag]
        frame = pd.DataFrame({
            "case": case.tag,
            "relaxation_time": case.rheology.relaxation_time,
            "power_law_index": case.rheology.power_law_index,
            "y": ref.y,
            "u_reference": ref.u,
            "theta_reference": ref.theta,
            "gamma_dot_reference": ref.gamma_dot,
            "mu_app_reference": ref.mu_app,
        })
        for br in study.brinkman_reporting_values:
            frame[f"T_reference_Br{br:g}"] = br * ref.theta
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(save_path, index=False)
    print(f"[Export] Benchmark reference profiles written to {save_path}")


def export_study_design(cases: List[CaseConfig], study: StudyParameters,
                        save_path: Path) -> None:
    """Export the design of experiments."""
    rows = [{
        "case": case.tag,
        "relaxation_time": case.rheology.relaxation_time,
        "power_law_index": case.rheology.power_law_index,
        "seeds": list(study.seeds),
        "brinkman_reporting_values": list(study.brinkman_reporting_values),
    } for case in cases]
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"[Export] Study design written to {save_path}")


def save_study_metadata(records: List[RunRecord], total_wall_time_s: float,
                        save_path: Path) -> None:
    """Persist the complete study-level reproducibility record (JSON)."""
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "formulation": {
            "version": "2.2",
            "temperature": "theta = T/Br (Br-invariant); T(Br) = Br*theta exact",
            "wall_constraints": "hard, multiplicative (1 - (y/H)^2)",
            "loss_terms": list(LOSS_COMPONENT_NAMES),
        },
        "reproducibility": {
            "float_dtype": TRAIN.float_dtype,
            "dde_backend": getattr(getattr(dde, "backend", None), "backend_name", "pytorch"),
            **{k: (v if isinstance(v, (str, bool)) else str(v))
               for k, v in accelerator_summary().items()},
        },
        "software_versions": collect_version_metadata(),
        "study_design": asdict(STUDY),
        "default_parameters": {
            "rheology": asdict(DEFAULT_RHEO),
            "flow": asdict(DEFAULT_FLOW),
        },
        "training_configuration": asdict(TRAIN),
        "runs": [
            {
                "case": r.tag,
                "seed": r.seed,
                "relaxation_time": r.relaxation_time,
                "power_law_index": r.power_law_index,
                "detailed": r.detailed,
                "timings_s": r.timings,
                "final_losses": r.final_losses,
                "final_total_loss": r.final_total_loss,
                "metrics": r.metrics_records,
            }
            for r in records
        ],
        "total_wall_time_s": total_wall_time_s,
    }
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"[Export] Study metadata written to {save_path}")


def main() -> None:
    """Orchestrate the full study: design -> benchmarks -> runs -> aggregation."""
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print_configuration_banner()


    cases = build_study_cases(STUDY, DEFAULT_RHEO, DEFAULT_FLOW)
    plan = build_run_plan(cases, STUDY)
    n_runs = len(plan)
    print(f"[Study] {len(cases)} case configuration(s) x {len(STUDY.seeds)} seed(s) "
          f"= {n_runs} training run(s) ({STUDY.relaxation_times[0]:g}-corner first). "
          f"Br is a reporting axis ({list(STUDY.brinkman_reporting_values)}) "
          f"- not a training axis.")
    print("[Study] Tip: the smartest smoke test is the stiff corner alone, e.g. "
          "StudyParameters(relaxation_times=(5.0,), power_law_indices=(0.3,), "
          "seeds=(42,)) - see module docstring.")


    H = cases[0].flow.half_channel_height
    y_grid = np.linspace(-H, H, STUDY.validation_points)
    references: Dict[str, ReferenceSolution] = {}
    for case in cases:
        references[case.tag] = compute_reference_solution(
            y_grid, case.rheology, case.flow, STUDY.quadrature_points, verbose=True
        )
        ref = references[case.tag]
        theta0 = float(ref.theta.max())
        t_summary = " | ".join(
            f"T(0; Br={br:g})={br * theta0:.6f}" for br in STUDY.brinkman_reporting_values
        )
        print(f"[Benchmark | {case.tag}] u(0)={ref.u.max():.6f} | "
              f"theta(0)={theta0:.6f} | {t_summary} | "
              f"mu(H)={ref.mu_app[-1]:.6f} | gamma(H)={ref.gamma_dot[-1]:.6f}")
        if len(STUDY.brinkman_reporting_values) >= 2:
            br_lo, br_hi = (STUDY.brinkman_reporting_values[0],
                            STUDY.brinkman_reporting_values[-1])
            print(f"[Scaling   | {case.tag}] T(0) ratio Br={br_hi:g}/Br={br_lo:g} "
                  f"= {br_hi / br_lo:.6f} (exact linearity of T in Br)")

    export_study_design(cases, STUDY, OUTPUT_DIR / "study_design.csv")
    export_benchmark_profiles(cases, references, STUDY,
                               OUTPUT_DIR / "benchmark_reference_profiles.csv")


    records: List[RunRecord] = []
    for index, (case, seed, detailed) in enumerate(plan, start=1):
        print(f"\n{'#' * 79}")
        print(f" RUN {index} / {n_runs}")
        print(f"{'#' * 79}")
        run_dir = OUTPUT_DIR / "runs" / f"{case.tag}_seed{seed}"
        records.append(execute_run(case, TRAIN, STUDY, seed, detailed, run_dir,
                                   references[case.tag]))


    export_study_loss_history(records, OUTPUT_DIR / "study_loss_history.csv")

    raw_df = build_raw_metrics_table(records)
    raw_path = OUTPUT_DIR / "study_metrics_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"\n[Export] Raw study metrics written to {raw_path}")

    summary_df = build_summary_table(raw_df)
    summary_path = OUTPUT_DIR / "study_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[Export] Aggregated study summary written to {summary_path}")

    print_study_summary(summary_df, cases, STUDY)


    plot_study_summary(records, cases, STUDY, OUTPUT_DIR / "study_summary.png")
    plot_parametric_profiles(records, cases, STUDY,
                             OUTPUT_DIR / "parametric_profiles.png")

    total_time = time.perf_counter() - t_start
    save_study_metadata(records, total_time, OUTPUT_DIR / "run_metadata.json")


    rule = "=" * 79
    print("\n" + rule)
    print(" EXECUTION SUMMARY")
    print(rule)
    print(f" Runs executed     : {len(records)}")
    print(f" Total wall-clock  : {total_time:8.1f} s ({total_time / 60.0:.1f} min, "
          f"{total_time / 3600.0:.1f} h)")
    print(f" Mean time per run : "
          f"{float(np.mean([r.timings['total_s'] for r in records])):8.1f} s")
    print(" Study-level artifacts:")
    for name in ("study_design.csv", "benchmark_reference_profiles.csv",
                 "study_loss_history.csv", "study_metrics_raw.csv",
                 "study_summary.csv", "study_summary.png",
                 "parametric_profiles.png", "run_metadata.json"):
        print(f"   - {OUTPUT_DIR / name}")
    print(f" Per-run artifacts : {OUTPUT_DIR / 'runs'}")
    print(rule)


if __name__ == "__main__":
    main()