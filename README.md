# Physics-Informed Neural Networks for Coupled Non-Newtonian Flow and Heat Transport

Official code repository for the manuscript:  
**"Coupled Non-Newtonian Flow and Heat Transport in a Plane Channel: A Physics-Informed Neural Network Study"**

**Author:** Fodil Mohammed El Amine  
**Affiliation:** University Centre of Maghnia, B.P 19, 13000 Maghnia, Algeria  
**Contact:** [votre_email@domaine.com]

---

## Overview

This repository contains the official standalone Python implementation for modeling steady, fully developed laminar flow of an incompressible generalized Newtonian fluid under Carreau–Yasuda shear-thinning rheology between parallel plates, accounting for internal heat generation by viscous dissipation.

The numerical solver uses **DeepXDE** with a **PyTorch** backend. It integrates a novel $\theta$-formulation to resolve thermal gradient flow pathologies, enforces hard boundary constraints directly through network architecture, and validates PINN predictions against a high-precision semi-analytical quadrature benchmark over a two-factor $(\lambda, n)$ parameter grid.

---

## Key Features

- **$\theta$-Formulation**: Trains on the Brinkman-invariant thermal field $\theta = T / \text{Br}$, eliminating energy loss gradient flow imbalances across arbitrary Brinkman numbers.
- **Hard Boundary Constraints**: Exact satisfaction of wall conditions $u(\pm H) = 0$ and $\theta(\pm H) = 0$ by construction via structural neural network parameterization.
- **Semi-Analytical Benchmark**: High-precision ground truths generated via quadrature integration with runtime finite-difference verification.
- **Automated Artifact Export**: Automatically logs per-run histories, raw validation metrics, reference profiles, and complete reproducibility metadata (`run_metadata.json`).

---

## Repository Structure