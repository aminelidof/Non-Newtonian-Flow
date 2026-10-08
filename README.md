# Physics-Informed Neural Networks for Coupled Non-Newtonian Flow and Heat Transport
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![DeepXDE](https://img.shields.io/badge/DeepXDE-1.9+-orange.svg)](https://github.com/lululxvi/deepxde)

Official code repository for the manuscript:  
**"A validated physics-informed neural network framework for Carreau–Yasuda thermofluidics: benchmark, diagnosis, and Brinkman-invariant formulation"**

**Authors:** Mohamed El Amine Fodil<sup>1,2,*</sup>, Merwan Abdelbari<sup>3</sup>, Meriem Fodil<sup>3</sup>  
<sup>1</sup> Department of Hydraulics, Maghnia University Centre, Tlemcen, Algeria  
<sup>2</sup> Laboratoire Ingénierie et Sciences Appliquées (IScApp), Maghnia, Tlemcen, Algeria  
<sup>3</sup> Department of Mechanics, Hassiba Ben Bouali University, Chlef, Algeria  
<sup>*</sup> **Corresponding Author Email:** fodilmedam@gmail.com


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

## 📌 Repository Structure

```plaintext
TRIZ-PINN-Crustal-Stress/
├── output_results               
├── main.py              
├── LICENSE           
├── README.md         
└── requirements.txt  
