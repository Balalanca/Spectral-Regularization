# Spectral Regularization for Image Restoration

[![Tests](https://github.com/Balalanca/Spectral-Regularization/actions/workflows/tests.yml/badge.svg)](https://github.com/Balalanca/Spectral-Regularization/actions)

A numerical investigation of ill-posed inverse problems, Fourier-domain image restoration and regularization methods.

## Overview

Recovering an image from noisy and blurred observations is an ill-posed inverse problem. Direct inversion amplifies high-frequency noise, making regularization essential for stable reconstructions.

## Mathematical Framework

The forward observation model is

$$
g=Kf+\eta
$$

Generalized Tikhonov regularization solves

$$
f_\mu=\arg\min_f\left(\|Kf-g\|_2^2+\mu\|Pf\|_2^2\right).
$$

In the Fourier domain, the reconstruction becomes

$$
\hat f_\mu=\frac{\overline{\hat K}\hat g}{|\hat K|^2+\mu|\hat P|^2}.
$$

Three regularization penalties are investigated:

| Method | Squared penalty symbol |
|---|---|
| L2 | $1$ |
| H1 | $\omega^2$ |
| H2 | $\omega^4$ |

Hard spectral cutoff is also implemented as an alternative reconstruction strategy.

## Numerical Experiments

- Gaussian blurring in the Fourier domain
- Additive Gaussian noise at 20 dB and 40 dB SNR
- Parameter sweeps across 25 regularization strengths
- Comparison of L2, H1 and H2 penalties
- Relative reconstruction error analysis
- Analytical bias-variance decomposition
- Monte Carlo validation of expected reconstruction risk

## Reconstruction Results

### 20 dB SNR

![Reconstruction at 20 dB](results/reconstruction_20.png)

### 40 dB SNR

![Reconstruction at 40 dB](results/reconstruction_40.png)

## Regularization Parameter Analysis

The regularization parameter controls the trade-off between reconstruction fidelity and numerical stability.

| 20 dB | 40 dB |
|---|---|
| ![Error](results/error_20.png) | ![Error](results/error_40.png) |

## Bias-Variance Trade-off

The expected squared reconstruction error admits the decomposition

$$
\mathbb{E}\|f_\mu-f\|^2
=
\|\mathbb{E}[f_\mu]-f\|^2
+
\mathbb{E}\|f_\mu-\mathbb{E}[f_\mu]\|^2.
$$

| 20 dB | 40 dB |
|---|---|
| ![Risk](results/risk_20.png) | ![Risk](results/risk_40.png) |

## Reproducibility

Experiments use a deterministic synthetic image, seeded noise generation, circular convolution and an angular-frequency grid.

Parameter selection uses known ground truth as an oracle benchmark.

    pip install -r requirements.txt
    python -m pytest -q
    python experiment.py

Generated figures are saved in `results/`.

## Technologies

Python, NumPy, Matplotlib, Fourier analysis, inverse problems and numerical methods.