# SABR Equipped with AI Wings — Training and Evaluation Summary

**Source:**  
Hideharu Funahashi (2022), *“SABR Equipped with AI Wings”*  
Kanagawa University, Faculty of Economics  
Supported by JSPS KAKENHI Grant JP22K13436  

---

## 🧠 Overview

The paper proposes an **Artificial Neural Network (ANN)** correction framework for the **SABR volatility model**.  
Instead of directly learning implied volatilities, the ANN learns the **residual difference** between:

\[
D(\xi) = \sigma_{MC}(\xi) - \sigma_{App}(\xi)
\]

where:
- \( \sigma_{MC} \): Implied volatility from Monte Carlo simulation  
- \( \sigma_{App} \): Analytical asymptotic approximation (Hagan or Antonov–Spector zero mapping)

The final predicted volatility is:

\[
\sigma_{DANN}(\xi) = \sigma_{App}(\xi) + D_{ANN}(\xi)
\]

This **difference-learning approach** improves convergence, accuracy in wings (deep ITM/OTM), and training efficiency.

---

## ⚙️ Training Procedure

### Steps

1. **Generate parameter samples**  
   Random vectors \(\xi_k = \{ \alpha, \beta, \nu, \rho, r, K, S_0, T \}\) for \(k = 1, …, N\).

2. **Compute Monte Carlo implied volatilities** \( \sigma_{MC}(\xi_k) \).

3. **Compute analytical approximations**
   - **Short maturity (T < 5 yrs):** Hagan’s SABR asymptotic formula  
   - **Long maturity (T ≥ 5 yrs):** Antonov–Spector zero-mapping

4. **Compute residuals:** \( D(\xi_k) = \sigma_{MC} - \sigma_{App} \).

5. **Train ANN** on pairs \(\{ \xi_k, D(\xi_k) \}\) using the Adam optimizer.

6. **Predict** implied volatilities as \( \sigma_{DANN} = \sigma_{App} + D_{ANN} \).

---

## 🧩 Parameter Sampling Ranges

### Classical SABR (Short/Long Maturity)

| Parameter | Range | Description |
|------------|--------|-------------|
| α | 0.05 – 0.6 | Initial vol level |
| β | 0.3 – 0.9 | CEV elasticity |
| ν | 0.05 – 0.9 | Vol-of-vol |
| ρ | –0.75 – 0.75 | Correlation |
| T | 1 – 10 years | Maturity |
| K | 0.4f – 2f | Strike, see strike logic below |

**Strike limits:**
\[
K_1 = \max(f - 1.8\sqrt{V}, 0.4f), \quad
K_2 = \min(f + 2\sqrt{V}, 2f)
\]
21 equally spaced strikes between \(K_1\) and \(K_2\).

---

### Free-Boundary SABR (Negative Rate Case)

| Parameter | Range | Notes |
|------------|--------|-------|
| f | 0.0025 – 0.01 | Forward rate |
| α′ | 0.4 – 0.8 | Vol level |
| β | 0.1 – 0.3 | Elasticity |
| ν | 0.15 – 0.4 | Vol-of-vol |
| ρ | –0.5 – 0.5 | Correlation |
| T | 3 years | Fixed |

Filtered to keep only cases with:
\[
20 \text{ bps} < \sigma_{ZM(fb)} < 50 \text{ bps}
\]

---

## 📚 Dataset and Monte Carlo Settings

| Setting | Short / Long Maturity | Free-Boundary SABR |
|----------|-----------------------|--------------------|
| Total samples (N) | 105,000 | ~105,000 |
| After filtering | 10,454 (short), 104,059 (long) | ~95,000 |
| Strikes per sample | 21 | 21 |
| Total labeled samples | ≈ 2.2 million | ≈ 2.0 million |
| MC trials | 1,000,000 | 500,000 |
| Time steps | 300 | 500 |
| Integration scheme | Chen et al. (2012) | Antonov et al. (2015) |
| Runtime per training | ~1 ms/σ_Hagan; 2–3 ms/σ_DANN | ~7000 sec per trial |

---

## 🧠 Network Architecture and Hyperparameters

| Component | Setting |
|------------|----------|
| Architecture | Fully connected feedforward ANN |
| Hidden layers | 5 |
| Neurons per layer | 32 |
| Activation | ReLU (classical), PReLU (free-boundary) |
| Optimizer | Adam |
| Loss function | Mean squared error (MSE) |
| Epochs | 100 |
| Batch size | 128 |
| Train/test split | 80% / 20% |

---

## 📈 Evaluation Metrics

- **Primary metrics:**  
  - Mean Squared Error (MSE)  
  - Relative percentage error of implied volatility  
  - Visual fit of implied volatility smiles across maturities

- **Benchmark comparison:**  
  - Direct mapping ANN \( M: \xi → σ_{ANN} \)  
  - Residual-based ANN \( M_D: \xi → σ_{DANN} \)

---

## ✅ Results and Performance

| Case | Model | Avg. Error vs MC | Observations |
|-------|--------|------------------|---------------|
| Short maturity (T=3) | σ_Hagan | up to **5–10%** error in wings | Deteriorates deep ITM/OTM |
| Short maturity | σ_DANN | < **0.5%** RMSE | Almost identical to MC |
| Long maturity (T=10) | σ_Hagan | > **10%** in far wings | Loses stability |
| Long maturity | σ_DANN | < **1%** | Stable even for high ν |
| Free-boundary SABR | σ_ZM | deviates 20–30% for β<0.2 | inaccurate for low β |
| Free-boundary SABR | σ_DANN(fb) | < **1%** overall | robust for negative rates |

**Convergence:**  
- \( M_D \) (residual ANN) converged **3× faster** than direct ANN.  
- Achieved high accuracy with **<10% of training data** compared to full direct approach.

---

## 📊 Visual Summary (Described in Paper)

- ANN-corrected implied volatility curves match MC results **almost perfectly**, even for:
  - Long maturities (up to 10 years)
  - Deep in/out-of-money regions (“AI wings”)
  - Negative rate free-boundary regime

- Hagan’s asymptotic approximation fails to capture these wings.

---

## 🧾 Key Takeaways

1. **Difference-learning (residual mapping)** greatly improves data efficiency.  
2. **Stable and accurate** across both normal and free-boundary SABR regimes.  
3. **Reduces training cost** by an order of magnitude compared to direct ANN calibration.  
4. **Eliminates wing arbitrage** and numerical instability seen in asymptotic formulas.  

---

**Reference Implementation Notes:**
> Section 5 of the paper provides the full training configuration and result tables.  
> Numerical tests were run on standard GPUs (unspecified) with Python/TensorFlow implementation.

---

**Citation:**
> Funahashi, H. (2022). *SABR Equipped with AI Wings.*  
> Faculty of Economics, Kanagawa University. JSPS KAKENHI Grant JP22K13436.

