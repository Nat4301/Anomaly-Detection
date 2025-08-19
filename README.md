# Anomaly Detection from Stochastic Differential Equations  

## INTRODUCTION  
This project develops an anomaly detection pipeline for synthetic audio, motivated by the rise of generative AI and its potential misuse. We treat detection as a time-series anomaly problem, combining:  
- **SDE-based modeling** to capture temporal dynamics  
- **Spectral features** (Mel spectrograms, MFCCs, etc.)  
- **Statistical + ML classifiers** for interpretability and robustness  

**Goal:** detect fake audio by analyzing stochastic deviations that traditional spectral-only methods may miss.  

---

## STOCHASTIC DIFFERENTIAL EQUATIONS  
SDEs describe processes influenced by **deterministic drift** and **stochastic noise**:  

Each **SDE realization** is a simulated trajectory, solved numerically (e.g., Euler–Maruyama). These dynamics help characterize normal vs. abnormal audio signals.  

---

## PROCESSING  
Pipeline:  
1. **Waveform → Mel spectrogram** (time–frequency representation)  
2. **Feature extraction**: statistical descriptors (range, smoothness, increments)  
3. **SDE fitting**: drift + noise residuals per band  
4. **Global audio features**: MFCCs, centroid, bandwidth, roll-off, zero-crossings  

This produces a **high-dimensional feature set** mixing deterministic + stochastic information.  

---

### MODELS  
Two detection approaches:  

**Rule-based (statistical):**  
- Compute feature Z-scores vs. real-audio baseline  
- If enough features exceed thresholds → classify as fake  
- Interpretable & tunable  

**Random Forest (ML):**  
- Ensemble trees classify real vs. fake audio  
- Handles high-dimensional data  
- Provides feature importance & SHAP analysis for transparency  

---

## CONCLUSION  
We propose a hybrid framework for synthetic audio detection:  
- SDE residual analysis for mathematical rigor  
- Spectral + statistical features for signal richness  
- Rule-based + ML classifiers for both interpretability and performance  

Results show both models successfully distinguish real from fake audio. SHAP improves transparency, making the system suitable for misinformation prevention, biometric security, and media integrity.  

---
