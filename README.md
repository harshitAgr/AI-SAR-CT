# AI-SAR-CT

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)

Collection of deep learning-based scatter artifact reduction articles for CT/CBCT Imaging.

**Real-time scatter estimation for medical CT using the deep scatter estimation: Method and robustness analysis with respect to different anatomies, dose levels, tube voltages, and data truncation** <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> \
J. Maier et al. *Medical Physics*, 2018. [[doi](https://doi.org/10.1002/mp.13274)]
### Summary

**Key Idea**:

DSE leverages a deep convolutional neural network (CNN), specifically a modified U-Net architecture, to approximate MC scatter estimates from CT projection data. The network predicts scatter maps from input projections represented in log, normalized, or combined ("pep") forms.

**Methodology**:

- Simulated CBCT projections from head, thorax, and abdomen CT scans, incorporating varied tube voltages (80–140 kV), noise levels, and truncation scenarios. Images were downsampled to 384×256, then upsampled post-prediction.
- A U-Net-like encoder-decoder CNN with skip connections, trained using mean absolute percentage error (MAPE) loss.
- Target outputs were scatter maps derived from MC simulations. Three input functions were evaluated:
    - **Mep:** Normalized intensities ($e^{-p}$)
    - **Mp:** Log-transformed projection data ($p$)
    - **Mpep:** Product of projection and normalized intensity ($p \cdot e^{-p}$)

**Results**:

- On simulated data, DSE achieves less than 2% error compared to MC scatter in most scenarios, substantially outperforming KSE (11–21%) and HSE (6–293%).
- Robust across a range of tube voltages and noise levels.
- Generalizes effectively to different anatomies included in the training set.
- Enables real-time inference (~10 ms/projection), supporting practical clinical application.
- The Mp and Mpep inputs yielded better performance than Mep, with Mep being slightly less effective.
- On real data from a slit scan, DSE performed better than KSE and HSE, with error of 6 HU compared to 123 HU (KSE), and 65 HU (HSE).

