# AI-SAR-CT

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)

Collection of deep learning-based scatter artifact reduction articles for CT/CBCT Imaging.

## 01. Real-time scatter estimation for medical CT using the deep scatter estimation: Method and robustness analysis with respect to different anatomies, dose levels, tube voltages, and data truncation <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> 
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
--------
<br/>
<br/>

## 02. Deep learning architecture for scatter estimation in cone-beam computed tomography head imaging with varying field-of-measurement settings <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised">
H. Agrawal et al. *Journal of Medical Imaging*, 2024. [[doi](https://doi.org/10.1117/1.JMI.11.5.053501)][[paper](https://research.aalto.fi/files/165367277/053501_1.pdf)]
### Summary
**Key Idea**:

A deep learning-based scatter estimation method for CBCT head imaging, addressing the challenge of varying field-of-measurement (FOM) settings. The key idea is to provide the information of the FOM size to the encoder of the U-Net network, allowing the model to learn the scatter characteristics specific to different FOMs.
**Methodology**:
- Simulated training data from head CT scans with varying FOM sizes (18 sizes in training and 30 sizes in testing). Images were downsample to 320x256 and then upsampled post-prediction. A total of 172,800 training samples, 43,200 samples, and 600,000 testing samples were generated. Real data from water phantoms and clinical head scans were also used for evaluation.
- The study uses a U-Net architecture with modifications to incorporate FOM information as additional input channels.
- The method was plugged into a U-Net, DSE-Net, and Spline-Net.
- A loss function combining mean squared error (MSE) and high-frequency loss was proposed to train the model.

**Results**:
- The simulation study demonstrates that the method reduced average MAPE for U-Net by 38%, Spline-Net by 40%, and DSE-
net by 33% for the scatter estimation in the 2D projection domain.
- The root-mean-square error (RMSE) on the 3D reconstructed volumes was improved for U-Net by 43%, Spline-Net by 30%, and DSE-Net by 23%.
- The method improved contrast and image quality on real datasets such as water phantom and clinical data. Although, the improvement was not as significant as in the simulation study.
--------

