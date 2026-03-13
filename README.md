# AI-SAR-CT

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)

Collection of deep learning-based scatter artifact reduction articles for CT/CBCT Imaging.

## 01. Real-time scatter estimation for medical CT using the deep scatter estimation: Method and robustness analysis with respect to different anatomies, dose levels, tube voltages, and data truncation <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Projection--domain-yellow.svg" alt="Projection-domain">
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

## 02. Deep learning architecture for scatter estimation in cone-beam computed tomography head imaging with varying field-of-measurement settings <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Projection--domain-yellow.svg" alt="Projection-domain">
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
<br/>
<br/>

## 03. Image-based scatter correction for cone-beam CT using flip swin transformer U-shape network <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Image--domain-red.svg" alt="Image-domain">
X. Zhang et al. *Medical Physics*, 2023. [[doi](https://doi.org/10.1002/mp.16277)]
### Summary

**Key Idea**:

FSTUNet combines the strengths of CNNs for local texture detail extraction with Swin Transformer for global correlation modeling to perform image-domain scatter correction in CBCT. The key architectural novelty is a Flip Swin Transformer Block that replaces the standard tandem Swin Transformer structure to achieve more powerful inter-window association extraction.

**Methodology**:

- Training data generated from Monte Carlo (MC) simulations of CBCT scatter distributions, as well as a frequency-split dataset generated by a validated method. The model operates in the image domain, taking scatter-contaminated CBCT images as input and outputting scatter-corrected images.
- A U-shaped encoder-decoder architecture where shallow features are extracted by CNN blocks (capturing texture detail) and deep features by Swin Transformer blocks (capturing global context). Skip connections bridge corresponding encoder-decoder levels.
- The Flip Swin Transformer Block modifies the original Swin Transformer's shifted window scheme to enhance cross-window information exchange, improving the network's ability to model long-range dependencies in scatter distributions.
- Compared against five baseline methods: UNet, DRCNN, DSENet, Pix2pixGAN, and 3DUNet.

**Results**:

- On the MC simulated dataset, FSTUNet reduced RMSE from over 100 HU (uncorrected) to approximately 7 HU, with SSIM and UQI values close to 1.
- FSTUNet outperformed all five comparison methods (UNet, DRCNN, DSENet, Pix2pixGAN, 3DUNet) in both qualitative and quantitative evaluations.
- Demonstrated effectiveness on both MC simulation and frequency-split datasets, showing the potential to improve accuracy of CBCT image-guided radiation therapy.
--------
<br/>
<br/>

## 04. Adaptive scatter kernel deconvolution modeling for cone-beam CT scatter correction via deep reinforcement learning <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Projection--domain-yellow.svg" alt="Projection-domain">
Z. Piao et al. *Medical Physics*, 2024. [[doi](https://doi.org/10.1002/mp.16618)]
### Summary

**Key Idea**:

This paper integrates scatter kernel deconvolution (SKD) with deep reinforcement learning (DRL) for CBCT scatter correction. Conventional SKD methods rely on Monte Carlo simulation for fixed kernel parameter determination. The proposed framework instead uses a deep Q-network (DQN) to optimize scatter kernel parameters adaptively on a per-projection basis.

**Methodology**:

- A scatter kernel model iteratively convolves with raw CBCT projections to estimate the scatter distribution. The kernel is parameterized by amplitude, width, and offset terms that control scatter shape.
- A deep Q-network from the DRL framework is introduced as the intelligent agent that interacts with the scatter kernel environment. The DQN observes the current state (projection data and kernel parameters), selects actions (parameter adjustments), and receives rewards based on scatter estimation accuracy improvements.
- Evaluated on simulated CBCT data for head and pelvis phantoms, as well as experimental CBCT measurement data validated against a hardware-based beam stop array (BSA) algorithm for scatter-free reference projections.
- Compared against conventional SKD and U-Net-based scatter estimation methods.

**Results**:

- In the simulation study, the DRL-SKD method achieved MAPE < 9.72% and PSNR > 23.90 dB, compared to conventional SKD (MAPE $\geq$ 17.92%, PSNR $\leq$ 19.32 dB).
- In the measurement study, the method achieved MAPE < 17.79% and PSNR > 16.34 dB on experimental CBCT data.
- The measurement study showed a larger performance gap relative to simulation, indicating reduced generalization to real-world acquisition conditions.
--------
<br/>
<br/>

## 05. Scatter correction for cone-beam CT via scatter kernel superposition-inspired convolutional neural network <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Projection--domain-yellow.svg" alt="Projection-domain">
X. Zhuo et al. *Physics in Medicine & Biology*, 2023. [[doi](https://doi.org/10.1088/1361-6560/acbe8f)]
### Summary

**Key Idea**:

This paper combines the physics-based scatter kernel superposition (SKS) method with a convolutional neural network. Instead of estimating scatter at individual pixel levels, the CNN learns to predict the amplitude and width maps of Gaussian scatter kernels from projection images, which are then convolved to compute the final scatter field. Embedding the SKS physical model into the network architecture reduces the number of trainable parameters compared to purely data-driven approaches like Deep Scatter Estimation.

**Methodology**:

- Monte Carlo (MC) simulation was used to generate training data from a modeled CBCT system imaging a human chest phantom. Pairs of scattered and scatter-free projection images were obtained at different dose levels.
- The CNN predicts two parameter maps — scatter kernel amplitude and width — rather than directly predicting the scatter signal. These maps are fed into a differentiable SKS layer that computes the scatter distribution via Gaussian kernel convolution.
- The physics-inspired architecture constrains the output space, resulting in a more compact model with fewer parameters than conventional end-to-end scatter estimation networks.
- Compared against conventional iterative MC-based SKS method and other deep learning approaches including Deep Scatter Estimation (DSE).

**Results**:

- In the projection domain, the method achieved a 58.5% reduction in RMSE, 18.1% increase in PSNR, and 3.4% increase in SSIM compared to the MC-based iterative SKS method on average.
- Produced lower errors than both the conventional SKS method and other deep learning-based methods on simulated projections and reconstructed CT volumes.
- Evaluation was limited to a single chest phantom anatomy; generalization to other body regions was not assessed.
--------
<br/>
<br/>

## 06. Task-based transferable deep-learning scatter correction in cone beam computed tomography: a simulation study <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Projection--domain-yellow.svg" alt="Projection-domain">
J. P. Cruz-Bastida et al. *Journal of Medical Imaging*, 2024. [[doi](https://doi.org/10.1117/1.JMI.11.2.024006)][[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10960584/)][[code](https://github.com/calcutech/task-based-DSC)]
### Summary

**Key Idea**:

This paper proposes a two-stage transfer learning strategy for CBCT scatter correction. A U-Net CNN is first pre-trained on a large dataset of simple geometric phantom projections to learn general scatter patterns, then fine-tuned with a small task-specific dataset of anthropomorphic phantom projections. This reduces both the data requirements and training time needed to adapt the scatter correction model to new imaging tasks.

**Methodology**:

- A U-Net architecture with seven downsampling stages predicts 2D scatter ratio maps ($SR = S/T$) from log-normalized projection data at 512×256 pixel resolution (0.784 mm pixel size).
- Pre-training used 5,184 image pairs from 24 cylindrical phantom configurations (varied diameter and length). Transfer learning required only 250 training pairs per imaging task (~12× less data than pre-training).
- Loss function: $L = 1 - \text{SSIM}$, optimized with Adam (learning rate $1 \times 10^{-4}$). Pre-training ran for 150 epochs (~6 hours on an NVIDIA RTX 3060); transfer learning converged in ~50 epochs (~5 minutes, ~70× faster).
- Monte Carlo simulations were generated using fastCAT software to produce ground truth scatter distributions.
- Evaluated on simulated adult head and pediatric pelvis imaging tasks.

**Results**:

- Pre-trained CNN achieved SSIM $\geq$ 0.91 for scatter ratio predictions across all test cases, with the most frequent SSIM range being (0.99, 1.0).
- Task-specific CNNs after transfer learning achieved SSIM $\geq$ 0.93 for both adult head and pediatric pelvis tasks.
- Reconstructed CBCT images achieved CT number accuracy within $\pm$25 HU ($\pm$2.5%).
- Pediatric pelvis reconstructions showed lower SSIM than adult head, attributed to greater anatomical complexity.
- Validation was limited to simulation data only; no real experimental CBCT acquisitions were tested. No direct comparison with existing scatter correction methods was provided.
--------

