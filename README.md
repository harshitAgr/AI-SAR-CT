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
<br/>
<br/>

## 07. Projection-domain scatter correction for cone beam computed tomography using a residual convolutional neural network <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Projection--domain-yellow.svg" alt="Projection-domain">
Y. Nomura et al. *Medical Physics*, 2019. [[doi](https://doi.org/10.1002/mp.13583)][[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6684491/)]
### Summary

**Key Idea**:

A residual convolutional neural network based on U-Net is trained to predict scatter distributions in the projection domain for CBCT. The network is trained exclusively on non-anthropomorphic digital phantoms using Monte Carlo simulations and generalizes to anthropomorphic anatomy at inference. A transfer learning strategy enables adaptation from full-fan to half-fan scan geometries with minimal additional data.

**Methodology**:

- A 25-layer U-Net CNN with skip connections, batch normalization, and ReLU activations. Input and output are 372×372 pixel projections. The network predicts the scatter signal directly, which is subtracted from the measured projection.
- Training data consisted of 1,800 projection pairs from five non-anthropomorphic digital phantoms simulated with GATE/GEANT4 Monte Carlo ($6.25 \times 10^8$ photons per projection). Data augmentation (random 90° rotations and flips) expanded the set to 14,400 samples.
- Mean absolute error (MAE) loss outperformed mean squared error (MSE) loss, particularly in low-intensity regions.
- Optimized with Adam ($\alpha = 0.001$, weight decay $10^{-4}$) for 100 epochs (~10 hours on an NVIDIA GTX 1070).
- Transfer learning for half-fan geometry fine-tuned only the last two convolutional layers using 360 additional pairs at a reduced learning rate ($\alpha = 10^{-5}$) for 15 epochs.
- Compared against the fast adaptive scatter kernel superposition (fASKS) method.

**Results**:

- On a full-fan anthropomorphic head phantom, the CNN (MAE loss) achieved MAE of 17.9 $\pm$ 5.7 HU, PSNR of 37.2 $\pm$ 2.6 dB, and SSIM of 0.9997, outperforming fASKS (MAE 21.8 HU, PSNR 35.6 dB, SSIM 0.9995) with $p < 10^{-4}$ on all metrics.
- Median HU error was $-$1.60 HU for the CNN versus $-$13.1 HU for fASKS, indicating reduced cupping artifacts.
- Transfer learning for half-fan lung phantom scans achieved MAE of 29.0 $\pm$ 2.5 HU and PSNR of 31.7 $\pm$ 0.8 dB, improving over both fASKS and the non-fine-tuned CNN ($p < 10^{-6}$).
- Inference took ~13 ms per projection (~5 seconds for 360 projections), compared to ~5.5 minutes for fASKS.
- Evaluation was limited to Monte Carlo simulated data; no real experimental CBCT acquisitions were tested.
--------

<br/>
<br/>

## 08. Geometry adaptive projection-domain deep scatter estimation for multi-source semi-stationary cone-beam computed tomography <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Projection--domain-yellow.svg" alt="Projection-domain">
T. McSkimming et al. *Medical Physics*, 2026. [[doi](https://doi.org/10.1002/mp.70231)]
### Summary

**Key Idea**:

Adaptive deep scatter estimation (ADSE) is a projection-domain scatter estimator for multi-source, curved-panel semi-stationary CBCT (sCBCT) geometries, where lateral truncation, angular under-sampling, and large geometric variation between projection poses limit conventional projection-domain and volume-domain estimators. Scatter-contaminated sCBCT projections are warped into a view-invariant surrogate CBCT geometry, where a CNN-based scatter estimator and a scatter fluence weighting operator are applied iteratively. The final sCBCT scatter estimate is obtained by inverse fluence-weighting and re-transformation into the sCBCT geometry.

**Methodology**:

- Projections from the sCBCT geometry are transformed into a view-invariant surrogate CBCT geometry, so a single projection-domain network can be applied across source elements.
- A projection-domain CNN scatter estimator and a scatter fluence weighting operator are applied iteratively, so the output converges toward the scatter estimate in the surrogate geometry.
- Evaluated against high-fidelity Monte Carlo (MC) simulations on in-silico head phantoms derived from high-quality human head CT scans.
- Compared to a geometry-aware DSE trained directly on sCBCT data (gDSE), naive projection-domain scatter estimation, non-iterative adaptive estimation with a single fluence-weighting, and iterative MC (iMC) scatter estimation.
- Projection-domain metrics: pixel percentage error versus projection angle and global MAPE. Image-domain metrics: voxelwise error against a scatter-free ground truth.
- Experimental validation used a physical anthropomorphic head phantom on an sCBCT test bench with a curved-panel detector. Metrics were residual cupping, CT number non-uniformity, and contrast and contrast-to-noise ratio (CNR) recovery for thirteen embedded spherical inserts (2–12 mm; nominal contrast −329 to 871 HU), with an MDCT ground truth.

**Results**:

- In silico, non-truncated projections: MAPE of 3.88% for ADSE, compared to 4.42% for iMC and 5.13% for gDSE.
- When truncated projections were included, ADSE MAPE rose to 5.18%, while iMC (4.32%) and gDSE (5.26%) stayed about the same.
- On the physical phantom, uncorrected reconstructions showed 58.87% contrast loss and 84.44% CNR loss relative to MDCT.
- ADSE recovered 48.67% of contrast and 25.03% of CNR, compared to 45.87% and 16.91% for iMC, and 40.45% and 21.44% for gDSE.
- Cupping magnitude and CT number non-uniformity decreased by 79% and 71% with ADSE, compared to 85% and 53% for iMC (gDSE: 114% and 59%).
- ADSE accuracy decreased in the presence of truncated projections, and it recovered only part of the contrast and CNR lost to scatter in the experimental data.
--------

<br/>
<br/>

## 09. A dual-domain network with division residual connection and feature fusion for CBCT scatter correction <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Dual--domain-brightgreen.svg" alt="Dual-domain">
S. Yang et al. *Physics in Medicine & Biology*, 2025. [[doi](https://doi.org/10.1088/1361-6560/adaf06)]
### Summary

**Key Idea**:

A dual-domain network for CBCT scatter correction that reduces scatter artifacts while retaining structural detail. A projection-domain sub-network uses a division residual connection to amplify the difference between scatter and imaging signals, which eases learning of the scatter signal. An image-domain sub-network with dual encoders and a single decoder then fuses features from the scatter-contaminated reconstruction and the reconstruction from the pre-processed projections.

**Methodology**:

- Projection-domain sub-network: a division residual network that amplifies the difference between scatter signals and imaging signals to facilitate scatter learning.
- Image-domain sub-network: two encoders extract features in parallel from two inputs, and a single decoder fuses the features and maps them to the final image.
- The two image-domain inputs are the scatter-contaminated image analytically reconstructed from the raw projections, and the image reconstructed from the pre-processed projections produced by the projection-domain sub-network.
- Evaluated on synthetic and real data and compared against U-Net, DSE-Net, deep residual CNN (DRCNN), and a collimator-based method.

**Results**:

- On synthetic data, mean absolute error was reduced by 74% and peak signal-to-noise ratio increased by 57% relative to the scatter-contaminated images.
- On real data, contrast-to-noise ratio increased by 38% relative to the scatter-contaminated image.
- The method outperformed U-Net, DSE-Net, DRCNN, and the collimator-based method; the abstract does not give per-method numbers.
--------

<br/>
<br/>

## 10. Ultrafast Deep Learning-Based Scatter Estimation in Cone-Beam Computed Tomography <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Projection--domain-yellow.svg" alt="Projection-domain">
H. Agrawal et al. *arXiv preprint*, 2025. [[doi](https://doi.org/10.48550/arXiv.2509.08973)][[paper](https://arxiv.org/pdf/2509.08973)]
### Summary

**Key Idea**:

A study of how input resolution affects deep learning-based CBCT scatter estimation, aimed at deployment on mobile CBCT systems and edge devices. Because scatter is a low-frequency signal, the network can run at a much lower resolution than the projection, with the estimate upsampled afterwards. Reducing input size and network depth lowers FLOPs, inference time, and GPU memory while keeping MAPE and MSE comparable to the full-resolution baseline.

**Methodology**:

- Reconstruction error from down-up sampling of the scatter signal was compared at six resolutions (320×256, 160×128, 80×64, 40×32, 20×16, 10×8) using four interpolation methods (nearest-neighbor, area, bilinear, bicubic). Bicubic gave the lowest error.
- The baseline is Aux-Net, a U-Net with auxiliary field-of-measurement (FOM) channels (7.3 × 10⁶ parameters at 320×256). The number of downsampling blocks was reduced from 5 to 3–4 for smaller inputs, giving 1.8 × 10⁶ parameters at 40×32 and 0.5 × 10⁶ at 20×16.
- Simulated data: Monte Carlo projections of head CT scans from the HNSCC-3DCT-RT dataset and three anthropomorphic phantom CT scans, on a Planmeca Viso G7 geometry (210° arc, 0.278 mm pixels). Training used 15 scans × 100 projections × 18 FOM sizes (270,000 projections, 2,500 photons per pixel); testing used 6 scans × 500 projections × 30 unseen FOM sizes (90,000 projections, 25,000 photons per pixel).
- Real data: water jar (18 cm), water bottle (6.5 cm), and SedentexCT IQ phantom scanned on the Viso G7 (100 kV, 110 mAs) at FOMs of 170×170 mm and 130×30 mm.
- Training: MSE loss, Adam, batch size 64, learning rate decayed logarithmically from 10⁻⁴ to 10⁻⁵ over 30 epochs, 5-fold cross-validation, NVIDIA RTX A4500.

**Results**:

- Compared to the 320×256 baseline (4.71 GFLOPs, MAPE 4.42 ± 0.18%, MSE 2.01 ± 0.14 × 10⁻², 90 ms, 3,890 MB), the 40×32 network used 0.06 GFLOPs (78× fewer), MAPE 3.85 ± 0.10%, MSE 1.34 ± 0.09 × 10⁻², 5.6 ms (16× faster), and 310 MB (12× less memory).
- The 160×128 network reached MAPE 3.84 ± 0.14% at 1.18 GFLOPs and 35 ms. The 20×16 network reached MAPE 4.35 ± 0.23% at 0.01 GFLOPs and 3 ms.
- In simulated reconstructions, RMSE was 8.85 ± 2.92 HU for 160×128 and 8.96 ± 2.90 HU for 40×32.
- Scatter-corrected reconstructions of the real water and SedentexCT phantom scans were reported to be robust.
- The training data contained no small objects, which led to over-correction for small phantoms. Only 2D downsampling was studied (angular downsampling is left to future work), and real-data validation was limited to phantom scans.
--------

