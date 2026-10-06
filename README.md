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

## 08. Ultrafast Deep Learning-Based Scatter Estimation in Cone-Beam Computed Tomography <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Projection--domain-yellow.svg" alt="Projection-domain">
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

<br/>
<br/>

## 09. Spectral deep learning-based patient and bowtie scatter correction for clinical photon-counting CT <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Projection--domain-yellow.svg" alt="Projection-domain">
L. Hennemann et al. *Medical Physics*, 2026. [[doi](https://doi.org/10.1002/mp.70442)][[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13125421/)]
### Summary

**Key Idea**:

This paper extends deep scatter estimation (DSE) to clinical photon-counting CT (PCCT) in two ways. First, a single network estimates bowtie-filter scatter and patient scatter jointly, where earlier DL methods only addressed patient scatter. Second, the network takes several energy thresholds as input and estimates scatter for all of them at once ("spectral DSE"), using the fact that each threshold is affected differently by scatter.

**Methodology**:

- Monte Carlo data were generated with MOCASSIM, matched to a Siemens NAEOTOM Alpha.Peak scanner: 1376 × 144 detector pixels, a 2D anti-scatter grid (ASG) covering 2 × 3 pixels in standard mode, 140 kV, $10^9$ photons per projection, and four energy thresholds at 20, 55, 70, and 90 keV.
- Training set: elliptical and cylindrical water phantoms (20–40 cm, 15 each) and 60 FORBILD thorax phantoms with random scaling (0.7–1.3) and shifts (±8 cm), giving 7,560 scatter pairs per threshold, split 80:20 for training and validation. Testing used 45 unseen FORBILD head phantoms and XCAT average and obese phantoms.
- Because the coarse ASG makes scatter high-frequency, each projection is split into six sub-signals (one per pixel position under the ASG lamellae). The network processes them as six input channels and the six output channels are merged back into the full detector signal.
- Joint patient and bowtie estimation was compared against two separately trained networks. The spectral variants are DSE1→2, DSE1→4, DSE2→2, and DSE4→4 (n thresholds in → m out), compared against single-threshold DSE and a kernel-based scatter estimation (KSE) reference.
- Loss: scatter-to-primary-weighted MAPE (SPMAPE), $\frac{100\%}{N}\sum \frac{|I_{scatter,\chi} - I_{scatter,MC}|}{I_{pri}}$, which weights errors by the scatter-to-primary ratio.
- Training: TensorFlow 2.10 on an NVIDIA V100, Adam, initial learning rate $10^{-4}$ (halved every 25 epochs without improvement), batch size 32, 500 epochs with early stopping, about 6 h for single-threshold and up to 11 h for multi-threshold models.
- Real data: head and thorax anthropomorphic phantoms scanned on a NAEOTOM Alpha.Peak (140 kV, 200 mA, 1008 projections, four thresholds in research mode), with a 2.4 mm slit scan as the low-scatter reference.

**Results**:

- Joint and separate patient and bowtie estimation both reduced the global MAE from about 8 HU to about 1 HU on simulated test data (8.3 HU to 1.3 HU for separate DSE). The joint network was marginally worse than two separate networks.
- All DSE variants outperformed the convolution-based reference. DSE4→4 gave the best overall results across all thresholds, though differences from non-spectral DSE were small per threshold.
- Spectral DSE reduced scatter errors from the patient and bowtie from up to 8 HU to below 1 HU. For voxels with uncorrected errors above 10 HU (about 25% of the volume), the MAE10 fell from 23.8 HU to 1.6 HU.
- In virtual monoenergetic images, MAE fell from about 16 HU to about 2 HU at 45 keV, from about 8 HU to about 1 HU at 70 keV, and from 5 HU to under 1 HU at 100 keV.
- On measured phantoms at 45 keV VMI, the deviation from the slit scan was 39.2 HU (uncorrected), 8.1 HU (DSE), and 5.8 HU (DSE4→4) for the head, and 40.4 HU, 13.4 HU, and 8.6 HU for the thorax.
- Inference took about 1.8 ms per projection.
- Limitations: only the standard bowtie filter was studied (cardiac or pediatric filters were not), and only forward scatter was handled (cross-scatter in dual-source systems is left for future work). Only standard-resolution mode was evaluated, not ultra-high-resolution mode.
--------

<br/>
<br/>

## 10. ScatterNet for projection-based 4D cone-beam computed tomography intensity correction of lung cancer patients <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Projection--domain-yellow.svg" alt="Projection-domain">
H. Schmitz et al. *Physics and Imaging in Radiation Oncology*, 2023. [[doi](https://doi.org/10.1016/j.phro.2023.100482)][[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10480315/)]
### Summary

**Key Idea**:

This retrospective study applies a U-shaped CNN (ScatterNet) to projection-based scatter correction of 4D CBCT in lung cancer patients and evaluates it through proton dose calculation. The network is trained on pairs of raw and corrected projections, where the corrected projections come from a slower conventional 4D projection-based scatter correction workflow (CBCTcor). The goal is to replace that workflow's deformable image registration and filtering steps with a single network pass of a few seconds.

**Methodology**:

- Data from 26 lung cancer patients treated with photon therapy: a free-breathing planning CT, a 4DCT, and CBCT projections from one treatment fraction (Elekta Synergy or VersaHD, 120 kVp, shifted detector, more than 600 projections).
- Training pairs: 17,564 raw and corrected 2D projection pairs, with the corrected projections generated by the 4D CBCTcor workflow (forward projection of a virtual CT, followed by smoothing of the scatter estimate). Patients were split 60/20/20 into 15 training, 6 validation, and 5 test patients.
- ScatterNet is a 2D U-Net with a single input layer and 8 channels, followed by resolution levels with 8, 16, 32, 64, 128, and 256 channels, to model low-frequency differences. Projections were zero-padded from 504 × 504 to 512 × 512, and mixup augmentation was applied. Training used 128 superior-inferior rows around the panel centre, since the corrected projections have a smaller field of view than the raw ones.
- Corrected projections were reconstructed with MA-ROOSTER into 4DCBCT$_{SN}$ and compared against 4DCBCT$_{cor}$ and a deformable-registration-based 4DvCT in image quality (ME and MAE in HU) and proton dose (intensity-modulated proton therapy plans in RayStation, ITV D98%, and 2%/2 mm and 3%/3 mm gamma analysis).

**Results**:

- Training stopped after 126 iterations (iteration 115 used for the final model) and took about 12 h. Correcting a full projection set (up to 732 projections) took 3.9 s on average (5.7 ms per projection).
- Scatter correction time per patient decreased from about 10 min (4DCBCT$_{cor}$) and about 30 min (4DvCT) to 3.9 s. This excludes reconstruction (about 10 min) and dose calculation (about 4 min).
- Averaged over ten breathing phases, MAE was 87 HU for 4DCBCT$_{SN}$ vs. 4DCBCT$_{cor}$ and 102 HU vs. 4DvCT. The corresponding mean errors were 4 HU and 23 HU.
- Median ITV D98% differences were below 0.4 Gy across the five test patients (up to 1.3 Gy for the largest per-patient case).
- Median 3%/3 mm gamma pass rates were 96% for both 4DCBCT$_{SN}$ vs. 4DCBCT$_{cor}$ and vs. 4DvCT (above 90% for all test patients), and median 2%/2 mm pass rates were 88–90%.
- Larger dose deviations appeared in lung tissue than in the target. The MAE was higher than that reported for 3D pelvic CBCT with the same network (87 HU vs. 46 HU), which the authors attribute to the poorer image quality of 4D CBCT.
- Limitations: only 5 test patients and a single acquisition site, with stability across machines and centres not tested. Only target dose was analysed (organ-at-risk dose was not), and the network depends on the conventional 4D workflow for its training labels.
--------

<br/>
<br/>

## 11. Deep learning for x-ray scatter correction in dedicated breast CT <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Projection--domain-yellow.svg" alt="Projection-domain">
J. J. Pautasso et al. *Medical Physics*, 2022. [[doi](https://doi.org/10.1002/mp.16185)][[paper](https://ris.utwente.nl/ws/files/295576406/Medical_Physics_2022_Pautasso_Deep_learning_for_x_ray_scatter_correction_in_dedicated_breast_CT.pdf)]
### Summary

**Key Idea**:

A U-Net estimates the scatter signal of a single dedicated breast CT (bCT) projection, trained on Monte Carlo (MC) primary and scatter projections of patient-based breast phantoms. Because the network sees only one 2D projection, two extra inputs carry 3D information: the breast thickness map along each ray and the horizontal position of the breast center of mass. The estimated scatter is subtracted from the measured projection before reconstruction, so no additional hardware, beam blocker or second scan is needed.

**Methodology**:

- Data: 115 patient-based phantoms (skin, adipose, fibroglandular tissue) segmented from scans of a clinical Koning bCT system (49 kV, 1.6 mm Al). 110 were used for training and 5 for internal validation. Phantoms were translated randomly by up to 40 mm to give a fourfold augmentation (440 training and 20 validation samples). An external test set of 10 phantoms came from a different bCT system (Doheny, 60 kV, 0.2 mm Cu).
- Simulation: Geant4-based MC with $2 \times 10^8$ tracked x-rays per simulation. 12 views (0 to 360 degrees, 30 degree step) per phantom gave 5460 primary and scatter projections of $128 \times 80$ pixels. Total projection = primary + scatter, normalized to the 95th percentile of intensity.
- Network: 2D U-Net (64 filters, $3 \times 3$ kernels, batch norm, ReLU, sigmoid output) trained projection by projection with the total projection as input and the MC scatter as label. The thickness map (normalized as $\exp(-\text{thickness}/100)$, passed through an extra downsampling block) and the breast location are concatenated at the bottleneck.
- Loss: pixelwise MSE, weighted ten times higher inside the breast than in the open field. Adam, batch size 8, learning rate $10^{-4}$, 2000 epochs.
- Evaluation: mean relative difference (MRD) and mean absolute error (MAE) of the scatter inside the breast, an ablation of the two extra inputs, a full MC-simulated 300-projection scan (1.2 degree step) reconstructed with ML-TR and scored by SSIM and MAE on 100 slices, and three patient bCT scans (projections resized to $128 \times 80$ for inference, then resized back) assessed by cupping profiles, local contrast and CNR.

**Results**:

- Projection domain: MRD / MAE of 0.04% / 2.94% on the internal validation set (one validation sample with thickness outside the 27-108 mm training range was excluded) and -0.64% / 2.84% on the external test set. Training set: 0.1% / 3.1%.
- Extra-input ablation (test set MAE): 3.00% with no extra inputs, 3.86% with thickness only, 2.89% with location only, 2.84% with both. Validation MAE: 3.19%, 3.28%, 3.03% and 2.94%, respectively.
- Error showed no clear dependence on breast thickness, density or position (test-set Pearson correlations of MRD: -0.217 with location, -0.190 with density). One test phantom with average thickness of 90 mm or more had a higher MAE, still below 5%.
- MC-simulated reconstruction: SSIM 0.99 and MAE 0.11% (range 0% to 0.35%), with a single outlier slice at 2.06%.
- Patient scans (n = 3): cupping was reduced, voxel values moved to the literature attenuation ranges for adipose and fibroglandular tissue, and local contrast increased by 25%, 30% and 20% (mean 25%). Mean CNR increased by 0.32, which was not significant (95% CI [-0.01, 0.65], p = 0.059).
- Speed: 0.2 s per projection on an NVIDIA GTX 1080; 58 s for 300 projections of $1024 \times 640$ pixels including the full pipeline.
- Limitations: accuracy may degrade for very large breasts (above about the 90th percentile of thickness); the model is trained for a single acquisition setting and must be retrained if imaging conditions change; only three patient scans were evaluated, without clinical task testing (e.g. microcalcification detection) or observer studies.
--------

<br/>
<br/>

## 12. X-Ray Scatter Estimation Using Deep Splines <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Projection--domain-yellow.svg" alt="Projection-domain">
P. Roser et al. *IEEE Transactions on Medical Imaging*, 2021. [[doi](https://doi.org/10.1109/TMI.2021.3074712)][[paper](https://arxiv.org/pdf/2101.09177)]
### Summary

**Key Idea**:

X-ray scatter in diagnostic CBCT is mostly low-frequency, so the scatter image is modeled as a cubic bivariate B-spline whose coefficients are predicted by a lean convolutional encoder plus a bottleneck network. The B-spline evaluation is written as matrix multiplications and embedded as a known operator in the computational graph, so training stays end-to-end differentiable while the output is restricted to smooth functions. The aim is to avoid the spurious high-frequency content that an unconstrained U-net can produce, while using fewer parameters and less runtime.

**Methodology**:

- Scatter model: $I = I_p + I_s$, with $\tilde{I}_{s,4} = U_4 \cdot C \cdot V_4^T$, where $C \in \mathbb{R}^{w_c \times h_c}$ are the spline coefficients and $U_4$, $V_4$ are pre-computed evaluation matrices for a uniform knot grid with cubic B-splines ($k=3$) and endpoint interpolation. The derivative $\partial \tilde{I}_{s,4}/\partial C = V_4 \otimes U_4$ is used for back-propagation.
- Network: an encoder of $d$ convolutional blocks (two 3x3 convolutions with $c$ channels and ReLU per block, 2x2 average pooling between blocks, optional pre-pooling $p$, and a final 1x1 convolution), followed by a bottleneck network that maps the latent variables to spline coefficients. Four bottleneck variants were compared: a constrained positive weighting matrix, an unconstrained fully-connected layer with ReLU, two fully-connected layers, and two convolutional blocks plus a fully-connected layer.
- Baselines: a deep U-net (DU-net) and a shallow U-net (SU-net, feature maps not doubled at each level), following the Deep Scatter Estimation approach.
- Synthetic data: MC-GPU Monte Carlo simulation on 20 head scans (HNSCC-3DCT-RT) and 15 thorax scans (CT Lymph Nodes) from TCIA, 260 projections per scan over 200 degrees, 1152 x 768 pixels, $5 \times 10^{10}$ photons per projection, 85 kV tungsten spectrum, source-to-isocenter 785 mm and source-to-detector 1300 mm. Projections were Gaussian filtered ($\sigma_p = 2$, $\sigma_s = 30$) and down-sampled to 384 x 256.
- Evaluation: nested cross-validation (4*3-fold for head, 5*4-fold for thorax). Adam optimizer, initial learning rate $10^{-4}$ for meta-parameter search (100 epochs, early stop after 20 epochs without improvement) and $10^{-5}$ for the final runs, Glorot initialization; scatter MAPE and SSIM of reconstructions against the ground truth.
- Real data: 12 C-arm CBCT scans (ARTIS icono floor) of an anthropomorphic thorax phantom (PBU-60), 397 projections of 648 x 472 pixels per short scan, 85 kV, with and without anti-scatter grid and with full or slit collimation. Grid plus slit scanning served as ground truth. The real data were used for testing only.
- Additional analyses: power spectral density and frequency response of the networks, robustness to Poisson noise (photon counts $10^3$ to $10^5$) with networks trained on noise-free data, and CPU runtime (12-core Xeon).

**Results**:

- Parameter search: all networks reached MAPE between 6 % and 9 %, and compact networks outperformed the DU-nets. The constrained weighting matrix was the best bottleneck, and it converged to a block-circulant matrix (similar to a convolution), while the unconstrained fully-connected variants were worse.
- Synthetic head data: scatter MAPE of approximately 5 % across folds and configurations, and reconstruction SSIM above 0.99 for all networks. The spline network gave consistent results across folds, whereas the U-net results varied more.
- Synthetic thorax data: MAPE of approximately 7.5 % for all networks, SSIM just below 0.98 on average, with larger error margins and outliers than for the head data. The spline approach was on par with the U-nets in the quantitative metrics.
- Spectral analysis: the U-nets increased high frequencies in the predicted scatter, whereas the spline network preserved the power spectral density of the ground truth over the whole spectrum. Its frequency response was closer to the ideal one in amplitude and phase, but a noticeable intensity shift was observed.
- Noise: the U-nets were sensitive to unseen noise levels, whereas the spline network was more robust.
- Runtime: 4 ms to 30 ms for the spline network, 34 ms to 50 ms for the SU-net (1.7 to 8.5 times slower) and 89 ms to 144 ms for the DU-net.
- Phantom study (mean absolute HU error against grid plus slit): grid with full field 39.31 HU, slit without grid 58.46 HU, full field without grid 123.84 HU, DU-net 62.86 HU, SU-net 64.52 HU, spline network 63.97 HU. The learned methods performed about as well as slit scanning without a grid, and the anti-scatter grid gave the lowest error.
- Limitations: the spline network output is smooth by construction, whereas the U-nets, especially the shallow one, retain some input detail. The U-net error rates were higher than previously reported, which the authors attribute to different simulation code, a smaller training corpus and data heterogeneity. The simulation assumes an ideal detector, and the synthetic-to-real domain shift was not addressed. The phantom study had no intensity or geometry calibration between scans.
--------

<br/>
<br/>

## 13. Scatter Correction in X-Ray CT by Physics-Inspired Deep Learning <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Dual--domain-brightgreen.svg" alt="Dual-domain">
B. Iskender et al. *IEEE Transactions on Computational Imaging*, 2022. [[doi](https://doi.org/10.1109/TCI.2022.3226300)][[paper](https://arxiv.org/pdf/2103.11509)]
### Summary

**Key Idea**:

PhILSCAT and OV-PhILSCAT estimate scatter in each projection view from two inputs: the scatter-corrupted projection and an initial reconstruction rotated to that view angle. The architecture follows a slice-by-slice scatter blurring model, with 2D convolutions in the detector plane and channel contraction along the beam direction. The training loss expresses an image-domain (filtered backprojection) error norm as a filtered norm on the projections, so no back-propagation through FBP is needed. OV-PhILSCAT additionally uses the fact that the difference of scatter in pi-opposite views can be computed exactly from the measurements, so the network only estimates the smoother average of the two.

**Methodology**:

- Measurement model: $\tau = p + s$ (total = primary + scatter), normalized by the bright-field fluence $I_0$. The initial reconstruction uses $\tilde{g} = -\ln\min\{\bar{\tau}, 1\}$ with FBP. The network estimates $\bar{s}^*_\theta = N_\gamma(\tilde{f}_\theta, -\ln\bar{\tau}_\theta)$, and the primary is $\bar{p}^*_\theta = \max\{\bar{\tau}_\theta - \bar{s}^*_\theta, \epsilon\}$.
- OV-PhILSCAT: for parallel-beam geometry, $\bar{\tau}_\theta - \hat{\bar{\tau}}_{\theta+\pi} = \Delta\bar{s}_\theta$ is known from the data. The network predicts only $\bar{b}_\theta = (\bar{s}_\theta + \hat{\bar{s}}_{\theta+\pi})/2$ and runs on K/2 views. The average $\bar{b}$ is smoother than $\Delta\bar{s}$ (in the 27 training phantoms, the DC component accounts for about 45 % of its energy and the first 11 components for about 75 %). A median filter on $\Delta s$ is proposed for subpixel misalignments.
- Loss: $\Lambda = \sum_\theta \|h * (g_\theta - g^*_\theta)\|_2^2 + \lambda\|g_\theta - g^*_\theta\|_1$ with the two-tap filter $h[n] = 0.5\delta[n+1] - 0.5\delta[n-1]$. The filtered $\ell_2$ term is equivalent to a perceptually weighted reconstruction-domain error, derived using Parseval's relation for the Radon transform. The $\ell_1$ term recovers the zero-frequency component, with $\lambda = 5 \cdot 10^{-2}$.
- Network: a "ladder" of 2D convolutional blocks (ReLU, batch normalization, skip connections) that compresses the channel dimension (the beam direction $u$) by factors of two, with a $d \times d \times (d+1)$ input and a $d \times d$ output (illustrated for $d = 64$).
- Data: Monte Carlo simulations. Parallel beam: GATE/GEANT4, 200 keV monoenergetic source, 128 x 128 detector, 360 views, $8 \times 10^6$ photons per view, random phantoms of prisms, cylinders and spheres in water, aluminium or titanium. Cone beam: MC-GPU, source-to-detector 180 cm, source-to-origin 130 cm, 128 x 128 detector, K = 360 views, 90 keV monoenergetic or a 120 kVp tungsten spectrum with 4.3 mm Al filter, titanium-rod phantoms and 30 anthropomorphic phantoms from the CT Lymph Nodes dataset mapped to five tissue types.
- Training: 27 training and 3 test phantoms per experiment, Pytorch with Adam. The parallel-beam networks were trained for 100 epochs. A noise-suppression pre-processing step was applied to the low-photon parallel-beam data to prevent the networks from learning denoising.
- Baseline: Deep Scatter Estimation (DSE, a U-net operating on the projection), trained on the same data with a relative scatter MAE loss. Metrics are PSNR, SSIM, MAE and peak error against the FBP/FDK reconstruction of the primary.

**Results**:

- Monochromatic parallel beam (3 test phantoms, peak reconstruction density 3407 HU): PSNR 51.1 dB (PhILSCAT) and 51.3 dB (OV-PhILSCAT) against 45.4 dB for DSE and 38.6 dB uncorrected. SSIM 0.998 for both proposed methods against 0.984 (DSE) and 0.964. MAE 3.6 HU against 6.6 HU (DSE) and 16.4 HU. Peak error 514 HU and 510 HU against 1228 HU (DSE) and 1572 HU uncorrected. OV-PhILSCAT needs half the network evaluations.
- Monochromatic CBCT, Ti rods: PSNR 51.6 dB (PhILSCAT) against 49.8 dB (DSE) and 35.8 dB uncorrected, SSIM 0.997 against 0.995, MAE 8.3 HU against 8.8 HU.
- Polychromatic CBCT, Ti rods: PSNR 51.7 dB against 49.9 dB (DSE) and 36.8 dB uncorrected, SSIM 0.997 for both learned methods, MAE 11.9 HU against 13.3 HU. The number of voxels with error above 500 HU was reduced about 8-fold relative to DSE (3,900 voxels). PhILSCAT had a 12 % larger MSE on the scatter estimate than DSE, yet 1.8 dB better reconstruction PSNR.
- Polychromatic CBCT, anthropomorphic phantoms: PSNR 37.2 dB against 36.8 dB (DSE) and 26.9 dB uncorrected, MAE 12.3 HU against 12.9 HU, peak error 800 HU against 970 HU (DSE) and 1699 HU. The gap to DSE is smaller here, which the authors relate to smoother scatter in these phantoms.
- Lower photon count (trained on $I_0$, tested on $I_0/4$, Ti rods): PSNR 45.7 dB (PhILSCAT) against 43.4 dB (DSE), with results deteriorating for both methods.
- Ablations (polychromatic Ti rods, PSNR / peak error): U-net architecture with the proposed loss 49.3 dB / 1630 HU, random image in place of the initial reconstruction 51.1 dB / 1098 HU, standard scatter MSE loss 51.4 dB / 1047 HU, full PhILSCAT 51.7 dB / 954 HU. Removing the initial reconstruction or the tailored loss roughly doubled the number of voxels above 500 HU.
- Limitations: only Monte Carlo simulated data were used (the authors list real CT data as future work), and OV-PhILSCAT applies to parallel-beam geometry. The 3D loss extension is exact only for parallel beam and approximate for small cone angles. The authors note that limited generalization to different objects or scanner settings should be expected, and propose training separate models per protocol. The comparison was against a projection-domain method only. The rotation and FBP steps dominate the reported runtime (22.6 s to 28.2 s per volume, of which the network takes 1.3 s to 2.6 s) in the CPU/FBP implementation.
--------

<br/>
<br/>

## 14. Evaluation of CBCT scatter correction using deep convolutional neural networks for head and neck adaptive proton therapy <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Projection--domain-yellow.svg" alt="Projection-domain">
A. Lalonde et al. *Physics in Medicine & Biology*, 2020. [[doi](https://doi.org/10.1088/1361-6560/ab9fcb)][[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8920050/)]
### Summary

**Key Idea**:

A U-Net is trained on Monte Carlo (MC) simulated head and neck CBCT projections to predict the normalised scatter distribution from raw projections, and the corrected projections are obtained by subtracting the prediction in the intensity domain. The study evaluates this MC-trained, projection-based correction for proton therapy: HU accuracy against MC scatter-free images, proton range in an anthropomorphic head phantom, IMPT dose agreement in simulated patients, and agreement with a prior-based empirical correction in patient CBCT images.

**Methodology**:

- Data: 48 head and neck patients (planning CTs from a 140 kVp GE scanner) split into 29 training, 9 validation and 10 test patients. CBCT projections were simulated with the GPU MC code MCGPU for an Elekta XVI geometry (100 kVp, centered panel, 20 cm collimator, no bowtie filter), scored on a 1024 × 1024 grid and downsampled to 512 × 512 (0.8 × 0.8 mm$^2$ pixels).
- Training/validation: 90 projections per patient over 360° with $6 \times 10^9$ photons each, augmented by horizontal and vertical flips to 360 per patient (13,680 projections in total). Test patients: 540 projections, reconstructed with FDK (RTK) into uncorrected, scatter-free and scatter-corrected volumes.
- Network: 7-level U-Net following Maier et al., with three 3 × 3 convolutions and PReLU per level, stride-2 convolutions for downsampling, bilinear upsampling in the decoder, and channels increasing from 16 to 1024. Input is the raw projection $p_{raw} = -\ln(I_{raw}/I_0)$ downsampled to 256 × 256.
- Output: the normalised scatter $s = S/I_0$ (rather than the scatter-free projection). The corrected projection is $p_{corr} = -\ln(e^{-p_{raw}} - \hat{s}_{NN})$, with $\hat{s}_{NN}$ clipped to 95% of $e^{-p_{raw}}$.
- Training: PyTorch, NVIDIA TITAN Xp, Adam, batch size 4, learning rate $5 \times 10^{-6}$, 150 epochs, Glorot uniform initialisation. Compared losses: MSE and MAPE, and a variant predicting the scatter-free projection $p_{SF}$ directly (MSE only).
- Evaluation: mean error (ME) and mean absolute error (MAE) of reconstructed HU against the scatter-free volume; proton range (R80) in a head phantom with a human skull (395 real projections, CT as reference, 1364 R80 values); 2%/2 mm gamma for IMPT plans (RayStation, RBE 1.1) optimised on scatter-free CBCT and recalculated on corrected CBCT; comparison with the prior-based correction of Niu/Park (deformed planning CT as prior) in 3 real patient CBCTs; a spectrum-mismatch test with 2 mm added aluminium filtration.

**Results**:

- Training took 21 h for the $p_{raw} \to s$ networks and 16.2 h for $p_{raw} \to p_{SF}$. Correction took 13.58 ms per projection, under 5 s for a 360-projection scan.
- HU error (ME, MAE) on the test patients: (−0.801, 13.41) HU for $p_{raw} \to s$ with MAPE loss, (1.73, 15.48) HU with MSE loss, (−3.57, 20.23) HU for $p_{raw} \to p_{SF}$, and (−28.61, 69.64) HU for uncorrected images.
- Head phantom: RMS error of R80 versus CT was 0.73 mm for the corrected CBCT and 16.06 mm for the uncorrected CBCT; the difference on the beam central axis was 1.0 mm.
- Simulated patients: mean 2%/2 mm gamma pass rate was 98.89% for corrected images (range 94.18% to 100%) versus 68.44% for uncorrected images, with the scatter-free CBCT as reference.
- Spectrum mismatch (one test patient): the pass rate fell from 99.56% to 98.56% when 2 mm Al filtration was added to the test spectrum but not the training spectrum, versus 72.15% for the uncorrected volume.
- Real patient CBCTs (3 cases) against the prior-based correction: 2%/2 mm pass rates of 79.41%, 81.92% and 73.13% (average 78.15%) and 3%/3 mm pass rates of 98.24%, 99.12% and 98.79% (average 98.72%).
- Limitations noted by the authors: the lower pass rate on real data may reflect an imperfect XVI model in the simulation and the reference method also correcting low-frequency effects such as cupping; performance depends on the accuracy of the spectrum model; only head and neck was evaluated (pelvis and other sites not tested); the method needs access to raw projections, which commercial systems do not always provide; the truncated anatomy of the centered panel excluded target portions below the shoulders.
--------

<br/>
<br/>

## 15. A Deep Learning-Based Scatter Correction of Simulated X-ray Images <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Projection--domain-yellow.svg" alt="Projection-domain">
H. Lee et al. *Electronics*, 2019. [[doi](https://doi.org/10.3390/electronics8090944)][[paper](https://mdpi-res.com/d_attachment/electronics/electronics-08-00944/article_deploy/electronics-08-00944.pdf)]
### Summary

**Key Idea**:

The paper trains a CNN on Monte Carlo-simulated projection pairs to estimate the scatter component of an X-ray projection, which is then subtracted from the scattered input. The network has two parallel branches, a conventional-convolution branch (Hs-Net) for high-frequency scatter and a dilated-convolution branch (Ls-Net) for low-frequency scatter. Because real paired scatter/scatter-free data are hard to acquire, training and testing both use simulated CBCT projections of a chest, and the method is compared with an MC-based iterative correction.

**Methodology**:

- Simulation: GATE (GPU) Monte Carlo model of a CBCT with SDD 685 mm, SOD 400 mm, 110 kV tube with a 0.2 mm Cu filter, and a 256 x 128 detector (686 x 343 mm). Input volumes are real chest MDCT volumes from the National Biomedical Imaging Archive (NBIA).
- Pair generation: the scattered image has 5000 photons per pixel on average and the scatter-free target 6000 photons per pixel (a 20% higher dose); the scatter-only image is the difference. One pair per degree over 360 degrees.
- Data: nine volumes with augmentation (180 degree rotation, horizontal and vertical flips) give 12,960 training pairs; 360 pairs from a separate volume without augmentation are used for testing. Generating the data took about 18.71 days on three desktop PCs.
- Architecture (DSCNN): Hs-Net is a 12-layer modified DnCNN (3 x 3 x 64 convolutions, BN, ReLU) and Ls-Net is a 12-layer modified IRCNN with dilation rates 1, 2, 4, 8, 16, 32, 32, 16, 8, 4, 2, 1 and a final receptive field of 253 x 253. The predicted scatter $R(y) = R_h(y) + R_l(y)$ is subtracted: $p' = y - R(y)$.
- Loss: $Loss(\Theta) = \alpha \, Loss_H(\Theta_h) + (1-\alpha) \, Loss_L(\Theta_l)$ with $\alpha = 0.5$; each term is an L2 loss against the high-frequency target or the low-frequency target, the latter being the scatter smoothed by a 21 x 21 Gaussian with $\sigma = 3$.
- Training: Adam, batch size 5, 500 epochs (13.3 days on a GTX 1060), learning rate from $10^{-3}$ decreased to $10^{-6}$.
- Baseline: MC-based iterative scatter correction (1000 photons per pixel, five iterations, about five days for 360 images). Metrics: RMSE, PSNR, SSIM against the scatter-free target.

**Results**:

- Projections (mean over 360 test images, MC iterative vs. DSCNN): RMSE 8.43 vs. 3.50, PSNR 42.10 vs. 49.72, SSIM 0.960 vs. 0.992. This corresponds to a 58.5% RMSE reduction and 18.1% and 3.4% increases in PSNR and SSIM.
- Reconstructed central slices: RMSE 122.11 vs. 88.41, PSNR 42.53 vs. 44.21, SSIM 0.918 vs. 0.960 (27.6% RMSE reduction, 4.0% and 4.6% increases in PSNR and SSIM).
- Correction took 17.3 ms per projection on the GPU PC.
- Ablation: Ls-Net alone improved overall contrast but left streak artifacts; Hs-Net alone reduced streaks but gave poor uniformity and artifacts along paths through thick, dense objects. The combination avoided both.
- Limitations: all training and test data are simulated; no real CBCT measurements were used. Only the chest region at low resolution (256 x 128) was examined, the scatter-free target is a dose-increased image (20% higher) rather than a true scatter-free image, and no comparison with other deep learning methods was made. The authors plan to build a real CBCT with the same specifications for validation.
--------

<br/>
<br/>

## 16. Cone Beam Computed Tomography Image Quality Improvement Using a Deep Convolutional Neural Network <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Image--domain-red.svg" alt="Image-domain">
S. Kida et al. *Cureus*, 2018. [[doi](https://doi.org/10.7759/cureus.2548)][[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6021187/)]
### Summary

**Key Idea**:

A U-net-based deep convolutional neural network is trained in the image domain to map CBCT slices of prostate cancer patients to deformably registered planning CT slices, targeting the shading artifact caused by reconstruction from scatter-contaminated and truncated projections. It is compared with an existing planning CT-based image-domain correction (called enhanced CBCT) in terms of spatial non-uniformity, PSNR and SSIM.

**Methodology**:

- Data: CBCT and planning CT (pCT) pairs from 20 prostate cancer patients treated with an Elekta Synergy linear accelerator; five CBCT sets per patient (XVI, 120 kV, 350 mAs). pCT is 512 x 512 per axial slice with 1.074 mm pixels and 1 mm slice thickness; CBCT was output at the same resolution.
- Preprocessing: Otsu thresholding masks, voxels outside the mask set to -1000 HU; pCT was rigidly and then deformably registered to each CBCT with Elastix, giving registered pCT (pCT_r) used as target.
- Model: a modified 2D U-net with 39 layers and about 125.8 million parameters (3 x 3 convolutions with ReLU, 2 x 2 max pooling/unpooling, 1 x 1 output convolution), implemented in Keras.
- Loss and training: mean absolute error, $MAE(\Theta) = \frac{1}{N}\frac{1}{M}\sum_{i=1}^{N}\sum_{j=1}^{M} |Y_{i,j} - P(X_{i,j};\Theta)|$; Adam with learning rate 0.001, $\beta_1 = 0.9$, $\beta_2 = 0.999$, batch size 10; training took about a day on an NVIDIA Titan X. Early stopping if test MAE did not improve for 20 epochs.
- Evaluation: fivefold cross-validation over patients (16 training cases, about 14400 training slices per fold). pCT_r is the reference. Metrics are SNU (difference between maximum and minimum mean values of five 10 x 10 pixel ROIs in fat or muscle), mean pixel value of the ROI with the largest CBCT-vs-pCT_r difference, PSNR and SSIM on those ROIs.

**Results**:

- RMSD of SNU relative to pCT_r fell from 109 to 13 HU (fat ROIs) and from 57 to 11 HU (muscle ROIs) for the proposed method; the enhanced CBCT gave 14 and 7 HU.
- RMSD of the ROI mean pixel value fell from 216 to 11 HU (fat) and from 247 to 14 HU (muscle); the enhanced CBCT gave 10 and 10 HU. Uniformity and pixel values were thus similar for the proposed and the existing correction.
- Averaged over all patients, PSNR was 31.1 (original CBCT), 49.6 (enhanced CBCT) and 50.9 (proposed); SSIM was 0.928, 0.945 and 0.967. The proposed method was better than the enhanced CBCT in PSNR and SSIM (p < 0.01 for both), which the authors attribute to suppression of high-frequency artifacts such as streaks.
- Inference took about 20 seconds for 180 slices of a new patient.
- The method suppressed false pCT_r structures (rectum, bladder) in two patients with large registration errors, but some structures (small intestines, right gluteus maximus muscle) were deformed or disappeared in the output.
- Limitations: registration errors in training pairs may cause false predictions; 2D slice-wise training; single-scanner training data, so applicability to other scanners may require preprocessing such as histogram matching; only 20 patients with pelvis anatomy; the metrics were computed on selected ROIs rather than whole images.
--------

<br/>
<br/>

## 17. Paired cycle-GAN-based image correction for quantitative cone-beam computed tomography <img src="https://img.shields.io/badge/Supervised-blue.svg" alt="Supervised"> <img src="https://img.shields.io/badge/Image--domain-red.svg" alt="Image-domain">
J. Harms et al. *Medical Physics*, 2019. [[doi](https://doi.org/10.1002/mp.13656)][[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7771209/)]
### Summary

**Key Idea**:

The paper corrects CBCT artifacts (streaking, shading, cupping, reduced contrast, HU inaccuracy) in the image domain by learning a mapping from CBCT to registered planning CT. It uses a cycle-GAN trained on paired, registered CBCT/CT data instead of unpaired data, with residual blocks in the generator and a compound loss combining an $l_p$-norm term ($p = 1.5$) and a gradient magnitude distance. The method is compared with a conventional shading/scatter correction and a random forest-based correction.

**Methodology**:

- Data: retrospective CBCT and planning CT from 24 brain and 20 pelvis patients. CT: Siemens SOMATOM Definition AS; CBCT: Varian TrueBeam onboard imager. CBCT was resampled to CT resolution and rigidly registered to planning CT (Velocity AI 3.2.1), followed by inter-patient rigid registration to a single target patient.
- Generator: two downsampling convolution layers, nine residual blocks (two convolution layers plus an element-wise sum each), two deconvolution layers and a tanh layer. A second generator (CT-to-CBCT) and two discriminators complete the cycle-GAN. The discriminators output a pixel-region real/fake map.
- Loss: adversarial loss (mean absolute difference between the discriminator map and a unit mask), cycle consistency, and an added synthetic consistency term between the corrected CBCT and the planning CT. The mean $l_p$-norm loss uses $p = 1.5$, chosen because $l_2$ smoothed the bone/soft-tissue boundary and $l_1$ led to tissue misclassification. A gradient magnitude loss is added. Weights: $\lambda_{adv} = 1$, $\lambda_{loss}^{cycle} = 10$, $\lambda_{loss}^{syn} = 1$, $\lambda_{MPL} = 1$, $\lambda_{GML} = 1$.
- Training: 96 x 96 x 5 patches, Adam with learning rate 2e-4, batch size 8, 150000 iterations (about 15 h on an NVIDIA TITAN XP). Generating a corrected CBCT for one patient takes about 2 min.
- Evaluation: leave-one-out cross validation, planning CT as ground truth; metrics are ME, MAE, PSNR, NCC and spatial non-uniformity (SNU). Paired two-tailed t-tests were used for comparison.
- Comparators: a conventional method that segments soft tissue and builds a compensation map from the assumption of similar mean HU (pelvis only, qualitative comparison), and a random forest method using 32 x 32 x 32 patches.

**Results**:

- Brain (proposed vs. CBCT): MAE 13.0 HU vs. 23.8 HU, PSNR 37.5 dB vs. 32.3 dB, NCC 0.99 vs. 0.98, SNU 0.05 vs. 0.15.
- Pelvis (proposed vs. CBCT): MAE 16.1 HU vs. 56.3 HU, PSNR 30.7 dB vs. 22.2 dB, NCC 0.98 vs. 0.96, SNU 0.09 vs. 0.26. The abstract reports improvements over CBCT of 45%, 16%, 1% and 93% (brain) and 71%, 38%, 2% and 65% (pelvis) for MAE, PSNR, NCC and SNU.
- Random forest method: MAE 13.1 HU (brain) and 17.7 HU (pelvis), PSNR 34.6 dB and 28.0 dB. The proposed method differed significantly from it in PSNR (p < 0.001 in both sites), SNU (p < 0.001 pelvis, p = 0.04 brain) and pelvis NCC (p < 0.001), but not in ME or MAE (MAE p = 0.24 pelvis, p = 0.78 brain). The authors note that averaging over many pixels can hide local errors that are visible in the images.
- Visually, the conventional correction reduced shading and streaking but was limited near air and left more noise; the proposed method produced sharper images with lower noise than the random forest method.
- Limitations: operates only in the image domain with no physical model, so image quality is bounded by the planning CT and planning-CT artifacts (e.g., hip prostheses) can propagate. Air cavities (e.g., bowel gas) are corrected imperfectly and are output at about -1000 HU. In one pelvis patient the corrected body contour deviated because of CBCT artifacts. The bladder HU values were not fully restored. Registration between CT and CBCT is imperfect (2 mm uncertainty in brain), day-to-day anatomy differs from the planning CT, dose calculation was not evaluated, and the cohorts are small (24 brain, 20 pelvis).
--------

<br/>
<br/>

## 18. Artifact removal for unpaired thorax CBCT images using a feature fusion residual network and contextual loss <img src="https://img.shields.io/badge/Unsupervised-orange.svg" alt="Unsupervised"> <img src="https://img.shields.io/badge/Image--domain-red.svg" alt="Image-domain">
W. Zhuang et al. *Journal of Applied Clinical Medical Physics*, 2023. [[doi](https://doi.org/10.1002/acm2.13968)][[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10338820/)]
### Summary

**Key Idea**:

A feature fusion residual network (FFRN) maps thorax CBCT slices with scatter-related artifacts (shadow and cupping artifacts, collectively uneven grayscale artifacts, and streaks) to planning-CT-like images using unaligned CBCT/CT data. Because the images are not spatially aligned, pixel-wise L1/L2 losses are replaced by a contextual loss computed on VGG19 features, which matches regions by semantic and cosine feature similarity and tolerates slight misalignment. The network adapts a residual skip dense block (RSDB) design originally proposed for sparse-angle CT artifact removal.

**Methodology**:

- Data: 2,438 unpaired thorax CBCT and CT 2D slices from 18 patients acquired during image-guided radiotherapy with few motion artifacts, resized to 512 x 512; 200 paired CBCT and CT slices were collected for testing. CBCT slices are the network input and CT slices the target. The data are not publicly available.
- Contextual loss: features from a pretrained VGG19 (gray images replicated to three channels); cosine distance $d_{ij}$ normalized to $\tilde{d}_{ij}=d_{ij}/(\min_k d_{ik}+\epsilon)$ with $\epsilon=10^{-5}$, converted to similarity $w_{ij}=\exp((1-\tilde{d}_{ij})/h)$ with bandwidth $h=0.5$, then $CX(S,T)=\frac{1}{N}\sum_j\max_i CX_{ij}$ and $\mathcal{L}_{CX}=-\log CX$.
- Total loss: $\mathcal{L}(G)=\lambda\,\mathcal{L}_{CX}(G(s),t,l_t)+\mathcal{L}_{CX}(G(s),s,l_s)$ with $\lambda=5$, where the first term uses style features against the CT target and the second uses content features against the input CBCT.
- Network: FFRN built from residual blocks and RSDBs with local feature fusion (concatenation of block outputs) and global residual learning; 3 x 3 convolutions throughout.
- Training: TensorFlow on a GeForce GTX 1080 Ti, Adam, ReLU, 100 epochs, learning rate $10^{-4}$, step size 2.
- Compared with three prior approaches, reported as CNN+CX, U-net+CX and GAN+Per in Table 1 (the paper cites them by reference numbers).

**Results**:

- Average PSNR (test set): FFRN+CX 27.7 dB versus 26.3 dB for the input CBCT, 26.7 dB for CNN+CX, 24.1 dB for U-net+CX and 24.4 dB for GAN+Per.
- Average SSIM: FFRN+CX 0.89 versus 0.88 for CBCT, 0.88 for CNN+CX, and 0.80 for both U-net+CX and GAN+Per.
- MAE: FFRN+CX 33.4 versus 39.0 for CBCT, 34.6 for CNN+CX, 36.8 for U-net+CX and 35.7 for GAN+Per.
- Mean CT numbers (HU) against the reference CT: bone marrow 227.6 for FFRN+CX (CT 231.8, CBCT 221.4); skin -147.0 for FFRN+CX (CT -140.4, CBCT -186.6).
- Average test time per chest image was 0.072 s for the proposed method versus 0.229 s for the comparison method (0.074 s versus 0.213 s per head image).
- Qualitatively, streak and uneven grayscale artifacts were suppressed while nodule texture was preserved in the examples shown, whereas one comparison method lost nodule detail and deformed the esophagus, one produced segmented image blocks, and another did not remove artifacts entirely.
- Limitations: the paper reports no standard deviations or statistical tests; evaluation used 200 slices from the authors' own thorax data that cannot be shared, and the claim of applicability to other anatomies is not tested; the authors note that more complex generation networks could further improve the results.
--------

