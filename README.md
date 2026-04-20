# SpectraGAN: Multi-Spectral Super-Resolution for Earth Observation

## What it is
SpectraGAN is a novel Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) engineered explicitly for Earth Observation (EO) and remote sensing. While traditional super-resolution architectures are hardcoded for 3-channel (RGB) natural photography, SpectraGAN is built from the ground up to ingest, process, and super-resolve 4-channel (RGB + Near-Infrared) satellite telemetry. 

By mapping Low-Resolution (64x64) multi-spectral inputs to High-Resolution (256x256) outputs, SpectraGAN preserves critical radiometric signatures and topological boundaries that are invisible in the standard visual spectrum. This framework is designed to assist urban planners, climate scientists, and agricultural analysts by synthetically bridging the gap between high-frequency (but low-resolution) satellite passes and expensive, sub-meter aerial photography.

## Architecture and Training Method
SpectraGAN utilizes a "Hybrid Extraction" paradigm, leveraging the mathematical stability of Residual-in-Residual Dense Blocks (RRDB) while stripping away the rigid constraints of traditional RGB-only frameworks.

* **Multi-Spectral Generator:** The RRDBNet architecture has been modified at the ingestion and projection layers to natively handle `in_channels=4` and `out_channels=4`. To preserve absolute radiometric contrast across diverse geographical terrains, all Batch Normalization (BN) layers have been strictly removed.
* **Relativistic Discriminator (RaGAN):** The adversarial component utilizes a multi-spectral VGG-style network that calculates a relativistic Binary Cross-Entropy (BCE) loss. It trains the generator not just to create realistic images, but to actively decrease the discriminator's confidence in the genuine ground-truth data.
* **Tensor-Sliced Perceptual Loss:** Because standard VGG19 feature extractors crash when presented with 4-channel data, SpectraGAN introduces a dynamic Tensor Slicing loss function. It enforces strict L1 pixel loss across all four channels (RGB+NIR) for radiometric fidelity, while dynamically peeling off the NIR band to calculate structural perceptual loss on the visible spectrum using standard ImageNet priors.
* **Optimization:** Both networks are trained utilizing the Adam optimizer (B1=0.9, B2=0.999) with an initial learning rate of 1e-4 within a custom PyTorch loop.

## Dataset
The model is trained on a custom, cloud-native derivation of the **SEN12MS** dataset. 
* **Size:** 127,270 rigorously aligned multi-spectral image pairs.
* **Format:** Pre-processed, standardized 4-channel PNGs preserving the Near-Infrared (NIR) band alongside RGB.
* **Degradation Pipeline:** High-Resolution (256x256) ground truths were artificially degraded by a factor of 4x using bicubic interpolation to generate mathematically co-registered Low-Resolution (64x64) training inputs.
* **Cloud Streaming:** To bypass the standard 19GB disk limits of cloud-GPU environments (e.g., Kaggle, Colab), the dataset is deployed as a Hugging Face `DatasetDict` and streamed directly into GPU memory via a custom PyTorch DataLoader.

## Live Demo
A deployable microservices architecture has been developed to democratize access to the SpectraGAN inference engine. 

The frontend provides an interactive, geospatial drag-and-drop interface. Users can upload low-resolution satellite patches and receive a high-resolution output rendered into a dynamic "Before/After" slider. The backend, powered by FastAPI and PyTorch, natively ingests 4-channel `.tif` files. It also includes an extrapolation module to approximate a neutral pseudo-NIR band for users who only possess standard commercial 3-channel RGB imagery, preventing tensor dimensionality errors during live inference.

## Project Structure
```text
SpectraGAN/
│
├── models/
│   ├── generator.py       # 4-channel RRDBNet
│   └── discriminator.py   # 4-channel RaGAN
│
├── core/
│   ├── loss.py            # Tensor Sliced VGG + L1 Loss
│   └── dataset.py         # Hugging Face cloud streaming dataloader
│
├── train.py               # Monolithic PyTorch training loop
├── eval.py                # PSNR and SSIM benchmarking script
├── inference.py           # Single-image processing script
│
├── requirements.txt       # Environment dependencies
├── README.md              # Project documentation
└── LICENSE                # MIT License
```

## Setup and Recreation

To replicate the training environment and reproduce the benchmarking metrics, follow the steps below.

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/SpectraGAN.git](https://github.com/your-username/SpectraGAN.git)
cd SpectraGAN
```

**2. Install Dependencies**
Ensure you are using a Python 3.8+ environment. A GPU-enabled PyTorch installation is strictly required.
```bash
pip install -r requirements.txt
```

**3. Dataset Access**
You do not need to download the 30GB dataset locally. Ensure your machine has internet access, and the `core/dataset.py` script will stream the tensors automatically from the Hugging Face Hub during training.

**4. Train the Model**
To initialize the multi-spectral training loop from scratch:
```bash
python train.py
```
Model weights will be automatically checkpointed and saved to the `weights/` directory at the conclusion of every epoch.

**5. Evaluate and Benchmark**
To run the automated evaluation script, which calculates multi-spectral PSNR and SSIM across the unseen test split and generates visual comparisons:
```bash
python eval.py --weights weights/SpectraGAN_Gen_Epoch_Best.pth
```

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this architecture for academic and commercial Earth Observation research.