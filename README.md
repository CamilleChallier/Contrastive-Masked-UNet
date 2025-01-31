# Self-Supervised Learning for Coronary Artery Segmentation in X-Ray Angiography

## Abstract

**BACKGROUND.** Medical image segmentation is a fundamental process for converting raw biomedical images into structured, interpretable data, which plays a pivotal role in clinical applications. It helps reduce diagnostic errors, relieve the radiologist's workload, and accelerate diagnosis. This study focuses on the segmentation of coronary arteries in X-ray angiography images for disease detection, a complex task hindered by several factors. Accurate segmentation is complicated by the intricate geometry of coronary vessels, low signal-to-noise ratios, and overlapping structures such as catheters, the spine, and ribs. Furthermore, training deep learning models for these tasks typically requires large and meticulously annotated data sets, a resource often scarce in the medical imaging domain due to the complexity of annotation, restricted access to data, and the rarity of certain conditions.

**AIM.** To address this, we introduce CM-UNet, which leverages self-supervised pretraining on unannotated datasets and transfer learning on limited annotated data, enabling accurate disease detection while minimizing the reliance on extensive manual annotations.

![Alt text](figures/CMUNet_architecture.png)

**METHODS/FINDINGS.** Fine-tuning CM-UNet with only 18 images instead of 500 resulted in a 48.7\% increase in Dice score, compared to a 114.73\% rise in baseline models without pretraining. This demonstrates that self-supervised learning can enhance segmentation performance and reduce dependence on large datasets. This is one of the first studies to highlight the importance of self-supervised learning in improving coronary artery segmentation from X-ray angiography, with potential implications for advancing diagnostic accuracy in clinical practice. 

**CONCLUSION.** This study highlights the potential of self-supervised learning in enhancing segmentation performance and reducing the need for extensive labeled datasets. The proposed method demonstrates significant improvements in coronary artery segmentation, even with limited annotated data, and showcase its robustness in adapting to datasets with varying amounts of labeled information.

## Files

```txt
.
├── data_processing
│   └── ... # Directory for data pre processing scripts
├── dataset
│   └── ... # Directory containing FAME2 dataset 
├── Finetuning # Directory containing U-Net implementation for finetuning. 
│   ├── config.py # Set dataset path
│   ├── dataset.py # Define PyTorch Dataset for image segmentation tasks
│   ├── metrics.py # Metrics used for evaluation
│   ├── model.py # UNet model architecture
│   ├── train.py # Run evaluation, hyperparameters finetuning
│   ├── utils.py
│   └── work_dir /
│       └── ... # Saved models, losses and metrics
├── Pretraining
│   ├── CMUNet
│   │   └── ... # Directory containing self-supervised CMUNet implementation
│   ├── MOCO
│   │   └── ... # Directory for MOCO (Momentum Contrast) utilities from PyTorch Lightning Bolts
│   ├── Spark
│       └── ... # Directory for Spark-related pretraining resources
│   └── Transformation_based
│      └── ... # Directory for Model Genesis and MAE self-supervised methods
├── .gitignore # Git configuration file to specify files to ignore
├── README.md # Project README file
└── requirements.txt # File listing project dependencies
```
## Installation
**Clone the repository**

```bash
git clone https://github.com/CamilleChallier/Contrastive-Masked-UNet.git
cd Contrastive-Masked-UNet
```

**Create a conda environment**

```bash
conda create --name env python=3.10.12 -y
conda activate env
```

**Installations**

```bash
pip install -r requirements.txt
```

## Dataset

Place the dataset inside the dataset/ folder with the following structure:
```txt
.
├── dataset
│   ├── imgs/    # Directory containing input images
│   └── masks/ # Directory containing corresponding masks
```

## Pretraining

### With MAE:
In the config.py replace model with 'MAE' 
```bash
conda activate env
cd Pretraining/Transformation_based/
python -W ignore Genesis_Chest_CT.py
```
*Note: Original Code by [MrGiovanni/ModelsGenesis](https://github.com/MrGiovanni/ModelsGenesis).*

### With Model Genesis:
In the config.py replace model with 'Model Genesis' 
```bash
cd Pretraining/Transformation_based/
conda activate env
python -W ignore pytorch/Genesis_Chest_CT.py
```
*Note: Original Code by [MrGiovanni/ModelsGenesis](https://github.com/MrGiovanni/ModelsGenesis).*

### With MoCo:

**Installation**
```bash
conda create --name SSL_Contrastive python==3.10
conda activate SSL_Contrastive
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
cd Pretraining/MoCo/
pip install -r requirements.txt
```

**Run**
```bash
cd pl_bolts/models/self_supervised/moco/
python moco2_module.py  --dataset=medical --max_epochs=500 
```

*Note: Original Code by [Wolfda95/SSL-MedicalImagining-CL-MAE](https://github.com/Wolfda95/SSL-MedicalImagining-CL-MAE) and [keyu-tian/SparK](https://github.com/keyu-tian/SparK).*

### With Spark:

**Installations**

```bash
conda create --name SSL_Masked_Autoencoder python==3.8
conda activate SSL_Masked_Autoencoder
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch
cd Pretraining/Spark/
pip install -r requirements.txt
```

```bash
cd Pretraining/spark/pretrain/
screen python main.py --full_unet True --model "unet_sparse"
```

*Note: Original Code by [keyu-tian/SparK](https://github.com/keyu-tian/SparK).*

### With CM-UNet:

**Environment Installations**

```bash
cd Pretraining/CM-UNet/
conda env create -f environment.yml
conda activate cmunet
```

**Run**
```bash
bash training/dist_train.sh configs/cmunet_config.py 1 --local_rank 0
```

*Note: Original Code by [ZhichengHuang/CMAE](https://github.com/ZhichengHuang/CMAE).*


## Finetuning

The training script supports the following arguments:

| Argument            | Short | Type    | Default                                    | Description                          |
|---------------------|-------|---------|--------------------------------------------|--------------------------------------|
| `--epochs`         | `-e`  | `list`  | `[2]`                                     | Number of epochs                    |
| `--batch-size`     | `-b`  | `list`  | `[16, 32]`                                | Batch size options                   |
| `--learning-rate`  | `-l`  | `list`  | `[0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]`     | Learning rate values                 |
| `--pretrained`     | `-p`  | `str`   | `None`                                    | Path to a pretrained model          |
| `--name`           | `-n`  | `str`   | `"base"`                                  | Name for the trained model          |
| `--ratio`          | `-r`  | `float` | `0.1`                                     | Ratio of the fine-tuning dataset    |

To train the UNet model, use the following command:

```bash
conda activate env
cd Finetuning/
python train.py -p "path_to_pretrained_model"
```
