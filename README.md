# brain_seg_QC
Perform the quality control on T1 MRI data using deep learning


# BrainSegQC: Automated Brain Segmentation Quality Control

This project automates quality control for brain tissue segmentation in T1-weighted MRI using a 3D U-Net model in PyTorch.

## Structure
- `dataset.py`: Loads T1 MRI and segmentation data.
- `models/unet3d.py`: 3D U-Net for segmentation refinement.
- `train.py`: Model training script.
- `infer.py`: Inference and prediction visualization.
- `utils.py`: Utility functions (e.g., metrics, image loading).

## Data Structure
Organize your NIfTI files like this:
```
data/
├── t1/          # Raw T1-weighted images
├── seg_init/    # Initial segmentations (pre-QC)
└── seg_qc/      # Corrected segmentations (post-QC)
```

## Install
```
pip install -r requirements.txt
```

## Train
```
python train.py
```

## Inference
```
python infer.py --input path_to_t1_and_init_seg
```

## Requirements
- PyTorch
- numpy
- nibabel
- scikit-learn
- matplotlib
- tqdm
```



