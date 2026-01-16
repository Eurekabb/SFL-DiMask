# 🔍 SFL-DiMask: Split Federated Learning with Gradient-Saliency Masking for Domain Generalization

This repository provides the implementation of a split federated learning (SFL) framework designed to address domain shift challenges, evaluated on **PACS** and **OfficeHome** datasets using ResNet-18 and GoogLeNet.

---

## 🚀 Features

- Address the domain shift challenges by masking the intermediated representation in SFL
- Modular split ResNet-18 and GoogLeNet models (client/server)
- Gradient-Saliency based selective feature masking
- Evaluation across clients, globally, and per domain

---

## 📦 Dataset Preparation

### 🧊 PACS (DeepLake format)

Due to its size, the PACS dataset is **not included in this repository**.

#### Preparation steps:

1. **Download** the original PACS dataset from:  
   [https://domaingeneralization.github.io](https://domaingeneralization.github.io)

2. **Convert** the raw images to DeepLake format using your own script. Example:
   
   ```python
   import deeplake
   deeplake.create_dataset('pacs-train-local', overwrite=True, ...)

  3.**Directory structure**:

```
	/path/to/your/dataset/PACS/
	├── pacs-train-local/    # DeepLake training set
	└── pacs-test-local/     # DeepLake test set
```



4. **Update paths** in `load_PACS_dataset()` inside `load_data.py` to point to the correct local directory.

🗂 OfficeHome (standard folder format)

1. **Download** the dataset from:
    https://www.hemanthdv.org/officeHomeDataset

2. **Expected directory structure**:

   ```
   swift
   /path/to/your/dataset/OfficeHome/
   ├── Art/
   ├── Clipart/
   ├── Product/
   └── Real_World/
   ```

3.**Verify** the `data_root` path in `load_officehome_dataset()` matches your local setup.



## ⚙️ Running the Code

✅ PACS (ResNet-18)

```
python SFL-DiMask.py -dataset PACS -model ResNet18
```

✅ OfficeHome (GoogLeNet)

```
python SFL-DiMask.py -dataset OfficeHome -model GoogleNet
```

## 📊 Output

Results are saved as `.txt` files in the following structure:

```
./results
```

