# Classification Multiclasse d'Images Faciales pour l'Estimation de l'Âge via EfficientNetB3 Fine-Tuné

**Filière :** 3IACN2 — ENSA de Fès  
**Auteurs :** Ikram Boukhabza & Fatima El Ouazzani Thami  
**Encadrant :** Pr. Oussama EL GANNOUR  
**Module :** Deep Learning & NLP

---

## Description

Ce projet implémente un système d'**estimation de l'âge** à partir d'une photo de visage, formulé comme un problème de **classification en 8 classes** :

| Classe | Tranche d'âge |
|--------|--------------|
| 0 | 0 – 10 ans |
| 1 | 11 – 20 ans |
| 2 | 21 – 30 ans |
| 3 | 31 – 40 ans |
| 4 | 41 – 50 ans |
| 5 | 51 – 60 ans |
| 6 | 61 – 70 ans |
| 7 | 71+ ans |

Deux architectures sont comparées :
- **CNN Baseline** — entraîné from scratch
- **EfficientNetB3** — transfer learning depuis ImageNet (modèle principal)

---

## Dataset

**UTKFace-New** disponible sur Kaggle :  
🔗 https://www.kaggle.com/datasets/jangedoo/utkface-new

- ~23 708 images de visages annotées (âge, genre, origine)
- Format de nommage : `[age]_[gender]_[race]_[date].jpg`
- Split utilisé : **70% train / 15% validation / 15% test** (stratifié)

---

## Structure du projet

```
AgeEstimation_CNN/
├── checkpoints/        # Meilleurs poids sauvegardés par époque
├── models/             # Modèles complets (Phase 1 et Phase 2)
├── logs/               # Historiques d'entraînement (JSON)
├── splits/             # Fichiers CSV train/val/test
├── figures/            # Courbes d'entraînement et visualisations
├── model_final/        # Modèle final Keras + export TFLite + config JSON
└── results/            # Métriques d'évaluation
```

---

## Reproduire les expériences

### 1. Sur Kaggle (recommandé)

1. Créer un nouveau notebook sur [kaggle.com](https://www.kaggle.com)
2. Activer le GPU : **Settings → Accelerator → GPU P100**
3. Ajouter le dataset : **Add Data → jangedoo/utkface-new**
4. Importer et exécuter le notebook
5. Les résultats sont sauvegardés automatiquement dans `/kaggle/working/AgeEstimation_CNN/`

### 2. En local

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-username>/<votre-repo>.git
cd <votre-repo>

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Télécharger le dataset UTKFace via l'API Kaggle
kaggle datasets download -d jangedoo/utkface-new
unzip utkface-new.zip -d data/UTKFace/

# 4. Lancer Jupyter
jupyter notebook
```

---

## Hyperparamètres principaux

| Paramètre | Valeur |
|-----------|--------|
| `img_size` | 224 × 224 |
| `batch_size` | 32 |
| `num_classes` | 8 |
| `dropout_rate` | 0.6 |
| `l2_lambda` | 2e-4 |
| `lr_phase1` | 1e-3 |
| `epochs_phase1` | 15 |
| `lr_phase2` | 1e-5 |
| `epochs_phase2` | 20 |
| `unfreeze_layers` | 50 |
| `label_smoothing` | 0.12 |
| `SEED` | 42 |

---

## Environnement

| Composant | Version |
|-----------|---------|
| Python | 3.12.12 |
| TensorFlow | 2.19.0 |
| Keras | 3.10.0 |
| NumPy | 2.0.2 |
| scikit-learn | 1.6.1 |
| OpenCV | 4.13.0 |
| Matplotlib | 3.10.0 |
| Seaborn | 0.13.2 |
| Pandas | 2.3.3 |
| GPU | NVIDIA Tesla P100 (Kaggle) |
| RAM | 30 GB |

---

## Résultats

| Modèle | Accuracy (test) |
|--------|----------------|
| CNN Baseline | — |
| EfficientNetB3 Phase 1 | — |
| EfficientNetB3 Phase 2 | — |

> Complétez ce tableau avec vos résultats après exécution.

---

## Licence

Projet académique — ENSA de Fès, 2024–2025.
