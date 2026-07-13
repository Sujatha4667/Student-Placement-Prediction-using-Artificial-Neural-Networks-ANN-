# Student Placement Prediction using Artificial Neural Networks (ANN)

An **Artificial Neural Network (ANN)** built with **TensorFlow/Keras** to predict whether a student will be **Placed** or **Not Placed**, based on academic performance and skill-related attributes.

---

## Project Overview

This project was built as part of an assignment on Artificial Neural Networks. It covers the complete ML workflow:

1. Data Exploration (EDA)
2. Data Preprocessing (scaling & train/test split)
3. Building & training an ANN using TensorFlow/Keras
4. Model evaluation (accuracy, confusion matrix, classification report)
5. Hyperparameter experimentation (comparing two different network architectures/learning rates)
6. Observations & conclusions

---

## Repository Structure

```
student-placement-ann/
├── Student_Placement_ANN.ipynb   # Main Jupyter Notebook (fully executed)
├── data/
│   ├── student_placement.csv     # Dataset (1200 records)
│   └── generate_dataset.py       # Script used to generate the dataset
├── images/                       # Screenshots of results (auto-saved by the notebook)
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_boxplots.png
│   ├── baseline_accuracy_loss.png
│   ├── baseline_confusion_matrix.png
│   ├── experiment_accuracy_loss.png
│   ├── experiment_confusion_matrix.png
│   └── model_comparison.png
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation (this file)
```

---

## Dataset

**File:** `data/student_placement.csv` (1200 rows, 9 columns)

A synthetically generated but realistically-distributed student placement dataset. The label
(`Placed`) is derived from a weighted combination of the features plus random noise, mimicking
real-world placement patterns while keeping the classes reasonably balanced (~55% / 45%).

| Column | Description | Range |
|---|---|---|
| `CGPA` | Cumulative Grade Point Average | 4.0 – 10.0 |
| `IQ` | IQ score | 70 – 150 |
| `Internships` | Number of internships completed | 0 – 3 |
| `Projects` | Number of academic/personal projects | 0 – 6 |
| `Communication_Skill` | Communication skill rating | 1 – 10 |
| `Backlogs` | Number of pending backlogs | 0 – 4 |
| `Aptitude_Score` | Score in college aptitude test | 0 – 100 |
| `Extra_Curricular` | Number of extra-curricular activities | 0 – 5 |
| `Placed` | **Target label** — 1 = Placed, 0 = Not Placed | 0 / 1 |

> You can regenerate the dataset any time by running `python data/generate_dataset.py`.
> If you'd rather use a real-world dataset, a popular public alternative is the
> ["Campus Recruitment" / "Placement Data Full Class" dataset on Kaggle](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement) — the notebook's
> preprocessing/ANN pipeline can be adapted to it with minor column-name changes.

---

## Model Architecture

**Baseline Model**
```
Input (8 features)
 → Dense(16, activation='relu')
 → Dense(8,  activation='relu')
 → Dense(1,  activation='sigmoid')
Optimizer: Adam (lr=0.001) | Loss: Binary Crossentropy | Epochs: 100
```

**Experiment Model** (hyperparameter comparison — deeper/wider network + lower learning rate)
```
Input (8 features)
 → Dense(64, activation='relu')
 → Dense(32, activation='relu')
 → Dense(16, activation='relu')
 → Dense(1,  activation='sigmoid')
Optimizer: Adam (lr=0.0005) | Loss: Binary Crossentropy | Epochs: 100
```

---

## Results

| Model | Hidden Layers | Learning Rate | Test Accuracy | Test Loss |
|---|---|---|---|---|
| Baseline | [16, 8] | 0.001 | **81.25%** | 0.4518 |
| Experiment | [64, 32, 16] | 0.0005 | 75.42% | 0.8034 |

**Key takeaway:** the smaller baseline network generalized *better* than the larger experiment
network on this dataset — the extra capacity in the experiment model led to overfitting rather
than improved accuracy, as seen in the growing gap between training and validation curves.
See `images/model_comparison.png` and the notebook's "Observations & Conclusion" section for
full details. (Exact numbers may vary slightly on re-run due to random weight initialization.)

## ⚙️ Setup & Execution Steps

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/student-placement-ann.git
cd student-placement-ann
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Regenerate the dataset
```bash
python data/generate_dataset.py
```

### 5. Launch Jupyter Notebook
```bash
jupyter notebook Student_Placement_ANN.ipynb
```
Then click **Run All** (Cell → Run All / Kernel → Restart & Run All) to re-execute every cell
top to bottom. All plots will be regenerated and saved into the `images/` folder automatically.

> Alternatively, run it headlessly from the command line:
> ```bash
> jupyter nbconvert --to notebook --execute --inplace Student_Placement_ANN.ipynb
> ```

---

## Tech Stack
- Python 3.12
- TensorFlow / Keras
- scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

---

## Observations & Conclusion

- Both models learned the placement pattern well since `CGPA`, `Internships`,
  `Communication_Skill`, and `Aptitude_Score` are strong predictors in this dataset.
- The **baseline model (2 hidden layers, smaller)** outperformed the **larger experiment model
  (3 hidden layers, more neurons, lower learning rate)** on test accuracy — a good illustration
  that bigger networks are not automatically better, especially on datasets of this size.
- Regularization techniques (Dropout, L2), early stopping, and k-fold cross-validation would
  likely help the larger model close this gap and are natural next steps.

---

## Author

sujatha4667


Artificial Neural Networks (ANN) - Week 2 Assignment
