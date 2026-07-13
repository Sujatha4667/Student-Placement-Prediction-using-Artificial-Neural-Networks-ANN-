"""
generate_dataset.py
Generates a synthetic but realistic "Student Placement" dataset and
saves it as data/student_placement.csv

Features:
    CGPA                 : float (4.0 - 10.0)
    IQ                    : int   (70 - 150)
    Internships           : int   (0 - 3)
    Projects              : int   (0 - 6)
    Communication_Skill   : int   (1 - 10)  (self-rated / HR-rated score)
    Backlogs              : int   (0 - 4)
    Aptitude_Score        : int   (0 - 100) (score in the college aptitude test)
    Extra_Curricular      : int   (0 - 5)   (activities/clubs participated)
    Placed                : 0/1   (target label)
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1200

cgpa = np.round(np.random.normal(7.2, 1.1, N).clip(4.0, 10.0), 2)
iq = np.random.normal(105, 15, N).clip(70, 150).astype(int)
internships = np.random.choice([0, 1, 2, 3], size=N, p=[0.30, 0.35, 0.25, 0.10])
projects = np.random.poisson(2.2, N).clip(0, 6)
communication = np.random.randint(1, 11, N)
backlogs = np.random.choice([0, 1, 2, 3, 4], size=N, p=[0.55, 0.20, 0.15, 0.07, 0.03])
aptitude = np.random.normal(65, 15, N).clip(0, 100).astype(int)
extra_curricular = np.random.choice([0, 1, 2, 3, 4, 5], size=N, p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05])

# Build a latent "placement score" that combines the features with some
# realistic weighting + noise, then threshold it to create the label.
score = (
    0.9 * (cgpa - 4) / 6          # normalized cgpa contribution
    + 0.006 * (iq - 70)
    + 0.55 * internships
    + 0.25 * projects
    + 0.30 * communication
    - 0.6 * backlogs
    + 0.02 * aptitude
    + 0.12 * extra_curricular
)

noise = np.random.normal(0, 1.0, N)
final_score = score + noise

threshold = np.percentile(final_score, 45)  # roughly balanced classes
placed = (final_score > threshold).astype(int)

df = pd.DataFrame({
    "CGPA": cgpa,
    "IQ": iq,
    "Internships": internships,
    "Projects": projects,
    "Communication_Skill": communication,
    "Backlogs": backlogs,
    "Aptitude_Score": aptitude,
    "Extra_Curricular": extra_curricular,
    "Placed": placed
})

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "student_placement.csv")
df.to_csv(csv_path, index=False)
print(df["Placed"].value_counts())
print(df.head())
print("Saved dataset with shape:", df.shape)
