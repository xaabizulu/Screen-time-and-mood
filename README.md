# Digital Habits vs. Mental Health Prediction in Generation Z

A comparative study evaluating the performance of **Classical Machine Learning Pipelines** against **Deep Neural Network Architectures** to predict mental well-being (`mood_score`) based on daily digital behavior logs and lifestyle indicators,
done as the final project of the Articial Intelligence & Machine Learning postgraduate program in the University of Essex.

**Author:** Xabier Zulueta Zarate
**Dataset:** [Kaggle Digital Habits vs Mental Health Dataset](https://www.kaggle.com/datasets/abhishekdave9/digital-habits-vs-mental-health-dataset)

---

## Project Workflow & Methodology

The `Final_Project.ipynb` notebook is organized into the following components:

### 1. Exploratory Data Analysis & Integrity
* **Data Sanitization:** Evaluated missing entries and intentionally preserved 1,865 duplicate entries to retain authentic, overlapping behavioral profiles within a tight feature space.
* **Logical Filtering:** Executed explicit constraint validations ensuring that application logs did not mathematically surpass global uptime thresholds (`hours_on_TikTok <= screen_time_hours`).

### 2. Feature Engineering & Target Mapping
* **Risk Flag Derivation:** Instantiated high-variance binary features capturing critical behavioral thresholds: `low_sleep` and `high_stress`.
* **Target Isolation:** Selected `mood_score` as the primary operational target instead of `stress_level`.

### 3. Preprocessing & Extreme Class Imbalance Handling
* **Data Partitioning:** Implemented a stratified split into Training, Validation, and Test subsets.
* **Target Binning:** Categorized the raw 1-10 numeric `mood_score` into a multi-class matrix: *Happy*, *Neutral*, and *Sad*.
* **Imbalance Assessment:** Documented a highly skewed validation/test distribution with **17,318 Happy**, **2,671 Neutral**, and only **11 Sad** observations (the critical minority group).

### 4. Machine Learning Paradigm 
* **Pipeline Design:** Enclosed estimators inside a scikit-learn `Pipeline` paired with a standard scaler.
* **Models Trained:** Gradient Boosting, Random Forest, Decision Tree, Logistic Regression, SVC, and Gaussian Naive Bayes.
* **Bias Correction:** Leveraged cost-sensitive algorithmic weights (`class_weight="balanced"`) to heavily penalize errors committed on the scarce minority profiles.

### 5. Deep Learning Paradigm 
* **Data Resampling:** Applied **SMOTE** on the training arrays to expand decision boundaries before packing them into mini-batches of size 64.
* **Architecture (`nn.Sequential`):**
  * Linear projection into a wide hidden layer (`hidden_dim`).
  * **Batch Normalization (`BatchNorm1d`)** to eliminate internal covariate shift and stabilize training.
  * **ReLU** activation coupled with **Dropout** regularization.
  * A structural bottleneck layer (`hidden_dim // 2`) to enforce generalized representations.
  * Linear output projection layer for the 3 target categories.
* **Hyperparameter Tuning:** Conducted 100 Optuna trials to optimize dimensions, learning rate, and dropout against a customized scoring objective function:
  $$\text{Score} = 0.5 \cdot \text{F1-Weighted} + 0.25 \cdot \text{Recall-Neutral} + 0.25 \cdot \text{Recall-Sad}$$

---

## Evaluation & Performance Matrix

Final evaluation scores computed on the completely unseen testing partition ($N=20,000$):

| Predictive Model / Architecture | Global Accuracy | F1-Score (Weighted) | Macro F1-Score | Minority Recall (`Sad`) |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting Classifier** | **91.24%** | **0.91** | 0.59 | 18.00% (2 / 11) |
| **Random Forest (Balanced)** | 89.75% | 0.89 | 0.57 | 0.00% (0 / 11) |
| **Deep Neural Network (Optuna)** | 83.56% | 0.86 | **0.61** | **64.00% (7 / 11)** |

### Optimization Synthesis & Trade-offs
* **Classical ML:** Reached the highest global accuracy (**91.24%**), but failed to resolve the critical psychological risk category, capturing only 2 out of 11 at-risk cases.
* **Deep Learning Advantage:** Safely balanced non-linear data boundaries. By training on SMOTE batches and targeting minority metrics inside the Optuna custom score, the PyTorch network achieved the highest **Macro F1-Score (0.61)** and a **64.00% Recall on the Sad class** (capturing 7 out of 11 cases). In sensitive screening environments, the Deep Learning approach is optimal since the cost of a False Negative (missing a person suffering from severe burnout) far outweighs the administrative cost of a False Positive.

---

## 🚀 Setup and Local Replication

Install the required core ecosystem dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn torch optuna
