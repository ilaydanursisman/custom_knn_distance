# custom_knn_distance

A custom distance metric for K-Nearest Neighbors (KNN) classification that adapts the radius based on local density. This project offers an alternative to the traditional Euclidean distance by improving classification performance, especially in imbalanced and sparse regions.

---

## 📌 Features

- ✅ Density-aware custom distance metric  
- ✅ Integration with `scikit-learn`’s `KNeighborsClassifier`  
- ✅ Comparative performance evaluation (Euclidean vs Custom)  
- ✅ Specificity, accuracy, precision, recall, and F1-score tracking  
- ✅ Compatible with multiple datasets (e.g., Car Evaluation)

---

## 🧠 How It Works

This project introduces a custom distance metric designed to enhance the performance of the K-Nearest Neighbors (KNN) algorithm, especially in datasets with varying local densities.

Instead of using a fixed metric like Euclidean distance, this approach dynamically adjusts the distance calculation by incorporating **local data density**, enabling the model to be more sensitive to the structure of the dataset.

---

### 1. Local Density Function \( D(y) \)

The local density around a point \( y \) is estimated using a **Gaussian kernel** function:

\[
D(y) = \frac{1}{2\pi r^2} \sum_{i=1}^{N} \exp\left(-\frac{\|x_i - y\|^2}{2r^2} \right)
\]

- \( x_i \): Training points in the dataset  
- \( r \): Initial radius parameter (set to 1.0)  
- \( \exp \): Gaussian kernel that emphasizes nearby points  

> **Note:** A Gaussian kernel \( K(d) = \exp(-d^2 / 2) \) is used for smooth density estimation.

---

### 2. Adaptive Radius \( r(y) \)

The radius around each point is adjusted according to its local density:

\[
r(y) = \frac{1}{1 + D(y)}
\]

This mechanism ensures:
- Smaller radius in **dense** regions  
- Larger radius in **sparse** regions  

Allowing the metric to adapt to local structure and avoid over/under-sampling bias.

---

### 3. Custom Distance Metric \( M(x, y) \)

The final distance between a query point \( x \) and a training point \( y \) is calculated as:

\[
M(x, y) = \frac{\|x - y\|}{D(y)}
\]

This formulation penalizes distances in **sparse areas** (where \( D(y) \) is small) and compresses distances in **dense areas**, improving KNN’s neighborhood selection quality.

---

### ✅ Python Integration

The entire formulation is implemented in the `custom_knn_distance/metric.py` file and can be directly integrated into any scikit-learn KNN model:

```python
from custom_knn_distance import create_custom_distance
from sklearn.neighbors import KNeighborsClassifier

custom_metric = create_custom_distance(X_train)
knn = KNeighborsClassifier(n_neighbors=3, metric=custom_metric)
```

This metric is particularly effective on datasets with class imbalance or local density variations such as:
- [Car Evaluation Dataset](https://archive.ics.uci.edu/ml/datasets/car+evaluation)  
- [Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)

By adapting the distance measure to the structure of the data, this method improves classification performance while preserving interpretability.

---

## 📦 Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/ilaydanursisman/custom_knn_distance.git
cd custom_knn_distance
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
custom_knn_distance/
├── custom_knn_distance/
│  ├─ distance.py
│  ├─ evaluation.py
│  ├─ knn_wrapper.py
│  ├─ preprocessing.py
│  └─ utils.py
│
├─ datasets/
│  ├─ car.data
│  └─ student.csv
│
├─ examples/
│  ├─ car_demo_notebook.ipynb
│  ├─ demo_car.py
│  └─ demo_student.py
│
├─ tests/
│  ├─ test_distance.py
│  ├─ test_evaluation.py
│  └─ test_preprocessing.py
│
├─ .gitignore
├─ LICENSE
├─ pyproject.toml
├─ README.md
├─ requirements.txt
└─ setup.py
```

---

## 📊 Example Usage

### ▶️ Run CLI Example

```bash
python examples/demo_car.py
```

### 📒 Jupyter Notebook (Visual + Interactive)

See [`car_demo_notebook.ipynb`](examples/car_demo_notebook.ipynb) for step-by-step explanation.

---

## 📊 Results

This section summarizes the evaluation of the custom KNN distance metric against the standard Euclidean distance on two benchmark datasets.

---

### 🚗 Car Evaluation Dataset (UCI)

| Metric / Class | Precision (Euclidean) | Precision (Custom) | Recall (Euclidean) | Recall (Custom) | F1-Score (Euclidean) | F1-Score (Custom) |
|----------------|----------------------|--------------------|--------------------|-----------------|----------------------|------------------|
| acc            | 0.5439               | **0.6970**         | 0.4139             | **0.7707**      | 0.4686               | **0.7312**       |
| good           | 0.4000               | **0.5267**         | 0.0297             | **0.3890**      | 0.0552               | **0.4323**       |
| unacc          | 0.8430               | **0.9315**         | 0.9992             | 0.9413          | 0.9144               | **0.9363**       |
| vgood          | 0.6000               | **0.9167**         | 0.0461             | **0.3692**      | 0.0857               | **0.5070**       |
| **Macro Avg**  | 0.5967               | **0.7680**         | 0.3722             | **0.6176**      | 0.3810               | **0.6517**       |
| **Weighted Avg** | 0.7494             | **0.8628**         | 0.7946             | **0.8599**      | 0.7498               | **0.8545**       |
| **Accuracy**   | 0.7946               | **0.8599**         | –                  | –               | –                    | –                |

**Specificity (per class)**:

| Class  | Euclidean | Custom Distance |
|--------|-----------|-----------------|
| unacc  | 0.4197    | **0.8054**      |
| acc    | 0.9039    | 0.9017          |
| vgood  | 1.0000    | 0.9980          |
| good   | 1.0000    | 0.9833          |

---

### 🎓 Student Performance Dataset (UCI)

| Model         | Precision | Recall  | F1-Score | Support | Avg Specificity | Accuracy |
|---------------|-----------|---------|----------|---------|-----------------|----------|
| KNN Euclidean | 0.6311    | 0.6124  | 0.6102   | 209.0   | 0.6759          | 0.5981   |
| KNN Custom    | 0.6285    | **0.6746** | **0.6396** | 209.0   | **0.6812**     | **0.6651** |


> **Observation:**  
> On this dataset as well, the custom distance improves recall, F1-score, and average specificity compared to Euclidean distance, resulting in higher overall accuracy.

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## 📃 License

This project is **not open source** and is protected under a custom license.  
All rights, including the custom distance metric algorithm and associated code, belong to **İlayda Nur Şişman**.

> Unauthorized copying, distribution, modification, or usage of any part of this repository is strictly prohibited without express written permission from the author.

For licensing inquiries, collaboration requests, or academic access, please contact:

📬 **ilaydasisman65@gmail.com**

> ⚠️ This codebase contains an original distance metric algorithm currently undergoing patent evaluation.

---

## 👩‍💻 Author

**İlayda Nur Şişman**  
Mathematics & Information Systems Engineering  
📬 [LinkedIn](https://www.linkedin.com/in/ilaydanursisman)

---

## ⭐ Acknowledgements

This project uses publicly available datasets from the **UCI Machine Learning Repository**:

- [Car Evaluation Dataset](https://archive.ics.uci.edu/ml/datasets/car+evaluation)  
- [Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)  

I sincerely thank the **UCI Machine Learning Repository** for providing these datasets.