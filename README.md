# 💧 Water Quality Prediction & Pollution Source Identification

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

This project is a **Machine Learning based system** that predicts whether water quality is **Good (Safe)** or **Bad (Unsafe)** and identifies the **source of pollution** using the Random Forest algorithm.

---

## 🎯 Objectives

- Predict water quality as **Good** or **Bad**
- Identify pollution sources — Industrial, Agricultural, Sewage
- Provide an interactive UI for real-time water quality analysis

---

## 🧪 Parameters Used

| Parameter | Unit | Safe Range |
|---|---|---|
| pH | 0-14 | 6.5 – 8.5 |
| Turbidity | NTU | 0 – 4 |
| Dissolved Oxygen | mg/L | 6 – 14 |
| Temperature | °C | 10 – 30 |
| Nitrates | mg/L | 0 – 10 |
| Conductivity | μS/cm | 200 – 800 |
| BOD | mg/L | 0 – 5 |

---

## 🤖 Algorithm Used

**Random Forest Classifier**
- Number of Trees: 100
- Task: Binary Classification (Good / Bad)
- Library: Scikit-learn

---

## 📁 Project Structure

```
water-quality-prediction/
│
├── dataset/
│   └── water_quality.csv        # Kaggle dataset
│
├── model/
│   └── water_quality_model.pkl  # Trained ML model
│
├── code/
│   └── water_quality.py         # Main ML code
│
├── ui/
│   └── app.py                   # Streamlit UI code
│
└── README.md                    # Project documentation
```

---

## ⚙️ How to Run This Project

**Step 1 — Install required libraries:**
```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn
```

**Step 2 — Run the ML model:**
```bash
python code/water_quality.py
```

**Step 3 — Run the UI:**
```bash
streamlit run ui/app.py
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~90% |
| Precision | ~88% |
| Recall | ~87% |
| F1 Score | ~88% |

---

## 🔍 Pollution Source Identification

| Source | Indicators |
|---|---|
| Agricultural Runoff | High Nitrates |
| Sewage / Organic Waste | High BOD, Low DO |
| Industrial Discharge | Abnormal pH, High Conductivity |
| Surface Runoff | High Turbidity |

---

## 🛠️ Technologies Used

- **Language:** Python 3.8+
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **UI:** Streamlit / HTML
- **Dataset:** Kaggle Water Quality Dataset
- **Algorithm:** Random Forest Classifier

---

## 👩‍💻 Developer

**Hema M**
- 📧 hemanava88@gmail.com
- 🎓 B.Tech Information Technology
- 🏫 University College of Engineering, Tindivanam
- 📅 2025 – 2026

---

## 📜 License

This project is developed for academic purposes as part of Final Year Project.
