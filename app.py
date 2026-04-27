import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# -----------------------------
# LOAD DATASET
# -----------------------------
data = pd.read_csv("Water_Quality_Assessment.csv")
data.columns = data.columns.str.strip()

# -----------------------------
# CREATE BALANCED WATER QUALITY SCORE
# -----------------------------
data['Water_Quality_Score'] = (
    0.3 * (data['Dissolved Oxygen (mg/L)'] / 14) +
    0.2 * (1 - data['BOD (mg/L)'] / 10) +
    0.2 * (1 - data['COD (mg/L)'] / 50) +
    0.2 * (1 - data['Ammonia (mg/L)'] / 5) +
    0.1
)

# Keep score between 0 and 1
data['Water_Quality_Score'] = data['Water_Quality_Score'].clip(0, 1)

# -----------------------------
# CREATE CATEGORY (DYNAMIC)
# -----------------------------
threshold = data['Water_Quality_Score'].mean()

data['Water_Category'] = data['Water_Quality_Score'].apply(
    lambda x: 1 if x >= threshold else 0
)

# -----------------------------
# FEATURES
# -----------------------------
X = data[['pH',
          'Dissolved Oxygen (mg/L)',
          'Turbidity (NTU)',
          'Electrical Conductivity (µS/cm)',
          'Total Dissolved Solids (TDS) (mg/L)',
          'Nitrate (NO3⁻) (mg/L)',
          'Phosphate (PO4³⁻) (mg/L)',
          'BOD (mg/L)',
          'COD (mg/L)',
          'Coliform Bacteria (CFU/mL)',
          'Ammonia (mg/L)']]

y_class = data['Water_Category']
y_reg   = data['Water_Quality_Score']

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# -----------------------------
# MODELS
# -----------------------------
clf = RandomForestClassifier(n_estimators=120, random_state=42)
reg = RandomForestRegressor(n_estimators=120, random_state=42)

clf.fit(X_train, y_class_train)
reg.fit(X_train, y_reg_train)

# -----------------------------
# SELECT 6th ROW (IMPORTANT CHANGE)
# -----------------------------
sample = X_test.iloc[[5]]   # 6th row
row = sample.iloc[0]

# -----------------------------
# PREDICTION
# -----------------------------
category = clf.predict(sample)[0]
predicted_score = reg.predict(sample)[0]

# -----------------------------
# OUTPUT
# -----------------------------
print("\n----------------------------------")
print("Water Quality Score :", round(predicted_score, 2))

print("\nWater Category :", category)

if category == 1:
    print("Water Quality : Good Water")
else:
    print("Water Quality : Bad Water")

# -----------------------------
# PARAMETER ISSUES
# -----------------------------
print("\nParameter Issues :")

issues=[]
reasons=[]
solutions=[]
sources=[]

# BOD
if row['BOD (mg/L)'] > 5:
    print("High BOD")
    issues.append("bod")
    reasons.append("Organic pollution")
    solutions.append("Improve sewage treatment")
    sources.append("Domestic sewage")

# COD
if row['COD (mg/L)'] > 10:
    print("High COD")
    issues.append("cod")
    reasons.append("Chemical pollution")
    solutions.append("Treat industrial wastewater")
    sources.append("Industrial waste")

# Coliform
if row['Coliform Bacteria (CFU/mL)'] > 1000:
    print("High Coliform Bacteria")
    issues.append("coliform")
    reasons.append("Fecal contamination")
    solutions.append("Disinfect water using chlorination")
    sources.append("Animal or human waste")

# -----------------------------
# RISK LEVEL
# -----------------------------
if len(issues) == 0:
    risk = "Low"
elif len(issues) == 1:
    risk = "Medium"
else:
    risk = "High"

print("\nBacteria Risk Level :", risk)

# -----------------------------
# REASONS
# -----------------------------
if reasons:
    print("\nReason :")
    for r in set(reasons):
        print("-", r)

# -----------------------------
# SOURCES
# -----------------------------
if sources:
    print("\nPollution Source Identification :")
    for s in set(sources):
        print("-", s)

# -----------------------------
# FINAL SOURCE
# -----------------------------
print("\nPredicted Pollution Source :")

if row['COD (mg/L)'] > 10:
    pollution_source = "Industrial Pollution"
elif row['BOD (mg/L)'] > 5 or row['Coliform Bacteria (CFU/mL)'] > 1000:
    pollution_source = "Domestic Sewage"
else:
    pollution_source = "Unknown"

print(pollution_source)

# -----------------------------
# SOLUTIONS
# -----------------------------
if solutions:
    print("\nRecommended Improvements :")
    for s in set(solutions):
        print("-", s)