import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# ─────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────
data = pd.read_csv("Water_Quality_Assessment.csv")
data.columns = data.columns.str.strip()

# ─────────────────────────────
# 2. WATER QUALITY SCORE
# ─────────────────────────────
data['Water_Quality_Score'] = (
    0.3 * (data['Dissolved Oxygen (mg/L)'] / 14) +
    0.2 * (1 - data['BOD (mg/L)'] / 10) +
    0.2 * (1 - data['COD (mg/L)'] / 50) +
    0.2 * (1 - data['Ammonia (mg/L)'] / 5) +
    0.1
)
data['Water_Quality_Score'] = data['Water_Quality_Score'].clip(0, 1)

# ─────────────────────────────
# 3. CATEGORY (DYNAMIC THRESHOLD)
# ─────────────────────────────
threshold = data['Water_Quality_Score'].mean()

data['Water_Category'] = data['Water_Quality_Score'].apply(
    lambda x: 1 if x >= threshold else 0
)

# ─────────────────────────────
# 4. FEATURES
# ─────────────────────────────
feature_cols = [
    'pH',
    'Dissolved Oxygen (mg/L)',
    'Turbidity (NTU)',
    'Electrical Conductivity (µS/cm)',
    'Total Dissolved Solids (TDS) (mg/L)',
    'Nitrate (NO3⁻) (mg/L)',
    'Phosphate (PO4³⁻) (mg/L)',
    'BOD (mg/L)',
    'COD (mg/L)',
    'Coliform Bacteria (CFU/mL)',
    'Ammonia (mg/L)'
]

X       = data[feature_cols]
y_class = data['Water_Category']
y_reg   = data['Water_Quality_Score']

# ─────────────────────────────
# 5. TRAIN TEST SPLIT
# ─────────────────────────────
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# ─────────────────────────────
# 6. TRAIN MODELS
# ─────────────────────────────
clf = RandomForestClassifier(n_estimators=120, random_state=42)
reg = RandomForestRegressor(n_estimators=120, random_state=42)

clf.fit(X_train, y_class_train)
reg.fit(X_train, y_reg_train)

# ─────────────────────────────
# 7. PREDICTIONS
# ─────────────────────────────
y_pred_class = clf.predict(X_test)
y_pred_score = reg.predict(X_test)

# ─────────────────────────────
# 8. COMPUTE METRICS
# ─────────────────────────────
accuracy  = accuracy_score(y_class_test, y_pred_class)
precision = precision_score(y_class_test, y_pred_class)
recall    = recall_score(y_class_test, y_pred_class)
f1        = f1_score(y_class_test, y_pred_class)
cm        = confusion_matrix(y_class_test, y_pred_class)

# ─────────────────────────────
# 9. ROW LIST — change rows here
# All must be between 0 and 199
# ─────────────────────────────
row_list = [10, 50, 85, 120, 175]

# ─────────────────────────────
# 10. SAFE LIMITS
# ─────────────────────────────
safe_limits = {
    'pH'                                  : 8.5,
    'Dissolved Oxygen (mg/L)'             : 6.0,
    'Turbidity (NTU)'                     : 4.0,
    'Electrical Conductivity (µS/cm)'     : 1500,
    'Total Dissolved Solids (TDS) (mg/L)' : 500,
    'Nitrate (NO3⁻) (mg/L)'              : 10.0,
    'Phosphate (PO4³⁻) (mg/L)'           : 2.0,
    'BOD (mg/L)'                          : 5.0,
    'COD (mg/L)'                          : 10.0,
    'Coliform Bacteria (CFU/mL)'          : 1000,
    'Ammonia (mg/L)'                      : 0.5
}

short_labels = [
    'pH', 'DO', 'Turbidity', 'EC', 'TDS',
    'Nitrate', 'Phosphate', 'BOD', 'COD',
    'Coliform', 'Ammonia'
]

# ─────────────────────────────
# 11. STORE RESULTS FOR GRAPHS LATER
# ─────────────────────────────
results = {}

# ══════════════════════════════
# FIRST LOOP — PRINT ALL OUTPUT
# ══════════════════════════════
for row_index in row_list:

    sample          = X_test.iloc[[row_index]]
    row             = sample.iloc[0]
    category        = clf.predict(sample)[0]
    predicted_score = reg.predict(sample)[0]

    # Pollution source
    if row['COD (mg/L)'] > 10:
        pollution_source = "Industrial Pollution"
    elif row['BOD (mg/L)'] > 5 or row['Coliform Bacteria (CFU/mL)'] > 1000:
        pollution_source = "Domestic Sewage"
    else:
        pollution_source = "Unknown / Clean"

    # Store for graph loop later
    results[row_index] = {
        'row'             : row,
        'category'        : category,
        'predicted_score' : predicted_score,
        'pollution_source': pollution_source
    }

    # Print output
    print("\n" + "=" * 50)
    print(f"SAMPLE PREDICTION — Row {row_index}")
    print("=" * 50)
    print(f"\nRow Index           : {row_index}")
    print(f"Water Quality Score : {round(predicted_score, 4)}")
    print(f"Water Category      : {category}")
    print(f"Water Quality       : {'Good Water' if category == 1 else 'Bad Water'}")

    print("\nParameter Issues:")

    issues    = []
    reasons   = []
    solutions = []
    sources   = []

    if row['BOD (mg/L)'] > 5:
        print("  - High BOD")
        issues.append("bod")
        reasons.append("Organic pollution")
        solutions.append("Improve sewage treatment")
        sources.append("Domestic sewage")

    if row['COD (mg/L)'] > 10:
        print("  - High COD")
        issues.append("cod")
        reasons.append("Chemical pollution")
        solutions.append("Treat industrial wastewater")
        sources.append("Industrial waste")

    if row['Coliform Bacteria (CFU/mL)'] > 1000:
        print("  - High Coliform Bacteria")
        issues.append("coliform")
        reasons.append("Fecal contamination")
        solutions.append("Disinfect water using chlorination")
        sources.append("Animal or human waste")

    if not issues:
        print("  - No major issues found")

    risk = "Low" if len(issues) == 0 else "Medium" if len(issues) == 1 else "High"
    print(f"\nBacteria Risk Level : {risk}")

    if reasons:
        print("\nReasons:")
        for r in set(reasons):
            print(f"  - {r}")

    if sources:
        print("\nPollution Source Identification:")
        for s in set(sources):
            print(f"  - {s}")

    print(f"\nPredicted Pollution Source : {pollution_source}")

    if solutions:
        print("\nRecommended Improvements:")
        for s in set(solutions):
            print(f"  - {s}")

print("\n" + "=" * 50)
print("ALL OUTPUT DONE! Now showing graphs...")
print("=" * 50 + "\n")

# ══════════════════════════════
# SECOND LOOP — SHOW ALL GRAPHS
# ══════════════════════════════

# Static Graph 1 — Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bad Water', 'Good Water'],
            yticklabels=['Bad Water', 'Good Water'],
            annot_kws={"size": 14, "weight": "bold"})
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Actual Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# Static Graph 2 — Feature Importance
importances = clf.feature_importances_
feat_df = pd.DataFrame({
    'Feature'   : feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(9, 6))
bars = plt.barh(feat_df['Feature'], feat_df['Importance'],
                color='steelblue', edgecolor='white')
for bar, val in zip(bars, feat_df['Importance']):
    plt.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
             f'{round(val * 100, 1)}%',
             va='center', fontsize=9, fontweight='bold')
plt.title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

# Static Graph 3 — Model Accuracy Metrics
plt.figure(figsize=(7, 5))
metrics    = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values_pct = [
    round(accuracy  * 100, 2),
    round(precision * 100, 2),
    round(recall    * 100, 2),
    round(f1        * 100, 2)
]
bar_colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
bars = plt.bar(metrics, values_pct,
               color=bar_colors, edgecolor='white', width=0.5)
for bar, val in zip(bars, values_pct):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f'{val}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.ylim(0, 110)
plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
plt.ylabel('Score (%)', fontsize=12)
plt.tight_layout()
plt.savefig('model_metrics.png', dpi=150)
plt.show()

# Dynamic Graphs — Score Meter + Parameter Values for each row
for row_index in row_list:

    r               = results[row_index]
    row             = r['row']
    category        = r['category']
    predicted_score = r['predicted_score']
    pollution_source= r['pollution_source']
    color_needle    = '#00e676' if category == 1 else '#ff4757'

    # Graph 4 — Score Meter
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0d1b2a')
    ax.set_facecolor('#061525')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.axis('off')

    t_red = np.linspace(np.pi, np.pi * 0.42, 100)
    ax.plot(np.cos(t_red), np.sin(t_red),
            color='#ff4757', linewidth=14, alpha=0.5)

    t_yellow = np.linspace(np.pi * 0.42, np.pi * 0.58, 50)
    ax.plot(np.cos(t_yellow), np.sin(t_yellow),
            color='#ffa502', linewidth=14, alpha=0.5)

    t_green = np.linspace(np.pi * 0.58, 0, 100)
    ax.plot(np.cos(t_green), np.sin(t_green),
            color='#00e676', linewidth=14, alpha=0.5)

    needle_angle = np.pi - (predicted_score * np.pi)
    needle_x     = 0.75 * np.cos(needle_angle)
    needle_y     = 0.75 * np.sin(needle_angle)

    ax.annotate('', xy=(needle_x, needle_y), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->',
                                color=color_needle,
                                lw=3.5, mutation_scale=22))

    circle = plt.Circle((0, 0), 0.08, color=color_needle, zorder=5)
    ax.add_patch(circle)

    ax.text(0, 0.38, f'{round(predicted_score, 4)}',
            ha='center', va='center', fontsize=30,
            fontweight='bold', color=color_needle,
            fontfamily='monospace')

    result_text = 'GOOD WATER' if category == 1 else 'BAD WATER'
    ax.text(0, 0.18, result_text,
            ha='center', va='center', fontsize=14,
            fontweight='bold', color=color_needle)

    ax.text(0, -0.05, f'Row Index : {row_index}',
            ha='center', fontsize=10,
            color='#80deea', fontfamily='monospace')

    ax.text(0, -0.18, f'Pollution Source : {pollution_source}',
            ha='center', fontsize=9,
            color='#80deea', fontfamily='monospace')

    ax.text(-1.15, -0.05, 'BAD\n0.0',  ha='center', fontsize=9,
            color='#ff4757', fontweight='bold')
    ax.text( 1.15, -0.05, 'GOOD\n1.0', ha='center', fontsize=9,
            color='#00e676', fontweight='bold')
    ax.text( 0,     1.12, '0.5',       ha='center', fontsize=9,
            color='#ffa502', fontweight='bold')

    plt.title(f'Water Quality Score Meter — Row {row_index}',
              fontsize=13, fontweight='bold',
              color=color_needle, pad=12)
    plt.tight_layout()
    plt.savefig(f'score_meter_row_{row_index}.png', dpi=150,
                facecolor='#0d1b2a', bbox_inches='tight')
    plt.show()

    # Graph 5 — Parameter Values
    bar_colors_sample = [
        '#e74c3c' if row[feat] > safe_limits[feat] else '#2ecc71'
        for feat in feature_cols
    ]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(short_labels, row.values,
                   color=bar_colors_sample, edgecolor='white', width=0.6)

    for bar, val in zip(bars, row.values):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(row.values) * 0.01,
                 f'{round(val, 2)}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.title(f'Row {row_index} — Parameter Values  '
              f'({"Good Water" if category == 1 else "Bad Water"})',
              fontsize=13, fontweight='bold')
    plt.ylabel('Value', fontsize=12)
    plt.xticks(fontsize=10)

    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', label='Within Safe Limit'),
        mpatches.Patch(facecolor='#e74c3c', label='Exceeds Safe Limit')
    ]
    plt.legend(handles=legend_elements, fontsize=10)
    plt.tight_layout()
    plt.savefig(f'sample_row_{row_index}.png', dpi=150)
    plt.show()

print("\n" + "=" * 50)
print(f"ALL {len(row_list)} ROWS DONE!")
print("All graphs also saved as PNG in your folder.")
print("=" * 50)