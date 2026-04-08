import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# 0. Load Dataset & Train Models
# ==========================================
try:
    data = pd.read_csv("Water_Quality_Assessment.csv")
    data.columns = data.columns.str.strip()
    
    # We assume 'Target' exists in realistic data, mapping it if necessary.
    if 'Target' not in data.columns:
        data['Target'] = np.where(data['BOD'] < 5, 1, 0)
except Exception as e:
    # Generate realistic mock data if CSV is not locally available
    np.random.seed(42)
    data = pd.DataFrame({
        'pH': np.random.uniform(5.5, 8.5, 500),
        'DO': np.random.uniform(2.0, 10.0, 500),
        'Turbidity': np.random.uniform(1.0, 15.0, 500),
        'Conductivity': np.random.uniform(300, 800, 500),
        'TDS': np.random.uniform(150, 500, 500),
        'Nitrate': np.random.uniform(0.5, 12.0, 500),
        'Phosphate': np.random.uniform(0.1, 4.0, 500),
        'BOD': np.random.uniform(0.5, 15.0, 500),
        'COD': np.random.uniform(5.0, 50.0, 500),
        'Coliform': np.random.randint(10, 1500, 500),
        'Ammonia': np.random.uniform(0.1, 5.0, 500),
    })
    # Correlate target with BOD for realistic modeling
    data['Target'] = np.where(data['BOD'] > 5, 0, 1) # 0: Bad, 1: Good

# Model Training
X = data.drop(columns=['Target'])
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
svm = SVC(random_state=42)
knn = KNeighborsClassifier()

rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
svm_pred = svm.predict(X_test)
knn_pred = knn.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
svm_acc = accuracy_score(y_test, svm_pred)
knn_acc = accuracy_score(y_test, knn_pred)

# Find Best Model
models_acc = {"Random Forest": rf_acc, "SVM": svm_acc, "KNN": knn_acc}
best_model_name = max(models_acc, key=models_acc.get)
best_acc = models_acc[best_model_name]

# ==========================================
# Pre-configure Outputs & Text
# ==========================================

# Selected Sample (Hardcoded as requested)
sample_params = ['pH', 'DO', 'BOD', 'COD', 'Ammonia']
sample_values = [6.8, 4.2, 12.0, 45.0, 1.8]
sample_df = pd.DataFrame([sample_values], columns=sample_params)

# Tab 3 Explanation
features_text = """
- **pH** → acidity level
- **DO** → oxygen level
- **BOD** → organic pollution
- **COD** → chemical pollution
- **Coliform** → bacteria contamination
- **Ammonia** → industrial waste
"""

# Tab 4 Processing
steps_text = """
### Workflow Steps:
1. **Load**
2. **Cleaning**
3. **Feature Selection**
4. **Score Calculation**
5. **Normalization**
6. **Category Assignment**
7. **Train-Test Split**
8. **Training Algorithm**
9. **Prediction**
"""

# Tab 6 Output
output_text = """
**Water Quality Score:** 0.63  
**Water Category:** Bad Water  
**Risk Level:** High  

**Issues:**
- High BOD  
- High Coliform  

**Pollution Source:**
- Domestic Sewage  

**Solutions:**
- Improve sewage treatment  
- Chlorination  
"""

# Tab 10 Code View
code_text = """
```python
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
svm_pred = svm.predict(X_test)
knn_pred = knn.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
svm_acc = accuracy_score(y_test, svm_pred)
knn_acc = accuracy_score(y_test, knn_pred)
```
"""

# Tab 11 Alert Logic
bod_val, cod_val, coliform_val = 12.0, 45.0, 1200
alerts_text = "### Logic Rules Evaluated:\n\n"
alerts_text += "- If BOD > 5 → Organic Pollution\n"
alerts_text += "- If COD > 10 → Industrial Pollution\n"
alerts_text += "- If Coliform > 1000 → Bacterial Risk\n\n"

if bod_val > 5 or cod_val > 10 or coliform_val > 1000:
    alerts_text += "## ⚠️ High Risk Water Detected\n"

# ==========================================
# Generate Plotly Visualizations (Tab 7)
# ==========================================

# 1. Radar Chart (go.Scatterpolar)
fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
      r=sample_values,
      theta=sample_params,
      fill='toself',
      name='Selected Sample'
))
fig_radar.update_layout(title="Water Parameter Radar Analysis")

# 2. Donut Chart (px.pie)
good_count = len(data[data['Target'] == 1])
bad_count = len(data[data['Target'] == 0])
fig_donut = px.pie(names=['Good Water', 'Bad Water'], values=[good_count, bad_count], hole=0.4, title="Water Category Distribution")

# 3. Correlation Heatmap (px.imshow)
corr_matrix = data.corr()
fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Parameter Correlation Heatmap")

# 4. Horizontal Bar Chart (px.bar) - Sorted
df_bar = pd.DataFrame({'Parameter': sample_params, 'Value': sample_values}).sort_values(by='Value', ascending=True)
fig_bar = px.bar(df_bar, x='Value', y='Parameter', orientation='h', title="Sample Parameter Values")

# ==========================================
# Generate Comparison Plot (Tab 9)
# ==========================================
fig_compare = px.bar(
    x=["Random Forest", "SVM", "KNN"],
    y=[rf_acc, svm_acc, knn_acc],
    title="Model Accuracy Comparison",
    labels={'x': 'Models', 'y': 'Accuracy'},
    color=["Random Forest", "SVM", "KNN"]
)
# Force y-axis limit for better visual distinction
fig_compare.update_layout(yaxis_range=[0.5, 1.0])

# ==========================================
# Build Gradio UI
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as app:
    
    # 1. Header
    gr.Markdown("<h1 style='text-align: center; color: #1976d2;'>Water Quality Prediction System</h1>")
    
    with gr.Tabs():
        # 2. Dataset
        with gr.TabItem("1. Dataset"):
            gr.Dataframe(data.head(50), interactive=True)
            
        # 3. Dataset Explanation
        with gr.TabItem("2. Dataset Explanation"):
            with gr.Card():
                gr.Markdown(features_text)
                
        # 4. Processing Steps
        with gr.TabItem("3. Processing Steps"):
            with gr.Card():
                gr.Markdown(steps_text)
                
        # 5. Sample Prediction
        with gr.TabItem("4. Sample Prediction"):
            gr.Markdown("### Selected Test Sample Row")
            gr.Dataframe(sample_df)
            
        # 6. Final Output
        with gr.TabItem("5. Final Output"):
            with gr.Card():
                gr.Markdown(output_text)

        # 7. Visualization
        with gr.TabItem("6. Visualization"):
            with gr.Row():
                gr.Plot(fig_radar)
                gr.Plot(fig_donut)
            with gr.Row():
                gr.Plot(fig_bar)
            with gr.Row():
                gr.Plot(fig_heatmap)
                
        # 8. Model Performance
        with gr.TabItem("7. Model Performance"):
            with gr.Card():
                gr.Markdown("### Dynamically Calculated Accuracies")
                gr.Markdown(f"**Random Forest Accuracy:** {rf_acc:.4f}\n\n**SVM Accuracy:** {svm_acc:.4f}\n\n**KNN Accuracy:** {knn_acc:.4f}")

        # 9. Model Comparison
        with gr.TabItem("8. Model Comparison"):
            with gr.Row():
                with gr.Column():
                    df_compare = pd.DataFrame({"Model": ["Random Forest", "SVM", "KNN"], "Accuracy": [rf_acc, svm_acc, knn_acc]})
                    gr.Dataframe(df_compare)
                    gr.Markdown(f"## 🏆 Best Model: {best_model_name} with accuracy {best_acc:.2%}")
                with gr.Column():
                    gr.Plot(fig_compare)

        # 10. ML Code View
        with gr.TabItem("9. ML Code View"):
            gr.Markdown(code_text)

        # 11. Smart Alert System
        with gr.TabItem("10. Smart Alert System"):
            with gr.Card():
                gr.Markdown(alerts_text)

if __name__ == "__main__":
    app.launch()
