
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


raw = load_diabetes()
X = pd.DataFrame(raw.data, columns=raw.feature_names)
y = (raw.target > 140).astype(int)  # Turn it into a binary problem

df = X.copy()
df["Disease"] = y


print("🔹 df.head():")
print(df.head())


print("\n🔹 Before Scaling:")
print(df.describe())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n🔹 After Scaling:")
print(pd.DataFrame(X_scaled, columns=X.columns).describe())  # Screenshot this


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()  # Screenshot this


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)


print("\n🔹 Model Training Report:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


def predict_disease(*inputs):
    data = np.array(inputs).reshape(1, -1)
    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]
    return "🟢 No Disease Detected" if pred == 0 else "🔴 Disease Risk Detected"


input_components = [gr.Number(label=col) for col in X.columns]


with gr.Blocks() as demo:
    gr.Markdown("## 🩺 AI-Powered Disease Predictor")
    gr.Markdown("Enter patient test data to predict disease risk.")
    iface = gr.Interface(fn=predict_disease, inputs=input_components, outputs="text")
    iface.render()


demo.launch(debug=False)
