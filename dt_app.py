import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Decision Tree Classifier",
    layout="wide",
)

st.title("Decision Tree Classification Web App")
st.write(
    "This app trains a Decision Tree classifier on your dataset and "
    "shows the confusion matrix in an interactive table, and visualises the tree using **Graphviz**."
)

# -----------------------------
# Data upload
# -----------------------------
st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Select target & features
# -----------------------------
st.sidebar.header("Columns Selection")

target_col = st.sidebar.selectbox(
    "Select target column (label)",
    df.columns,
    index=len(df.columns) - 1,  # default: last column
)

feature_cols = st.sidebar.multiselect(
    "Select feature columns",
    [c for c in df.columns if c != target_col],
    default=[c for c in df.columns if c != target_col],
)

if not feature_cols:
    st.error("Please select at least one feature column.")
    st.stop()

X = df[feature_cols]
y = df[target_col]

class_names = [str(c) for c in sorted(y.unique())]

# -----------------------------
# Model parameters (Decision Tree)
# -----------------------------
st.sidebar.header("Decision Tree Parameters")

test_size = st.sidebar.slider(
    "Test size (proportion for test set)",
    min_value=0.1,
    max_value=0.5,
    value=0.25,
    step=0.05,
)

criterion = st.sidebar.selectbox(
    "Criterion",
    options=["gini", "entropy"],
    index=1,  # default to entropy
)

max_depth = st.sidebar.number_input(
    "Max depth (0 = None)",
    min_value=0,
    value=0,
    step=1,
)

min_samples_split = st.sidebar.number_input(
    "min_samples_split",
    min_value=2,
    value=2,
    step=1,
)

min_samples_leaf = st.sidebar.number_input(
    "min_samples_leaf",
    min_value=1,
    value=1,
    step=1,
)

if max_depth == 0:
    max_depth_param = None
else:
    max_depth_param = int(max_depth)

# -----------------------------
# Train–test split & model training
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

model = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth_param,
    min_samples_split=int(min_samples_split),
    min_samples_leaf=int(min_samples_leaf),
    random_state=42,
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
st.subheader("Model Evaluation")

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=class_names)
acc = accuracy_score(y_test, y_pred)

# Interactive confusion matrix table
cm_df = pd.DataFrame(
    cm,
    index=[f"True {c}" for c in class_names],
    columns=[f"Pred {c}" for c in class_names],
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Confusion Matrix (interactive)**")
    st.dataframe(cm_df)  # interactive table

with col2:
    st.markdown("**Accuracy**")
    st.metric("Accuracy", f"{acc:.3f}")

# Classification report as interactive table
report_dict = classification_report(
    y_test, y_pred, output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report_dict).T

st.markdown("**Classification Report**")
st.dataframe(report_df.style.format("{:.3f}", na_rep="-"))

# -----------------------------
# Visualise Decision Tree (Graphviz)
# -----------------------------
st.subheader("Decision Tree Visualisation (Graphviz)")

# export_graphviz -> DOT string -> graphviz_chart
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=feature_cols,
    class_names=class_names,
    filled=True,
    rounded=True,
    special_characters=True,
)

st.graphviz_chart(dot_data)

st.caption(
    "The tree above is generated using `sklearn.tree.export_graphviz` and rendered "
    "in Streamlit using `st.graphviz_chart()`."
)
