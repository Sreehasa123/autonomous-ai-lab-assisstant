"""Help and Documentation Page."""
import streamlit as st
from frontend.components import render_section_header

def render_help_documentation(on_navigate):
    """User documentation, workflow instructions, and beginner guide."""
    render_section_header("Help & Documentation", "Complete user manual, system architecture, and beginner setup instructions for the Autonomous AI Lab Assistant.")

    st.markdown("""
### 🌟 Overview
The **Autonomous AI Lab Assistant** is a persistent, end-to-end laboratory workspace that guides researchers, students, and engineers through the complete lifecycle of Machine Learning and Deep Learning experimentation:

1. **Problem Formulation**: Enter your problem in natural language.
2. **Autonomous Planning**: The agents determine task type, objective, candidate metrics, model families, and risks.
3. **Multi-Source Dataset Discovery**: Search UCI ML Repository, OpenML, Hugging Face, Kaggle, or upload custom CSV datasets.
4. **Data Quality & Leakage Audit**: Inspect missing values, constant columns, duplicate instances, and identifier leakage.
5. **Preprocessing & Model Selection**: Configure leakage-free train/test partitioning, standard scaling, and candidate algorithms.
6. **Code Generation & AST Validation**: Generates verified Python scripts tailored to your dataset schema.
7. **Safe Execution**: Run training locally or export valid Jupyter notebooks (`.ipynb`) for Google Colab GPU training.
8. **Unbiased Empirical Evaluation**: Compute exact test metrics (Accuracy, F1, Precision, Recall, Confusion Matrix, R², RMSE, MAE).
9. **Permanent Persistence & Versioning**: All experiments, runs, checkpoints, and reports are saved in SQLite and survive application restarts!

---

### 🚀 Quick Start Guide (Windows PowerShell)

#### Step 1: Clone or navigate to the project directory
```powershell
cd "c:\\Users\\gnana\\Desktop\\autonomous ai lab assisstant"
```

#### Step 2: Create a virtual environment
```powershell
python -m venv .venv
```

#### Step 3: Activate the virtual environment
```powershell
.\\.venv\\Scripts\\Activate.ps1
```
*(If PowerShell script execution is restricted on your system, you can directly run Python from the virtual environment without modifying system policies)*:
```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app.py
```

#### Step 4: Install dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### Step 5: Start the Streamlit Application
```powershell
python -m streamlit run app.py
```
Open the local URL displayed in the terminal: `http://localhost:8501`.

---

### 🛡️ Permanent Database & Experiment Memory
- Every experiment you create is saved to SQLite (`data/lab_assistant.db`).
- Trained models are serialized with joblib into `artifacts/models/`.
- Interactive plots and execution logs are stored in `artifacts/plots/` and `artifacts/logs/`.
- Generated Google Colab notebooks are stored in `artifacts/notebooks/`.
- Compiled laboratory reports are stored in `artifacts/reports/`.
- You can close Streamlit, restart your computer, or refresh your browser at any time without losing your progress.
""")
