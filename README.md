# 🧪 Autonomous AI Lab Assistant Using Deep Learning

An autonomous, agentic AI laboratory assistant designed to help researchers, students, and machine learning practitioners conduct end-to-end Machine Learning (ML) and Deep Learning (DL) experiments from initial problem conception to empirical evaluation, persistent archiving, and formal report generation.

---

## 🌟 Key Features

1. **Natural Language Problem Formulation**
   - Enter ML/DL problems in plain English (e.g. *"Identify breast cancer tumor malignancy from cell biopsy dimensions"* or *"Forecast housing prices based on district demographics"*).
   - **Problem Understanding Agent** decomposes the problem into technical task categories, objectives, expected inputs/outputs, target variables, evaluation metrics, and potential risk vectors.

2. **Autonomous Strategic Experiment Planning**
   - **Experiment Planning Agent** synthesizes a formal execution strategy covering dataset requirements, preprocessing steps, baseline model families, validation protocols, and resource considerations.

3. **Multi-Source Dataset Discovery & Ranking**
   - **Multi-Source Ingestion**: Query verified benchmark repositories including **UCI Machine Learning Repository**, **OpenML**, **Hugging Face Hub**, **Kaggle**, or upload custom local CSV datasets.
   - **Dataset Comparison Agent**: Computes objective 0–100 suitability scores and provides comparative matrices.

4. **Deep Data Profiling & Quality / Leakage Audit**
   - Statistical summaries, dtypes, numerical distributions, categorical counts, and target class balance.
   - **Data Quality Agent**: Audits for missing values, duplicate instances, zero-variance columns, and identifier leakage (ID columns).

5. **Leakage-Free Preprocessing & Model Recommendation**
   - Imputation and scaling transformers are fit strictly on training splits.
   - **Model Selection Agent**: Recommends candidate algorithms (Random Forest, Gradient Boosting, Regularized Linear / Ridge, Multi-Layer Perceptrons) with explanations.

6. **Code Generation & Static AST Validation**
   - Synthesizes readable, self-contained Python pipelines matching the exact dataset schema.
   - **Code Validation Agent**: Validates Abstract Syntax Trees (AST), library dependencies, and security boundaries.

7. **Safe Local Training & Google Colab GPU Export**
   - **Local Execution**: Run training directly with real-time log capturing and empirical evaluation.
   - **Google Colab Notebooks**: Exports valid `.ipynb` notebooks containing GPU detection, environment setup, model definition, training loops, and an exportable results payload.
   - **Colab Result Import**: Import `experiment_results.json` generated in Colab back into the laboratory dashboard.

8. **Unbiased Empirical Evaluation & Visualizations**
   - Real test metrics (Accuracy, F1-Score, Precision, Recall, Confusion Matrix, ROC-AUC for classification; R², RMSE, MAE for regression).
   - Interactive Plotly visualizations (Confusion Matrix Heatmaps, Feature Importances, Actual vs Predicted, Residuals, Metric Comparisons).

9. **Error Analysis & Actionable Improvements**
   - **Error Analysis Agent** identifies overfitting, underfitting, and class imbalance disparities.
   - **Improvement Agent** recommends concrete next steps and creates new versioned experiment iterations.

10. **Permanent Database Persistence & Experiment Memory**
    - Every experiment, configuration snapshot, execution run, model checkpoint, plot, and log is permanently stored in **SQLite** (`data/lab_assistant.db`) via SQLAlchemy.
    - Experiments survive browser refreshes, application restarts, and computer reboots.
    - Full version history (v1, v2, v3), duplication, workflow resumption, run comparisons, and report re-downloading.

---

## 🏛️ Project Architecture

```
autonomous-ai-lab-assistant/
│
├── app.py                     # Streamlit application entrypoint & routing
├── requirements.txt           # Project dependencies
├── README.md                  # Complete documentation and setup guide
├── .env.example               # Environment variables template
├── .gitignore                 # Version control exclusions
│
├── config/                    # Centralized settings & path management
│   ├── __init__.py
│   └── settings.py
│
├── schemas/                   # Pydantic data contracts & schemas
│   ├── __init__.py
│   ├── experiment_schema.py
│   ├── dataset_schema.py
│   └── result_schema.py
│
├── database/                  # Persistent SQLAlchemy ORM layer
│   ├── __init__.py
│   ├── connection.py          # SQLite engine & scoped sessions
│   ├── models.py              # Experiment, Version, Run, Artifact models
│   ├── repository.py          # Full CRUD operations & queries
│   └── initialization.py      # Safe schema creation
│
├── agents/                    # Specialized autonomous agents
│   ├── __init__.py
│   ├── base_agent.py
│   ├── problem_understanding_agent.py
│   ├── experiment_planning_agent.py
│   ├── dataset_search_agent.py
│   ├── dataset_comparison_agent.py
│   ├── dataset_analysis_agent.py
│   ├── data_quality_agent.py
│   ├── preprocessing_agent.py
│   ├── model_selection_agent.py
│   ├── code_generation_agent.py
│   ├── code_validation_agent.py
│   ├── experiment_execution_agent.py
│   ├── evaluation_agent.py
│   ├── error_analysis_agent.py
│   ├── improvement_agent.py
│   └── report_generation_agent.py
│
├── orchestration/             # Workflow coordination & state machine
│   ├── __init__.py
│   ├── workflow_state.py
│   ├── task_manager.py
│   └── workflow_orchestrator.py
│
├── services/                  # Core business logic services
│   ├── __init__.py
│   ├── experiment_service.py
│   ├── dataset_service.py
│   ├── preprocessing_service.py
│   ├── model_service.py
│   ├── training_service.py
│   ├── evaluation_service.py
│   ├── visualization_service.py
│   ├── notebook_service.py
│   ├── report_service.py
│   ├── artifact_service.py
│   └── llm_service.py
│
├── integrations/              # Third-party platform adapters
│   ├── __init__.py
│   ├── huggingface_provider.py
│   ├── openml_provider.py
│   ├── uci_provider.py
│   ├── kaggle_provider.py
│   └── colab_exporter.py
│
├── frontend/                  # Streamlit UI & pages
│   ├── __init__.py
│   ├── styles.py              # Custom CSS styling (navy/charcoal/teal)
│   ├── components.py          # Reusable cards, badges, empty states
│   └── pages/
│       ├── __init__.py
│       ├── dashboard.py
│       ├── new_experiment.py
│       ├── dataset_explorer.py
│       ├── experiment_workspace.py
│       ├── model_laboratory.py
│       ├── previous_experiments.py
│       ├── experiment_comparison.py
│       ├── reports.py
│       ├── integrations.py
│       ├── settings.py
│       └── help_documentation.py
│
├── templates/reports/         # Report templates
│   └── report_template.md
│
├── tests/                     # Automated test suite
│   ├── __init__.py
│   ├── test_database.py
│   ├── test_datasets.py
│   ├── test_workflow.py
│   ├── test_notebook_export.py
│   └── test_experiment_persistence.py
│
├── data/                      # Local storage (created automatically)
│   ├── lab_assistant.db       # Persistent SQLite database
│   ├── datasets/              # User uploads & cached downloads
│   └── cache/
│
└── artifacts/                 # Serialized checkpoints & reports
    ├── models/                # Trained model checkpoints (.joblib)
    ├── plots/                 # Chart data (.json)
    ├── reports/               # Compiled reports (.md, .html, .json)
    ├── notebooks/             # Generated Colab notebooks (.ipynb)
    └── logs/                  # Execution output logs (.txt)
```

---

## 💻 Beginner Setup Instructions (Windows PowerShell)

Follow these simple steps in Windows PowerShell to set up and run the application:

### Step 1: Open Terminal in Project Folder
Open PowerShell or your code editor terminal in the project directory:
```powershell
cd "c:\Users\gnana\Desktop\autonomous ai lab assisstant"
```

### Step 2: Create a Python Virtual Environment
```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
> **Note on PowerShell Execution Policy**: If Windows displays an execution policy error when activating, you do not need to change administrative security policies. Simply invoke the virtual environment's Python executable directly:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Step 4: Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 5: (Optional) Configure External Credentials
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
*(No API keys are required for the application to function. Standard benchmark datasets and local rule-based intelligence work out of the box.)*

### Step 6: Start the Application
```powershell
python -m streamlit run app.py
```

### Step 7: Open the Application in your Browser
Streamlit will launch and display the local URL:
```
Local URL: http://localhost:8501
```
Open `http://localhost:8501` in Chrome, Edge, or Firefox.

---

## 🧪 Running the Automated Test Suite

Run the full automated test suite using `pytest`:
```powershell
python -m pytest tests/ -v
```
All 8 test suites (Database CRUD, Dataset Ingestion, Orchestrator Pipelines, Notebook Generation, and the Mandatory 13-Point Experiment Persistence Cycle) will execute.

---

## 🔒 Security & Data Integrity
- **No Arbitrary Code Execution**: Code generation is statically parsed via Python's Abstract Syntax Tree (`ast.parse`) before execution.
- **Safe Subprocess Limits**: Local execution runs with timeout controls and resource safeguards.
- **Credential Protection**: API keys are masked in the UI and never serialized into Colab notebooks or exported reports.
- **Data Leakage Prevention**: Transformers are fit strictly on training splits.

---

## 📜 License
MIT License. Built for autonomous AI research and reproducible machine learning.
