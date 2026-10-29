"""AI Assistant Page — Primary conversational interface."""
import json
import time
import streamlit as st
from typing import Optional

from database.repository import ExperimentRepository
from orchestration.workflow_orchestrator import WorkflowOrchestrator
from services.chat_service import ChatService


def _init_session():
    """Initialize session state defaults."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "ai_experiment_id" not in st.session_state:
        st.session_state.ai_experiment_id = None
    if "ai_generated_code" not in st.session_state:
        st.session_state.ai_generated_code = ""
    if "ai_execution_result" not in st.session_state:
        st.session_state.ai_execution_result = None
    if "ai_pending_problem" not in st.session_state:
        st.session_state.ai_pending_problem = ""
    if "ai_running" not in st.session_state:
        st.session_state.ai_running = False


def _add_message(role: str, content: str):
    st.session_state.chat_history.append({"role": role, "content": content})


def _render_chat_history():
    """Render all chat messages."""
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])


def _render_metrics(metrics: dict):
    """Render key metrics in columns."""
    if not metrics:
        return
    metric_items = list(metrics.items())
    cols = st.columns(min(len(metric_items), 4))
    for i, (k, v) in enumerate(metric_items[:4]):
        with cols[i]:
            label = k.replace("_", " ").title()
            if isinstance(v, float):
                st.metric(label, f"{v:.4f}")
            else:
                st.metric(label, str(v))


def _render_plots(results_json: dict):
    """Render interactive Plotly charts based on the experiment results data."""
    if not results_json:
        return
    from services.visualization_service import VisualizationService

    # 1. Gradient descent / optimization convergence
    if "history" in results_json and isinstance(results_json["history"], list):
        losses = [h.get("loss") for h in results_json["history"] if "loss" in h]
        if losses:
            fig = VisualizationService.plot_convergence_curve(losses)
            st.plotly_chart(fig, use_container_width=True)

    # 2. Activation functions comparison
    elif "activations" in results_json and "x_values" in results_json:
        fig = VisualizationService.plot_activation_functions(
            results_json["x_values"], results_json["activations"]
        )
        st.plotly_chart(fig, use_container_width=True)

    # 3. K-Means clustering space
    elif "X" in results_json and "labels" in results_json:
        fig = VisualizationService.plot_clustering_2d(
            results_json["X"], results_json["labels"], results_json.get("centers")
        )
        st.plotly_chart(fig, use_container_width=True)
        if "elbow_data" in results_json:
            ed = results_json["elbow_data"]
            fig_elbow = VisualizationService.plot_elbow_curve(ed.get("k_values", []), ed.get("inertias", []))
            st.plotly_chart(fig_elbow, use_container_width=True)



def _render_followups(followups: list):
    """Render clickable follow-up suggestion buttons."""
    if not followups:
        return
    st.markdown("**💡 Try asking:**")
    cols = st.columns(min(len(followups), 3))
    for i, suggestion in enumerate(followups[:3]):
        with cols[i]:
            if st.button(f"→ {suggestion[:40]}", key=f"followup_{i}_{hash(suggestion)}",
                         use_container_width=True):
                st.session_state.ai_pending_problem = suggestion
                st.rerun()


def render_ai_assistant(
    repo: ExperimentRepository,
    orchestrator: WorkflowOrchestrator,
    on_navigate=None,
):
    """Main AI Assistant page renderer."""
    _init_session()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1f3e 0%, #0d3b6e 100%);
                padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
                border-left: 4px solid #4facfe;'>
        <h2 style='color: #4facfe; margin: 0 0 0.5rem 0;'>🤖 AI Lab Assistant</h2>
        <p style='color: #a0b4d0; margin: 0;'>
            Describe any ML/DL problem in natural language — I'll analyze, generate code, execute it, and explain the results.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Toolbar ───────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        st.caption(f"💬 {len(st.session_state.chat_history)} messages in session")
    with col_b:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.ai_experiment_id = None
            st.session_state.ai_generated_code = ""
            st.session_state.ai_execution_result = None
            st.rerun()
    with col_c:
        if st.button("📂 My Experiments", use_container_width=True):
            if on_navigate:
                on_navigate("experiment_history")

    st.divider()

    # ── Welcome message if empty ─────────────────────────────────────────────
    if not st.session_state.chat_history:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("""
👋 **Welcome to your Autonomous AI Lab Assistant!**

I can help you with:
- 🧪 **Run complete ML/DL experiments** — just describe your problem
- 💻 **Generate & execute code** — gradient descent, K-means, neural nets, and more
- 📚 **Explain concepts** — overfitting, activation functions, backpropagation...
- 🔧 **Improve results** — hyperparameter tuning, regularization, ensembles
- 📊 **Compare algorithms** — side-by-side analysis

**Try one of these:**
            """)

            # Quick start buttons
            examples = [
                "Implement gradient descent and visualize convergence",
                "Compare activation functions (sigmoid, ReLU, tanh)",
                "K-means clustering demo with synthetic data",
                "Build a classification model for tabular data",
                "Explain overfitting and how to prevent it",
            ]
            for example in examples:
                if st.button(f"→ {example}", key=f"welcome_{hash(example)}",
                             use_container_width=True):
                    st.session_state.ai_pending_problem = example
                    st.rerun()

    # ── Chat History ─────────────────────────────────────────────────────────
    _render_chat_history()

    # ── Current Experiment Panel ──────────────────────────────────────────────
    if st.session_state.ai_experiment_id and st.session_state.ai_execution_result:
        result = st.session_state.ai_execution_result
        exec_data = result.get("execution_result", {})
        metrics = exec_data.get("metrics", {})

        if metrics:
            st.markdown("---")
            st.markdown("### 📊 Latest Results")
            _render_metrics(metrics)

        results_json = exec_data.get("results_json", {})
        if results_json:
            _render_plots(results_json)

        if st.session_state.ai_generated_code:
            with st.expander("📄 View Generated Code", expanded=False):
                st.code(st.session_state.ai_generated_code, language="python")

        exec_info = exec_data.get("execution_result", exec_data)
        stdout = exec_info.get("stdout", "")
        if stdout:
            with st.expander("📋 Execution Log", expanded=False):
                # Filter out RESULTS_JSON line for cleaner display
                log_lines = [l for l in stdout.split("\n") if not l.startswith("RESULTS_JSON:")]
                st.code("\n".join(log_lines), language="text")

    # ── Handle pending auto-submit (from button clicks) ───────────────────────
    pending = st.session_state.pop("ai_pending_problem", "") if "ai_pending_problem" in st.session_state else ""

    # ── Chat Input ────────────────────────────────────────────────────────────
    user_input = st.chat_input("Describe your ML problem or ask a question...")

    # Process either real input or pending auto-submit
    message_to_process = user_input or pending
    if message_to_process:
        _handle_message(message_to_process, repo, orchestrator)


def _handle_message(
    message: str,
    repo: ExperimentRepository,
    orchestrator: WorkflowOrchestrator,
):
    """Process a user message through the chat service and handle experiment execution."""
    chat_service = ChatService()

    # Add user message
    _add_message("user", message)
    with st.chat_message("user", avatar="👤"):
        st.markdown(message)

    # Get current experiment context
    experiment_context = None
    if st.session_state.ai_experiment_id:
        try:
            exp = repo.get_experiment(st.session_state.ai_experiment_id)
            if exp:
                exec_results = json.loads(getattr(exp, "execution_results_json", None) or "{}")
                experiment_context = {
                    "experiment_id": exp.id,
                    "status": exp.status,
                    "metrics": exec_results.get("metrics", {}),
                    "generated_code": st.session_state.ai_generated_code,
                }
        except Exception:
            pass

    # Process message
    chat_response = chat_service.process_message(
        message,
        experiment_context=experiment_context,
        chat_history=st.session_state.chat_history[:-1],
    )

    response_text = chat_response["response_text"]
    action = chat_response.get("action", "")
    intent = chat_response.get("intent", "")
    requires_experiment = chat_response.get("requires_experiment", False)
    followups = chat_response.get("suggested_followups", [])

    # Display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response_text)

        # If this is an experiment request, offer a run button
        if requires_experiment or action == "start_experiment" or intent == "new_experiment":
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("▶️ Run Experiment Now", key=f"run_{hash(message)}",
                             type="primary", use_container_width=True):
                    _run_experiment(message, repo, orchestrator)
            with col2:
                if st.button("📝 Open in Workspace", key=f"workspace_{hash(message)}",
                             use_container_width=True):
                    # Create experiment and navigate
                    try:
                        exp = repo.create_experiment(
                            name=f"Experiment: {message[:40]}",
                            problem_statement=message,
                        )
                        st.session_state["current_experiment_id"] = exp.id
                        st.info("Experiment created! Use the sidebar to navigate to 'New Experiment'.")
                    except Exception as e:
                        st.error(f"Could not create experiment: {e}")

        _render_followups(followups)

    _add_message("assistant", response_text)
    st.rerun()


def _run_experiment(
    problem_statement: str,
    repo: ExperimentRepository,
    orchestrator: WorkflowOrchestrator,
):
    """Execute a complete experiment autonomously with real-time progress."""

    # Create experiment record
    try:
        exp = repo.create_experiment(
            name=f"AI: {problem_statement[:50]}",
            problem_statement=problem_statement,
        )
        st.session_state.ai_experiment_id = exp.id
    except Exception as e:
        st.error(f"Failed to create experiment: {e}")
        return

    progress_messages = []

    def progress_callback(stage: str, message: str):
        progress_messages.append(f"[{stage.upper()}] {message}")

    with st.status("🚀 Running experiment...", expanded=True) as status:
        try:
            # Stage progress display
            st.write("🧠 Analyzing problem statement...")
            time.sleep(0.1)

            result = orchestrator.execute_complete_experiment(
                experiment_id=exp.id,
                problem_statement=problem_statement,
                timeout=120,
                progress_callback=progress_callback,
            )

            understanding = result.get("problem_understanding", {})
            generated_code = result.get("generated_code", "")
            exec_result = result.get("execution_result", {})
            dataset_required = result.get("dataset_required", True)

            st.session_state.ai_generated_code = generated_code
            st.session_state.ai_execution_result = result

            # Show progress
            category = understanding.get("experiment_category", "tabular_ml")
            st.write(f"📋 Category: `{category}` | Dataset required: `{dataset_required}`")

            if not dataset_required:
                st.write("⚡ Dataset-free experiment — skipping dataset search")
            else:
                st.write("🔍 Searched for datasets...")
                st.write("📊 Analyzed data profile...")

            st.write("💻 Generated experiment code...")

            inner = exec_result.get("execution_result", exec_result)
            success = inner.get("success", exec_result.get("success", False))
            metrics = inner.get("metrics", exec_result.get("metrics", {}))

            if success:
                st.write("✅ Execution successful!")
                status.update(label="✅ Experiment completed!", state="complete")

                # Build response message
                summary = inner.get("summary", exec_result.get("summary", ""))
                duration = inner.get("duration_seconds", exec_result.get("execution_result", {}).get("duration_seconds", 0))

                response_parts = [
                    f"✅ **Experiment Complete!** (ID: `{exp.id[:8]}`)\n",
                    f"**Problem:** {problem_statement[:80]}...\n" if len(problem_statement) > 80 else f"**Problem:** {problem_statement}\n",
                    f"**Category:** {category.replace('_', ' ').title()}\n",
                    f"**Duration:** {duration:.2f}s\n",
                ]
                if metrics:
                    response_parts.append("\n**📊 Results:**")
                    for k, v in list(metrics.items())[:6]:
                        label = k.replace("_", " ").title()
                        val = f"{v:.4f}" if isinstance(v, float) else str(v)
                        response_parts.append(f"  - {label}: **{val}**")
                if summary:
                    response_parts.append(f"\n**Summary:** {summary}")
                response_parts.append(
                    f"\n\n💾 Experiment saved with ID `{exp.id[:8]}`. "
                    f"Find it in **📂 Experiment History** to revisit, compare, or export."
                )

                response_text = "\n".join(response_parts)

            else:
                error_type = inner.get("error_type", "")
                stderr = inner.get("stderr", "")[:300]
                status.update(label="⚠️ Execution encountered issues", state="error")
                response_text = (
                    f"⚠️ **Experiment execution had issues.**\n\n"
                    f"**Error type:** `{error_type}`\n"
                    f"**Details:**\n```\n{stderr}\n```\n\n"
                    f"The generated code has been saved. You can view and edit it in the "
                    f"**Experiment Workspace** page."
                )

        except Exception as e:
            status.update(label="❌ Experiment failed", state="error")
            response_text = (
                f"❌ **Experiment failed with an unexpected error:**\n\n"
                f"```\n{str(e)}\n```\n\n"
                f"Please try again with a different problem statement or check your configuration."
            )
            logger.error(f"Experiment execution error: {e}", exc_info=True)

    # Add final result to chat
    _add_message("assistant", response_text)
    st.rerun()


import logging
logger = logging.getLogger(__name__)
