"""Chat Service — Intent detection + conversational processing for the AI Assistant."""
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ChatService:
    """
    Processes natural-language messages from the user.
    Detects intent, generates responses, and decides what action to take.
    Works entirely rule-based (no LLM required) but uses LLM when configured.
    """

    # Intent patterns — ordered by priority
    INTENT_PATTERNS = [
        # Conversation/question patterns
        ("explain",     [r"what\s+is\s+", r"explain\s+", r"how\s+does\s+", r"what\s+are\s+",
                         r"define\s+", r"tell\s+me\s+about\s+", r"describe\s+", r"why\s+is\s+"]),
        ("compare",     [r"compare\s+", r"difference\s+between\s+", r"vs\s+", r"versus\s+",
                         r"which\s+is\s+better\s+", r"pros\s+and\s+cons\s+"]),
        ("improve",     [r"improve\s+", r"optimize\s+", r"tune\s+", r"better\s+", r"enhance\s+",
                         r"suggestions\s+for\s+", r"how\s+to\s+improve\s+"]),
        ("resume",      [r"resume\s+", r"continue\s+", r"reopen\s+", r"load\s+experiment\s+",
                         r"previous\s+experiment"]),
        ("save",        [r"save\s+", r"store\s+this\s+", r"keep\s+this\s+"]),
        ("generate_code", [r"generate\s+code\s+", r"write\s+code\s+", r"create\s+code\s+",
                           r"code\s+for\s+", r"implement\s+code\s+"]),
        ("execute",     [r"run\s+", r"execute\s+", r"start\s+", r"train\s+", r"go\s+ahead\s+"]),
        # Full experiment request patterns
        ("new_experiment", [
            r"build\s+(a|an)\s+", r"develop\s+(a|an)\s+", r"create\s+(a|an)\s+model",
            r"implement\s+(a|an)\s+", r"train\s+(a|an)\s+", r"classify\s+", r"predict\s+",
            r"cluster\s+", r"detect\s+", r"forecast\s+", r"analyze\s+(data|dataset)",
            r"gradient\s+descent", r"compare\s+activation", r"k.?means",
        ]),
    ]

    # Experiment detection patterns (subset of new_experiment that clearly need execution)
    EXPERIMENT_PATTERNS = [
        r"build\s+(a|an)\s+.*(model|classifier|regressor|network|system)",
        r"develop\s+(a|an)\s+.*(model|cnn|rnn|lstm|classifier)",
        r"implement\s+.*(gradient|backprop|perceptron|kmeans|k-means)",
        r"train\s+(a|an)\s+", r"classify\s+", r"predict\s+house|car|price|churn",
        r"forecast\s+", r"detect\s+(anomal|fraud|disease)",
        r"compare\s+activation\s+function",
        r"gradient\s+descent.*minim",
        r"k.?means.*cluster",
        r"visualize\s+convergence",
    ]

    # Topics for explanation
    EXPLANATION_TOPICS = {
        "gradient descent": """**Gradient Descent** is an optimization algorithm that minimizes a function by iteratively stepping in the direction opposite to the gradient.

**Key variants:**
- **Batch GD**: uses all samples per update (stable but slow)
- **Stochastic GD (SGD)**: one sample per update (fast but noisy)
- **Mini-batch GD**: small batches (balance of speed and stability)

**Learning rate** controls the step size. Too large → divergence; too small → slow convergence.""",

        "activation function": """**Activation functions** introduce non-linearity into neural networks, enabling them to learn complex patterns.

| Function | Range | Pros | Cons |
|----------|-------|------|------|
| **Sigmoid** | (0,1) | Probabilistic output | Vanishing gradient |
| **ReLU** | [0,∞) | Fast, no vanishing gradient | Dead neurons |
| **Tanh** | (-1,1) | Zero-centered | Vanishing gradient |
| **Leaky ReLU** | (-∞,∞) | Fixes dead ReLU | Requires tuning α |
| **Swish** | (-∞,∞) | Often outperforms ReLU | Computationally heavier |""",

        "overfitting": """**Overfitting** occurs when a model learns the training data too well, including its noise, failing to generalize.

**Solutions:**
- **Regularization** (L1/L2) — penalizes large weights
- **Dropout** — randomly drops neurons during training
- **Data augmentation** — increases training diversity
- **Early stopping** — stops training when validation loss rises
- **Cross-validation** — better estimate of true performance""",

        "cnn": """**Convolutional Neural Networks (CNNs)** are specialized for grid-like data (images).

**Architecture layers:**
1. **Conv layer**: applies learned filters to detect features
2. **Activation (ReLU)**: non-linearity
3. **Pooling**: reduces spatial dimensions
4. **Fully connected**: classification head

Popular architectures: LeNet, AlexNet, VGG, ResNet, EfficientNet""",

        "random forest": """**Random Forest** is an ensemble of decision trees, each trained on a random subset of data and features.

**How it works:**
1. Build N decision trees on bootstrapped samples (bagging)
2. At each split, consider only m random features
3. Aggregate predictions via majority vote (classification) or mean (regression)

**Advantages**: robust to overfitting, handles missing values, provides feature importance.""",
    }

    def __init__(self):
        try:
            from services.llm_service import LLMService
            self.llm_service = LLMService()
        except Exception:
            self.llm_service = None

    def process_message(
        self,
        user_message: str,
        experiment_context: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message and return a structured response.

        Returns:
            {
                "response_text": str,
                "intent": str,
                "action": str,
                "requires_experiment": bool,
                "generated_code": str,
                "suggested_followups": List[str],
            }
        """
        message = user_message.strip()
        if not message:
            return self._build_response(
                "Please type a message or problem statement to get started.",
                intent="unknown",
            )

        # Detect intent
        intent = self._detect_intent(message)

        # Handle by intent
        if intent == "explain":
            return self._handle_explanation(message)

        elif intent == "compare":
            return self._handle_comparison(message)

        elif intent == "improve":
            return self._handle_improvement(message, experiment_context)

        elif intent == "new_experiment" or self._is_experiment_request(message):
            return self._handle_new_experiment(message)

        elif intent == "generate_code":
            return self._handle_code_request(message)

        elif intent == "resume":
            return self._handle_resume(message)

        elif intent == "execute":
            return self._handle_execute(message, experiment_context)

        else:
            # Fallback: check if it looks like an experiment
            if self._is_experiment_request(message):
                return self._handle_new_experiment(message)
            return self._handle_general(message)

    def _detect_intent(self, message: str) -> str:
        """Match message against intent patterns."""
        m_lower = message.lower()
        for intent, patterns in self.INTENT_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, m_lower):
                    return intent
        return "unknown"

    def _is_experiment_request(self, message: str) -> bool:
        """Check if message is a full experiment request (not just a question)."""
        m_lower = message.lower()
        for pattern in self.EXPERIMENT_PATTERNS:
            if re.search(pattern, m_lower):
                return True
        # Heuristic: long message with action verbs and ML nouns
        ml_nouns = ["model", "dataset", "accuracy", "train", "classify", "cluster",
                    "predict", "neural", "deep learning", "machine learning", "algorithm"]
        action_verbs = ["build", "develop", "create", "implement", "run", "design", "make"]
        has_noun = any(n in m_lower for n in ml_nouns)
        has_verb = any(v in m_lower for v in action_verbs)
        return has_noun and has_verb and len(message) > 30

    def _handle_explanation(self, message: str) -> Dict[str, Any]:
        """Handle explanation/question requests."""
        m_lower = message.lower()

        # Try to match a known topic
        for topic, explanation in self.EXPLANATION_TOPICS.items():
            if topic in m_lower:
                followups = [
                    f"Show me code for {topic}",
                    f"Compare {topic} with alternatives",
                    f"What are the risks of {topic}?",
                ]
                return self._build_response(
                    explanation, intent="explain",
                    followups=followups
                )

        # Generic ML explanation
        response = self._generic_ml_explanation(message)
        return self._build_response(response, intent="explain",
                                    followups=["Can you give me an example?",
                                               "Show me code for this",
                                               "What are the alternatives?"])

    def _handle_comparison(self, message: str) -> Dict[str, Any]:
        """Handle comparison requests."""
        m_lower = message.lower()

        if "activation" in m_lower:
            return self._build_response(
                self.EXPLANATION_TOPICS["activation function"],
                intent="compare",
                action="suggest_experiment",
                followups=[
                    "Run a comparison experiment with code",
                    "Which activation function should I use for image classification?",
                    "Show me activation function code",
                ]
            )

        if "random forest" in m_lower and ("gradient" in m_lower or "xgboost" in m_lower or "boost" in m_lower):
            response = """**Random Forest vs Gradient Boosting:**

| Aspect | Random Forest | Gradient Boosting |
|--------|--------------|-------------------|
| **Training** | Parallel (fast) | Sequential (slower) |
| **Overfitting** | Robust | Needs regularization |
| **Performance** | Good general purpose | Often better accuracy |
| **Tuning** | Few hyperparameters | Many hyperparameters |
| **Use case** | Production speed | Competition accuracy |

**Recommendation**: Start with Random Forest for baseline; use Gradient Boosting for maximizing performance."""
            return self._build_response(response, intent="compare")

        # Generic comparison
        return self._build_response(
            f"I can compare specific ML algorithms or techniques for you. "
            f"Could you specify what you'd like to compare? For example: "
            f"'Compare Random Forest vs Gradient Boosting' or 'Compare activation functions'.",
            intent="compare",
            followups=["Compare Random Forest vs Gradient Boosting",
                       "Compare activation functions",
                       "Compare SGD vs Adam optimizer"]
        )

    def _handle_improvement(self, message: str, experiment_context: Optional[Dict]) -> Dict[str, Any]:
        """Handle improvement/optimization requests."""
        suggestions = []
        if experiment_context:
            metrics = experiment_context.get("metrics", {})
            acc = metrics.get("accuracy", 0)
            if acc < 0.7:
                suggestions = [
                    "🔧 **Try feature engineering** — create interaction terms or polynomial features",
                    "🔧 **Address class imbalance** — use SMOTE or class_weight='balanced'",
                    "🔧 **Increase model complexity** — try Gradient Boosting or MLP",
                    "🔧 **Collect more data** — accuracy below 70% often signals insufficient data",
                ]
            elif acc < 0.85:
                suggestions = [
                    "🔧 **Hyperparameter tuning** — use GridSearchCV or RandomizedSearchCV",
                    "🔧 **Ensemble stacking** — combine multiple model predictions",
                    "🔧 **Feature selection** — remove irrelevant features with mutual_info_classif",
                    "🔧 **Cross-validation** — use StratifiedKFold for more reliable estimates",
                ]
            else:
                suggestions = [
                    "✨ **Already performing well!** Consider:",
                    "🔧 **Calibrate probabilities** — use CalibratedClassifierCV",
                    "🔧 **Explainability** — add SHAP values for feature importance",
                    "🔧 **Production readiness** — add model drift monitoring",
                ]
        else:
            suggestions = [
                "🔧 **Hyperparameter tuning** — RandomizedSearchCV with cross-validation",
                "🔧 **Feature engineering** — domain-specific feature transformations",
                "🔧 **Regularization** — L1/L2 or dropout for neural networks",
                "🔧 **Data augmentation** — more diverse training examples",
                "🔧 **Ensemble methods** — combine multiple diverse models",
            ]

        response = "**Improvement Suggestions:**\n\n" + "\n".join(suggestions)
        return self._build_response(response, intent="improve",
                                    followups=["Apply hyperparameter tuning",
                                               "Show me regularization code",
                                               "Explain ensemble methods"])

    def _handle_new_experiment(self, message: str) -> Dict[str, Any]:
        """Handle full experiment requests."""
        # Detect dataset requirement
        m_lower = message.lower()
        dataset_free_signals = [
            "gradient descent", "activation function", "compare activation",
            "k-means demo", "kmeans demo", "visualize convergence",
            "implement backprop", "mathematical function", "synthetic"
        ]
        is_dataset_free = any(s in m_lower for s in dataset_free_signals)

        if is_dataset_free:
            response = (
                f"🧪 **Experiment Detected** (Dataset-Free)\n\n"
                f"I'll create a self-contained experiment for:\n"
                f"> *{message[:100]}*\n\n"
                f"**Plan:**\n"
                f"1. 🧠 Analyze your problem statement\n"
                f"2. 💻 Generate problem-specific Python code\n"
                f"3. ▶️ Execute safely in isolated environment\n"
                f"4. 📊 Display real results and visualizations\n"
                f"5. 💾 Save experiment to your lab workspace\n\n"
                f"No external dataset required — I'll generate everything needed! Click **▶️ Run Experiment** to start."
            )
        else:
            response = (
                f"🧪 **Experiment Detected** (Dataset Required)\n\n"
                f"I'll plan a complete ML/DL experiment for:\n"
                f"> *{message[:100]}*\n\n"
                f"**Plan:**\n"
                f"1. 🧠 Analyze problem statement\n"
                f"2. 🔍 Search for suitable datasets (UCI, OpenML, HuggingFace)\n"
                f"3. 📊 Analyze and profile selected dataset\n"
                f"4. 💻 Generate customized experiment code\n"
                f"5. ▶️ Execute and evaluate with real metrics\n"
                f"6. 📈 Visualize results\n"
                f"7. 💾 Save complete experiment\n\n"
                f"Click **▶️ Run Experiment** below to launch the workflow."
            )

        return self._build_response(
            response,
            intent="new_experiment",
            action="start_experiment",
            requires_experiment=True,
            followups=["What dataset will you use?",
                       "What metrics will you evaluate?",
                       "How long will this take?"]
        )

    def _handle_code_request(self, message: str) -> Dict[str, Any]:
        """Handle explicit code generation requests."""
        response = (
            "I'll generate code for your request. "
            "Use the **New Experiment** page to get fully executable, validated code with results, "
            "or describe your problem below and click **▶️ Run Experiment**."
        )
        return self._build_response(response, intent="generate_code",
                                    action="suggest_experiment",
                                    followups=["Run a gradient descent demo",
                                               "Generate K-Means clustering code",
                                               "Compare activation functions"])

    def _handle_resume(self, message: str) -> Dict[str, Any]:
        """Handle experiment resume requests."""
        return self._build_response(
            "Use the **📂 Experiment History** page to view and reopen previous experiments. "
            "You can resume, inspect results, compare runs, and create new versions from any experiment.",
            intent="resume",
            action="navigate_history",
        )

    def _handle_execute(self, message: str, experiment_context: Optional[Dict]) -> Dict[str, Any]:
        """Handle execute/run requests."""
        if experiment_context and experiment_context.get("generated_code"):
            return self._build_response(
                "Running your experiment now! Watch the execution log below.",
                intent="execute",
                action="execute_code",
                requires_experiment=True,
            )
        return self._build_response(
            "No code is ready yet. Describe your ML problem first and I'll generate and run it for you.",
            intent="execute",
            followups=["Build a classification model",
                       "Run a gradient descent demo",
                       "Compare activation functions"]
        )

    def _handle_general(self, message: str) -> Dict[str, Any]:
        """Fallback general response."""
        return self._build_response(
            "I'm your **Autonomous AI Lab Assistant**! I can help you:\n\n"
            "- 🧪 **Run experiments** — just describe your ML/DL problem\n"
            "- 💻 **Generate code** — for any ML algorithm or visualization\n"
            "- 📚 **Explain concepts** — gradient descent, activation functions, overfitting...\n"
            "- 📊 **Compare techniques** — algorithms, optimizers, models\n"
            "- 🔧 **Improve results** — hyperparameter tuning, regularization tips\n"
            "- 📂 **Manage experiments** — reopen, compare, version previous work\n\n"
            "Try: *\"Build a CNN for image classification\"* or *\"Compare activation functions\"*",
            intent="unknown",
            followups=[
                "Build a classification model for tabular data",
                "Run a gradient descent visualization",
                "Compare activation functions",
                "Explain overfitting",
            ]
        )

    def _generic_ml_explanation(self, message: str) -> str:
        """Generate a generic ML explanation when no specific topic is matched."""
        m_lower = message.lower()
        if "neural network" in m_lower or "deep learning" in m_lower:
            return ("**Neural Networks** are computational models inspired by the brain. "
                    "They consist of layers of interconnected neurons that learn representations "
                    "from data through backpropagation and gradient descent.")
        if "machine learning" in m_lower:
            return ("**Machine Learning** enables computers to learn from data without explicit programming. "
                    "Key categories: supervised learning (labeled data), unsupervised learning (unlabeled), "
                    "and reinforcement learning (reward signals).")
        if "backpropagation" in m_lower:
            return ("**Backpropagation** computes gradients of the loss w.r.t. each weight using the chain rule, "
                    "flowing errors backward from output to input layers to update weights.")
        return (f"I can explain various ML/DL concepts in detail. "
                f"For your question about '{message[:50]}', please be more specific or try: "
                f"'Explain gradient descent', 'Explain overfitting', or 'Explain CNN'.")

    def _build_response(
        self,
        response_text: str,
        intent: str = "unknown",
        action: str = "",
        requires_experiment: bool = False,
        generated_code: str = "",
        followups: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "response_text": response_text,
            "intent": intent,
            "action": action,
            "requires_experiment": requires_experiment,
            "generated_code": generated_code,
            "suggested_followups": followups or [],
        }
