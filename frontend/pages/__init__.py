"""Frontend pages package."""
from frontend.pages.dashboard import render_dashboard
from frontend.pages.new_experiment import render_new_experiment
from frontend.pages.dataset_explorer import render_dataset_explorer
from frontend.pages.experiment_workspace import render_experiment_workspace
from frontend.pages.model_laboratory import render_model_laboratory
from frontend.pages.previous_experiments import render_previous_experiments
from frontend.pages.experiment_comparison import render_experiment_comparison
from frontend.pages.reports import render_reports
from frontend.pages.integrations import render_integrations
from frontend.pages.settings import render_settings
from frontend.pages.help_documentation import render_help_documentation

__all__ = [
    "render_dashboard",
    "render_new_experiment",
    "render_dataset_explorer",
    "render_experiment_workspace",
    "render_model_laboratory",
    "render_previous_experiments",
    "render_experiment_comparison",
    "render_reports",
    "render_integrations",
    "render_settings",
    "render_help_documentation",
]
