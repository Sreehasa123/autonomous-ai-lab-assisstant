"""Workflow State Machine and Invalidation Manager."""
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class WorkflowStep(str, Enum):
    PROBLEM_UNDERSTANDING = "problem_understanding"
    EXPERIMENT_PLANNING = "experiment_planning"
    DATASET_SELECTION = "dataset_selection"
    DATASET_ANALYSIS = "dataset_analysis"
    PREPROCESSING = "preprocessing"
    MODEL_SELECTION = "model_selection"
    CODE_GENERATION = "code_generation"
    CODE_VALIDATION = "code_validation"
    EXECUTION = "execution"
    EVALUATION = "evaluation"
    REPORTING = "reporting"

class WorkflowStateMachine(BaseModel):
    """Tracks state and determines execution prerequisites."""

    current_step: WorkflowStep = WorkflowStep.PROBLEM_UNDERSTANDING
    completed_steps: List[str] = Field(default_factory=list)
    step_outputs: Dict[str, Any] = Field(default_factory=dict)
    is_paused: bool = False
    last_error: Optional[str] = None

    def mark_completed(self, step: WorkflowStep, output: Optional[Dict[str, Any]] = None):
        if step.value not in self.completed_steps:
            self.completed_steps.append(step.value)
        if output:
            self.step_outputs[step.value] = output
        self.last_error = None

    def is_completed(self, step: WorkflowStep) -> bool:
        return step.value in self.completed_steps

    def invalidate_from(self, step: WorkflowStep):
        """Invalidates downstream steps when upstream settings are altered."""
        order = [
            WorkflowStep.PROBLEM_UNDERSTANDING,
            WorkflowStep.EXPERIMENT_PLANNING,
            WorkflowStep.DATASET_SELECTION,
            WorkflowStep.DATASET_ANALYSIS,
            WorkflowStep.PREPROCESSING,
            WorkflowStep.MODEL_SELECTION,
            WorkflowStep.CODE_GENERATION,
            WorkflowStep.CODE_VALIDATION,
            WorkflowStep.EXECUTION,
            WorkflowStep.EVALUATION,
            WorkflowStep.REPORTING
        ]
        try:
            idx = order.index(step)
            steps_to_invalidate = [s.value for s in order[idx:]]
            self.completed_steps = [s for s in self.completed_steps if s not in steps_to_invalidate]
            for s in steps_to_invalidate:
                self.step_outputs.pop(s, None)
            self.current_step = step
        except ValueError:
            pass
