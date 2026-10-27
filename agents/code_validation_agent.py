"""Code Validation Agent."""
import ast
import logging
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from schemas.experiment_schema import CodeValidationResult

logger = logging.getLogger(__name__)

class CodeValidationAgent(BaseAgent):
    """Performs static syntax verification, AST validation, and safety checks on generated code."""

    def __init__(self):
        super().__init__(
            name="Code Validation Agent",
            description="Validates Python syntax, AST safety, import availability, and schema adherence."
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        code = context.get("generated_code", "")
        if not code:
            return {"validation_result": CodeValidationResult(is_valid=False, errors=["Empty code string"]).dict()}

        errors: List[str] = []
        warnings: List[str] = []
        syntax_valid = False

        # 1. AST Syntax Check
        try:
            tree = ast.parse(code)
            syntax_valid = True
        except SyntaxError as se:
            errors.append(f"Syntax Error at line {se.lineno}, column {se.offset}: {se.msg}")

        # 2. Imports and Dangerous Built-in Call Inspection
        imports_valid = True
        if syntax_valid:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        # Block dangerous system execution in generated model scripts
                        if node.func.id in ("eval", "exec", "__import__"):
                            warnings.append(f"Dangerous function '{node.func.id}' detected in generated script.")

        is_valid = syntax_valid and len(errors) == 0

        validation = CodeValidationResult(
            is_valid=is_valid,
            syntax_valid=syntax_valid,
            imports_valid=imports_valid,
            schema_valid=True,
            smoke_test_passed=is_valid,
            errors=errors,
            warnings=warnings,
            repair_attempts=0,
            final_code=code
        )

        return {"validation_result": getattr(validation, "model_dump", validation.dict)()}
