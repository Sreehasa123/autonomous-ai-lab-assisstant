"""Background Task Manager."""
import threading
import uuid
import logging
from typing import Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)

class TaskManager:
    """Runs long background workloads asynchronously without blocking Streamlit."""

    _instances = {}
    _lock = threading.Lock()

    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if "default" not in cls._instances:
                cls._instances["default"] = cls()
            return cls._instances["default"]

    def submit_task(self, target_func: Callable, *args, **kwargs) -> str:
        task_id = str(uuid.uuid4())
        task_record = {
            "id": task_id,
            "status": "running",
            "result": None,
            "error": None,
            "thread": None
        }

        def runner():
            try:
                res = target_func(*args, **kwargs)
                task_record["result"] = res
                task_record["status"] = "completed"
            except Exception as e:
                logger.error(f"Background task {task_id} failed: {e}", exc_info=True)
                task_record["error"] = str(e)
                task_record["status"] = "failed"

        t = threading.Thread(target=runner, daemon=True)
        task_record["thread"] = t
        self.tasks[task_id] = task_record
        t.start()
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.tasks.get(task_id)
