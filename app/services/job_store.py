from typing import Any

JOB_STORE: dict[str, dict[str, Any]] = {}


def create_job(job_id: str):
    JOB_STORE[job_id] = {
        "job_id": job_id,
        "status": "running",
        "progress": 0,
        "stage": "created",
        "message": "Job creado",
        "result": None,
        "error": None,
    }


def update_job(job_id: str, progress: int, stage: str, message: str, result=None):
    if job_id in JOB_STORE:
        JOB_STORE[job_id]["progress"] = max(0, min(100, int(progress)))
        JOB_STORE[job_id]["stage"] = stage
        JOB_STORE[job_id]["message"] = message
        if result is not None:
            JOB_STORE[job_id]["result"] = result


def complete_job(job_id: str, result=None):
    if job_id in JOB_STORE:
        JOB_STORE[job_id]["status"] = "completed"
        JOB_STORE[job_id]["progress"] = 100
        JOB_STORE[job_id]["stage"] = "done"
        JOB_STORE[job_id]["message"] = "Proceso completado"
        JOB_STORE[job_id]["result"] = result


def fail_job(job_id: str, error: str):
    if job_id in JOB_STORE:
        JOB_STORE[job_id]["status"] = "failed"
        JOB_STORE[job_id]["stage"] = "error"
        JOB_STORE[job_id]["message"] = error
        JOB_STORE[job_id]["error"] = error


def get_job(job_id: str):
    return JOB_STORE.get(job_id)