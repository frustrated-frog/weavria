"""Formation journal SQL executed only inside the memory store's owned transactions."""

import sqlite3

from .contracts import MemoryConflictError
from .formation import MemoryFormationJob


def current_job(conn: sqlite3.Connection, scope: str, job_id: str) -> MemoryFormationJob | None:
    row = conn.execute(
        "SELECT data FROM memory_formation WHERE scope=? AND job_id=? "
        "ORDER BY revision DESC LIMIT 1",
        (scope, job_id),
    ).fetchone()
    return MemoryFormationJob.model_validate_json(row[0]) if row is not None else None


def append_job(
    conn: sqlite3.Connection,
    scope: str,
    job: MemoryFormationJob,
    *,
    expected_revision: int,
) -> None:
    current = current_job(conn, scope, job.job_id)
    if (current.revision if current is not None else 0) != expected_revision:
        raise MemoryConflictError("formation revision changed")
    if job.revision != expected_revision + 1:
        raise MemoryConflictError("formation revisions must be consecutive")
    conn.execute(
        "INSERT INTO memory_formation VALUES(?,?,?,?)",
        (scope, job.job_id, job.revision, job.model_dump_json()),
    )
