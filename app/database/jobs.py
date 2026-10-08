"""Job history and per-stage status in SQLite."""

from __future__ import annotations

import json
from typing import Any

from app.database.database import Database

_NOW = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


class JobsRepo:
    def __init__(self, db: Database):
        self._db = db

    def create(self, *, input_type: str, input_value: str, source_language: str,
               target_language: str, mode: str, output_dir: str | None) -> int:
        with self._db.conn:
            cur = self._db.conn.execute(
                "INSERT INTO jobs (input_type, input_value, source_language, target_language, mode, output_dir) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (input_type, input_value, source_language, target_language, mode, output_dir),
            )
        return int(cur.lastrowid)

    def update(self, job_id: int, *, status: str | None = None, input_hash: str | None = None,
               error: str | None = None, output_dir: str | None = None,
               outputs: dict | None = None, warnings: list | None = None, stats: dict | None = None) -> None:
        fields: dict[str, Any] = {}
        if status is not None:
            fields["status"] = status
        if input_hash is not None:
            fields["input_hash"] = input_hash
        if error is not None:
            fields["error"] = error
        if output_dir is not None:
            fields["output_dir"] = output_dir
        if outputs is not None:
            fields["output_paths"] = json.dumps(outputs, ensure_ascii=False)
        if warnings is not None:
            fields["warnings"] = json.dumps(warnings, ensure_ascii=False)
        if stats is not None:
            fields["stats"] = json.dumps(stats, ensure_ascii=False)
        if not fields:
            return
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._db.conn:
            self._db.conn.execute(
                f"UPDATE jobs SET {assignments}, updated_at = {_NOW} WHERE id = ?", (*fields.values(), job_id))

    def record_stage(self, job_id: int, stage: str, status: str, cache_key: str | None, error: str | None) -> None:
        started = f"{_NOW}" if status == "running" else "NULL"
        finished = f"{_NOW}" if status in ("completed", "failed", "skipped") else "NULL"
        with self._db.conn:
            self._db.conn.execute(
                f"INSERT INTO job_stages (job_id, stage, status, cache_key, error, started_at, finished_at) "
                f"VALUES (?, ?, ?, ?, ?, {started}, {finished}) "
                f"ON CONFLICT(job_id, stage) DO UPDATE SET status = excluded.status, "
                f"cache_key = COALESCE(excluded.cache_key, job_stages.cache_key), error = excluded.error, "
                f"started_at = COALESCE(excluded.started_at, job_stages.started_at), "
                f"finished_at = excluded.finished_at",
                (job_id, stage, status, cache_key, error),
            )

    def recent_stats(self, limit: int = 20) -> list[dict]:
        """`stats` of the last finished jobs: this machine's own timing history for ETA estimates."""
        rows = self._db.conn.execute(
            "SELECT stats FROM jobs WHERE status = 'completed' ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
        out: list[dict] = []
        for row in rows:
            try:
                value = json.loads(row["stats"] or "{}")
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                out.append(value)
        return out

    def ensure_series(self, name: str) -> int:
        with self._db.conn:
            self._db.conn.execute("INSERT INTO series (name) VALUES (?) ON CONFLICT(name) DO NOTHING", (name,))
        row = self._db.conn.execute("SELECT id FROM series WHERE name = ?", (name,)).fetchone()
        return int(row["id"])

    def set_series(self, job_id: int, series: dict) -> None:
        series_id = self.ensure_series(series["series_name"]) if series.get("series_name") else None
        with self._db.conn:
            self._db.conn.execute(
                f"UPDATE jobs SET series_id = ?, season = ?, episode = ?, episode_title = ?, year = ?, "
                f"updated_at = {_NOW} WHERE id = ?",
                (series_id, series.get("season"), series.get("episode"), series.get("episode_title"),
                 series.get("year"), job_id))

    def get(self, job_id: int) -> dict | None:
        row = self._db.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def stages(self, job_id: int) -> dict[str, dict]:
        rows = self._db.conn.execute("SELECT * FROM job_stages WHERE job_id = ?", (job_id,)).fetchall()
        return {r["stage"]: dict(r) for r in rows}

    def recent(self, limit: int = 50) -> list[dict]:
        rows = self._db.conn.execute(
            "SELECT jobs.*, series.name AS series_name FROM jobs "
            "LEFT JOIN series ON jobs.series_id = series.id "
            "ORDER BY jobs.id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def prepare_resume(self, job_id: int, extras: dict | None = None) -> dict | None:
        """Prepare an existing job to be re-run/resumed by setting status to 'pending'
        and ensuring it has queue_config and queue_order."""
        from pathlib import Path
        row = self.get(job_id)
        if not row:
            return None
        qconfig = {}
        if row.get("queue_config"):
            try:
                qconfig = json.loads(row["queue_config"])
            except Exception:
                pass
        if not qconfig:
            series_name = None
            if row.get("series_id"):
                s_row = self._db.conn.execute(
                    "SELECT name FROM series WHERE id = ?", (row["series_id"],)
                ).fetchone()
                if s_row:
                    series_name = s_row["name"]
            val = row.get("input_value", "")
            title = row.get("episode_title") or (
                Path(val).stem if row.get("input_type") == "file" else val
            )
            qconfig = {
                "extras": {k: v for k, v in (extras or {}).items() if not k.endswith("_api_key")},
                "series": {
                    "series_name": series_name,
                    "season": row.get("season"),
                    "episode": row.get("episode"),
                    "episode_title": row.get("episode_title"),
                    "year": row.get("year"),
                },
                "title": title or "",
            }
        order = row.get("queue_order")
        with self._db.conn:
            if order is None:
                max_row = self._db.conn.execute(
                    "SELECT MAX(queue_order) AS m FROM jobs WHERE queue_config IS NOT NULL"
                ).fetchone()
                max_order = max_row["m"] if max_row and max_row["m"] is not None else 0
                order = max_order + 1
            self._db.conn.execute(
                f"UPDATE jobs SET status = 'pending', queue_config = ?, queue_order = ?, "
                f"error = NULL, updated_at = {_NOW} WHERE id = ?",
                (json.dumps(qconfig, ensure_ascii=False), order, job_id),
            )
        return self.get(job_id)

    def mark_interrupted(self) -> int:
        """Jobs left 'running' by a crash or kill become 'paused' (their caches allow resuming)."""
        with self._db.conn:
            cur = self._db.conn.execute(
                f"UPDATE jobs SET status = 'paused', updated_at = {_NOW} WHERE status = 'running'")
        return cur.rowcount

    def enqueue(self, *, input_type: str, input_value: str, source_language: str,
                target_language: str, mode: str, output_dir: str | None = None,
                series: dict | None = None, extras: dict | None = None,
                title: str | None = None, queue_order: int | None = None) -> int:
        config_dict = {
            # API keys are never written to the database; they are read from the settings when the job runs.
            "extras": {k: v for k, v in (extras or {}).items() if not k.endswith("_api_key")},
            "series": series or {},
            "title": title or "",
        }
        with self._db.conn:
            if queue_order is None:
                row = self._db.conn.execute(
                    "SELECT MAX(queue_order) AS m FROM jobs WHERE queue_config IS NOT NULL"
                ).fetchone()
                max_order = row["m"] if row and row["m"] is not None else 0
                queue_order = max_order + 1
            series_id = self.ensure_series(series["series_name"]) if series and series.get("series_name") else None
            cur = self._db.conn.execute(
                "INSERT INTO jobs (input_type, input_value, source_language, target_language, mode, "
                "output_dir, series_id, season, episode, episode_title, year, status, queue_config, queue_order) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                (input_type, input_value, source_language, target_language, mode, output_dir,
                 series_id,
                 series.get("season") if series else None,
                 series.get("episode") if series else None,
                 series.get("episode_title") if series else None,
                 series.get("year") if series else None,
                 json.dumps(config_dict, ensure_ascii=False),
                 queue_order),
            )
        return int(cur.lastrowid)

    def get_queue(self) -> list[dict]:
        rows = self._db.conn.execute(
            "SELECT * FROM jobs WHERE queue_config IS NOT NULL ORDER BY queue_order ASC, id ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def next_pending_in_queue(self) -> dict | None:
        row = self._db.conn.execute(
            "SELECT * FROM jobs WHERE queue_config IS NOT NULL AND status IN ('pending', 'paused') "
            "ORDER BY queue_order ASC, id ASC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def cancel_queue(self) -> int:
        with self._db.conn:
            cur = self._db.conn.execute(
                f"UPDATE jobs SET status = 'cancelled', updated_at = {_NOW} "
                f"WHERE queue_config IS NOT NULL AND status = 'pending'"
            )
        return cur.rowcount

    def clear_completed_queue(self) -> int:
        with self._db.conn:
            cur = self._db.conn.execute(
                "UPDATE jobs SET queue_config = NULL, queue_order = NULL "
                "WHERE queue_config IS NOT NULL AND status IN ('completed', 'cancelled', 'failed')"
            )
        return cur.rowcount

    def remove_from_queue(self, job_id: int) -> None:
        with self._db.conn:
            self._db.conn.execute(
                "UPDATE jobs SET queue_config = NULL, queue_order = NULL WHERE id = ?", (job_id,)
            )

