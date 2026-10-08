"""Pipeline exceptions."""


class JobCancelled(Exception):
    """Raised inside a stage when the user cancels; completed work stays cached."""


class PipelineError(RuntimeError):
    """A failure that aborts the job, with a user-facing message key for the UI."""

    def __init__(self, message: str, ui_key: str = "error.pipeline_failed", **ui_args):
        super().__init__(message)
        self.ui_key = ui_key
        self.ui_args = ui_args
