"""Exception hierarchy.

`ToolError` subclasses carry messages that are safe and useful to show to the model
(they become tool results); everything else is an engine-side failure.
"""


class DeepReviewError(Exception):
    """Base class for all deep_review errors."""


class ConfigError(DeepReviewError):
    """Invalid or missing configuration."""


class LLMError(DeepReviewError):
    """The model endpoint failed or returned something unusable."""


class StructuredOutputError(LLMError):
    """Model output did not match the requested schema."""


class WorkspaceError(DeepReviewError):
    """Building or cleaning up the review checkout failed."""


class GitError(WorkspaceError):
    """A git command failed."""


class ToolError(DeepReviewError):
    """A tool call failed; the message is returned to the model."""


class ToolArgumentError(ToolError):
    """The model sent invalid tool arguments."""


class PathEscapeError(ToolError):
    """A path resolved outside the repository root (or into .git)."""


class FileTooLargeError(ToolError):
    """A file exceeded the configured size cap."""


class BinaryFileError(ToolError):
    """A binary or LFS-pointer file was requested as text."""
