"""Exceptions raised by ML Forge. Messages are written for the user, in French."""


class ForgeError(Exception):
    """Base class for every expected, user-facing error."""


class ConfigError(ForgeError):
    """The configuration file is missing or invalid."""


class StorageError(ForgeError):
    """A database could not be opened or migrated."""


class ProjectError(ForgeError):
    """A project operation cannot be performed."""


class ApprovalRequired(ForgeError):
    """A guarded action was refused or not explicitly approved."""


class ExperimentError(ForgeError):
    """An experiment card or status transition is invalid."""


class DataSourceError(ForgeError):
    """A data source cannot be registered or read."""
