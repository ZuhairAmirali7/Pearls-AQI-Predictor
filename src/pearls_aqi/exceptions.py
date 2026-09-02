"""Custom exception hierarchy for the Pearls AQI Predictor.

Using explicit exception types (rather than bare ``ValueError`` / ``RuntimeError``)
lets callers — API error handlers, pipeline retry logic, tests — react to
specific failure modes without string-matching messages.
"""

from __future__ import annotations


class PearlsError(Exception):
    """Base class for all project-specific errors."""


# --- Configuration ---------------------------------------------------------
class ConfigError(PearlsError):
    """Raised when configuration is missing, malformed, or inconsistent."""


# --- Data providers --------------------------------------------------------
class ProviderError(PearlsError):
    """Base class for data-provider (external API) failures."""


class ProviderRequestError(ProviderError):
    """A request to an external provider failed after retries."""


class ProviderResponseError(ProviderError):
    """A provider returned a malformed, empty, or unexpected response."""


class RateLimitError(ProviderError):
    """A provider signalled that we are being rate limited."""


# --- Storage / registry ----------------------------------------------------
class StorageError(PearlsError):
    """Base class for feature-store / registry failures."""


class FeatureStoreError(StorageError):
    """Reading from or writing to the feature store failed."""


class ModelRegistryError(StorageError):
    """Reading from or writing to the model registry failed."""


class ModelNotFoundError(ModelRegistryError):
    """No model (of the requested status) was found in the registry."""


# --- Validation ------------------------------------------------------------
class ValidationError(PearlsError):
    """A data or schema validation check failed."""


class SchemaMismatchError(ValidationError):
    """Feature schema is incompatible with the loaded model."""


# --- Modelling / prediction ------------------------------------------------
class TrainingError(PearlsError):
    """Model training failed or produced invalid artifacts."""


class PredictionError(PearlsError):
    """Generating a forecast failed."""
