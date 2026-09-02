"""Typed configuration models (pydantic) mirroring ``config/config.yaml``.

Validation happens at load time so a malformed config fails fast with a clear
message rather than surfacing as a confusing error deep in a pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from pearls_aqi.data.domain import Location


class ProjectConfig(BaseModel):
    name: str = "pearls-aqi-predictor"
    feature_pipeline_version: str = "1.0.0"


class CityEntry(BaseModel):
    country: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str


class LocationConfig(BaseModel):
    city: str
    country: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str

    def to_location(self) -> Location:
        return Location(
            city=self.city,
            country=self.country,
            latitude=self.latitude,
            longitude=self.longitude,
            timezone=self.timezone,
        )


class ForecastConfig(BaseModel):
    horizon_hours: int = Field(default=72, ge=1, le=336)
    frequency: str = "hourly"
    strategy: str = "direct"
    min_aqi: float = 0.0
    max_aqi: float = 500.0


class DataConfig(BaseModel):
    historical_days: int = Field(default=730, ge=1)
    aqi_standard: str = "us_epa"
    target_source: str = "local_computed"  # local_computed | provider


class ProvidersConfig(BaseModel):
    air_quality: str = "openmeteo"
    weather: str = "openmeteo"
    request_timeout_seconds: float = 20.0
    max_retries: int = 4
    backoff_seconds: float = 1.5
    batch_days: int = 30
    store_raw_payloads: bool = False


class FeatureGroup(BaseModel):
    name: str
    version: int = 1


class FeatureStoreConfig(BaseModel):
    backend: str = "local"  # local | hopsworks
    local_dir: str = "data/local/feature_store"
    groups: dict[str, FeatureGroup] = Field(default_factory=dict)


class ModelRegistryConfig(BaseModel):
    backend: str = "local"  # local | hopsworks
    local_dir: str = "artifacts/models"


class FeaturesConfig(BaseModel):
    target_columns: list[str] = Field(default_factory=lambda: ["aqi"])
    lag_hours: list[int] = Field(default_factory=lambda: [1, 3, 6, 12, 24, 48, 72])
    rolling_windows: list[int] = Field(default_factory=lambda: [3, 6, 12, 24, 48, 72])
    rolling_stats: list[str] = Field(
        default_factory=lambda: ["mean", "min", "max", "std", "median"]
    )
    lag_pollutants: list[str] = Field(default_factory=lambda: ["aqi", "pm2_5", "pm10", "o3", "no2"])
    rolling_min_fraction: float = 0.5


class TrainingConfig(BaseModel):
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    split_gap_hours: int = 72
    min_training_rows: int = 500
    cv_splits: int = 4
    cv_gap_hours: int = 24
    random_search_iterations: int = 20
    random_state: int = 42
    models: list[str] = Field(
        default_factory=lambda: [
            "persistence",
            "seasonal_naive",
            "rolling_average",
            "hour_of_day",
            "ridge",
            "elastic_net",
            "random_forest",
            "hist_gradient_boosting",
            "tensorflow",
        ]
    )

    @model_validator(mode="after")
    def _check_fractions(self) -> TrainingConfig:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"train+validation+test fractions must sum to <= 1.0 (got {total:.3f})"
            )
        return self


class ModelPromotionConfig(BaseModel):
    minimum_mae_improvement_percent: float = 1.0
    maximum_hazard_mae_degradation_percent: float = 2.0
    require_smoke_test: bool = True


class AlertsConfig(BaseModel):
    enabled: bool = True
    unhealthy_threshold: float = 151
    very_unhealthy_threshold: float = 201
    hazardous_threshold: float = 301
    consecutive_hours: int = 3
    spike_delta: float = 75
    spike_window_hours: int = 6
    channels: list[str] = Field(default_factory=lambda: ["dashboard", "log"])


class MonitoringConfig(BaseModel):
    max_data_age_hours: int = 3
    rolling_error_window_hours: int = 168
    drift_psi_threshold: float = 0.2


class DisplayConfig(BaseModel):
    timezone: str = "Asia/Karachi"


class AppConfig(BaseModel):
    """Fully-resolved application configuration."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    location: LocationConfig
    cities: dict[str, CityEntry] = Field(default_factory=dict)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    feature_store: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)
    model_registry: ModelRegistryConfig = Field(default_factory=ModelRegistryConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model_promotion: ModelPromotionConfig = Field(default_factory=ModelPromotionConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)

    # pydantic v2: silence the "model_" namespace warning for model_registry/promotion
    model_config = {"protected_namespaces": ()}

    @property
    def active_location(self) -> Location:
        return self.location.to_location()

    def location_for_city(self, city: str) -> Location:
        """Resolve a ``Location`` for a city name using the ``cities`` table.

        Falls back to the active location if the city matches it, else raises so
        callers get a clear error instead of a silent wrong-city forecast.
        """
        if city == self.location.city:
            return self.active_location
        entry = self.cities.get(city)
        if entry is None:
            known = ", ".join(sorted({self.location.city, *self.cities.keys()}))
            raise ValueError(f"Unknown city {city!r}. Known cities: {known}.")
        return Location(
            city=city,
            country=entry.country,
            latitude=entry.latitude,
            longitude=entry.longitude,
            timezone=entry.timezone,
        )
