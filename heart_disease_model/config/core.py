# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Dict, List
from pydantic import BaseModel
from strictyaml import YAML, load

import heart_disease_model

# Project Directories
PACKAGE_ROOT = Path(heart_disease_model.__file__).resolve().parent
#print(PACKAGE_ROOT)
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
#print(CONFIG_FILE_PATH)

DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    training_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    age: int
    sex: int
    cp: int
    trestbps: int 
    chol:int 
    fbs:int 
    restecg: int 
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: str

    test_size:float
    random_state: int
    n_estimators: int
    max_depth: int
    max_features: int

    # Add these fields
    thal_var: str
    thal_mappings: Dict[str, int]
    features: List[str] 


class Config(BaseModel):
    """Master config object."""

    app_config_: AppConfig
    model_config_: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # Extract features from the YAML
    features = parsed_config.data.get("features", [])

    # Create a dictionary with default values for ModelConfig fields
    model_config_defaults = {
        "age": 0,
        "sex": 0,
        "cp": 0,
        "trestbps": 0,
        "chol": 0,
        "fbs": 0,
        "restecg": 0,
        "thalach": 0,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 0,
        "ca": 0,
        "thal": "",
        "test_size": 0.2,
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": 10,
        "max_features": 3,
    }

    # Update the defaults with values from the YAML file
    model_config_data = {**model_config_defaults, **parsed_config.data}

    # Validate and create the Config object
    _config = Config(
        app_config_=AppConfig(**parsed_config.data),
        model_config_=ModelConfig(**model_config_data),
    )

    return _config


config = create_and_validate_config()