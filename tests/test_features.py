
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
from heart_disease_model.config.core import config
from heart_disease_model.processing.features import Mapper


def test_age_variable_transformer(sample_input_data):
    # Given
    transformer = Mapper(
        variables=config.model_config_.thal_var, 
        mappings=config.model_config_.thal_mappings, 
    )
    valid_index = sample_input_data[0].index[0] 
    initial_value = sample_input_data[0].loc[valid_index, 'thal']
    assert initial_value == "normal" 

    # When
    transformed_data = transformer.transform(sample_input_data[0])

    # Then
    assert transformed_data[config.model_config_.thal_var].isna().sum() == 0
    assert transformed_data[config.model_config_.thal_var].dtype == int