import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from heart_disease_model.config.core import config
from heart_disease_model.processing.data_manager import remove_thal_1_2



def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""
    if input_df['thal'] == '1' or input_df['thal'] == '2':
        pre_processed = remove_thal_1_2(data_frame=input_df)
    else:
        pre_processed = input_df.copy()
    validated_data = pre_processed[config.model_config_.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    age: Optional[int]
    sex: Optional[int]
    cp: Optional[int]
    trestbps: Optional[int]
    chol: Optional[int]
    fbs: Optional[int]
    restecg: Optional[int]
    thalach: Optional[int]
    exang: Optional[int]
    oldpeak: Optional[float]
    slope: Optional[int]
    ca: Optional[int]
    thal: Optional[str]
    
class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
