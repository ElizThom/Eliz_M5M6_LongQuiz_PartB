import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from heart_disease_model import __version__ as _version
from heart_disease_model.config.core import config
from heart_disease_model.pipeline import heart_disease_pipe
from heart_disease_model.processing.data_manager import load_pipeline
from heart_disease_model.processing.data_manager import remove_thal_1_2
from heart_disease_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
heart_disease_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.model_config_.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = heart_disease_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = heart_disease_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":

    data_in={'age':[58],'sex':[1],'cp':[4],'trestbps':[145],'chol':[195],'fbs':[0],'restecg':[2],'thalach':[178],'exang':[1],'oldpeak':[0.7],'slope':[2],'ca':[2],'thal':["normal"]}

    
    make_prediction(input_data=data_in)
