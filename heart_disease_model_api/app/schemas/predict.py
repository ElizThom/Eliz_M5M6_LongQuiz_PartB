from typing import Any, List, Optional

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]

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

