import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from heart_disease_model.config.core import config
from heart_disease_model.processing.features import Mapper

heart_disease_pipe=Pipeline([
    
     ##==========Mapper======##
     ("map_thal", Mapper(config.model_config_.thal_var, config.model_config_.thal_mappings)
      ),
    # scale
     ("scaler", StandardScaler()),
     ('model_rf', RandomForestClassifier(n_estimators=config.model_config_.n_estimators, 
                                         max_depth=config.model_config_.max_depth, 
                                         max_features=config.model_config_.max_features,
                                         random_state=config.model_config_.random_state))
          
     ])
