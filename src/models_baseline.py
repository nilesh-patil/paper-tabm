import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, ReLU
import xgboost as xgb

def build_mlp(input_shape, num_classes, num_layers=4, hidden_units=256, dropout_rate=0.2):

    model = Sequential()
    model.add(Dense(hidden_units, input_shape=input_shape, activation='relu'))
    model.add(Dropout(dropout_rate))

    for _ in range(num_layers - 1):
        model.add(Dense(hidden_units, activation='relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid'))

    return model


def build_xgboost(num_classes, 
                  n_estimators=100,
                  max_depth=6,
                  learning_rate=0.3,
                  subsample=0.8,
                  colsample_bytree=0.8,
                  min_child_weight=1,
                  reg_alpha=0,
                  reg_lambda=1,
                  early_stopping_rounds=10,
                  random_state=42):
    """
    Build an XGBoost model for classification or regression.
    
    Args:
        num_classes: Number of classes for classification. If 1 or None, creates a regressor.
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Boosting learning rate
        subsample: Subsample ratio of the training instances
        colsample_bytree: Subsample ratio of columns when constructing each tree
        min_child_weight: Minimum sum of instance weight needed in a child
        reg_alpha: L1 regularization term on weights
        reg_lambda: L2 regularization term on weights
        random_state: Random seed
    
    Returns:
        XGBoost classifier or regressor model
    """
    
    common_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'early_stopping_rounds': early_stopping_rounds,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_child_weight': min_child_weight,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'random_state': random_state,
        'tree_method': 'hist',  # Use histogram-based algorithm for efficiency
        'enable_categorical': True  # Enable categorical feature support
    }
    
    if num_classes is None or num_classes == 1:
        # Regression
        model = xgb.XGBRegressor(**common_params)
    elif num_classes == 2:
        # Binary classification
        model = xgb.XGBClassifier(
            **common_params,
            objective='binary:logistic',
            use_label_encoder=False
        )
    else:
        # Multi-class classification
        model = xgb.XGBClassifier(
            **common_params,
            objective='multi:softprob',
            num_class=num_classes,
            use_label_encoder=False
        )
    
    return model

