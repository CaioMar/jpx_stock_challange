from functools import partial
from typing import Dict, Any, List, Union

import pandas as pd
import numpy as np
import category_encoders as ce

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.metrics._scorer import _PredictScorer

import catboost
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.pyll import scope

def get_search_space() -> Dict[str, Any]:

    complete_search_space = {
        # type refers to classifier type: either random forest, xgboost, catboost or lightgbm
        'clf_type': hp.choice('clf_type', [
            {
                'type': 'xgboost', 
                'catbstencoder': {
                    'drop_invariant': hp.choice('xgboost.drop_invariant', [False, True])
                }, 
                'clf': {
                    'n_estimators': scope.int(hp.quniform('xgboost.n_estimators', 50, 200, 25)),
                    'max_depth': scope.int(hp.quniform("xgboost.max_depth", 2, 10, 1)),
                    'colsample_bytree': hp.uniform('xgboost.colsample_bytree', 0.2, 1), #same as max_features
                    'subsample': hp.uniform('xgboost.subsample', 0.3, 1), #sample rate for bagging
                    'eta': hp.uniform('xgboost.eta', 0.01, 1),
                    'lambda': hp.uniform('xgboost.lambda', 0.8, 1), #L2 reg
                    'alpha': hp.uniform('xgboost.alpha', 0, 0.05), #L1 reg
                    'booster': hp.choice ('xgboost.booster', ['gbtree','dart']),
                    'gamma': hp.uniform ('xgboost.gamma', 0,0.01),                                                
                    'min_child_weight': scope.int(hp.quniform ('xgboost.min_child_weight', 1, 20, 1)),
                    'scale_pos_weight': hp.uniform('xgboost.scale_pos_weight', 1, 30)
                }
            }, 
            {
                'type': 'catboost', 
                'imputer': {
                    'strategy': hp.choice('catboost.strategy', ['constant', 'most_frequent'])
                }, 
                'clf': {
                    'n_estimators': scope.int(hp.quniform('catboost.n_estimators', 50, 200, 25)),
                    'max_depth': scope.int(hp.quniform("catboost.max_depth", 2, 10, 1)),
                    'rsm': hp.uniform('catboost.rsm', 0.2, 1), #same as max_features
                    'subsample': hp.uniform('catboost.subsample', 0.3, 1), #sample rate for bagging
                    'eta': hp.uniform('catboost.eta', 0.01, 1),
                    'l2_leaf_reg': hp.uniform('catboost.l2_leaf_reg', 0.8, 1), #L2 reg, catboost does not have L1
                    'feature_border_type': hp.choice('catboost.feature_border_type', ['GreedyLogSum', 'MinEntropy']),                                                
                    'min_data_in_leaf': scope.int(hp.quniform ('catboost.min_data_in_leaf', 1, 20, 1)),
                    'scale_pos_weight': hp.uniform('catboost.scale_pos_weight', 1, 30)
                }
            }
        ])
    }

    return complete_search_space

def get_estimator(hps, cat_cols):
    """
    Constructs estimator
    """
    
    if hps['clf_type']['type'] == 'xgboost':
        model = Pipeline([
            ('catbstencoder', ce.CatBoostEncoder(**hps['clf_type']['catbstencoder'])), 
            ('clf', xgb.XGBClassifier(**hps['clf_type']['clf']))
        ])
    elif hps['clf_type']['type'] == 'catboost':
        model = Pipeline([
            ('imputer', SimpleImputer(**hps['clf_type']['imputer'], fill_value=0)), 
            ('clf', catboost.CatBoostClassifier(**hps['clf_type']['clf'], cat_features=cat_cols))
        ])
    else:
        raise KeyError('Unknown classifier type hyperparameter value: {0}'.format(hps['clf_type']['type']))
    
    return model

def objective(
    hps: Dict[str, Any],
    X: pd.DataFrame, 
    y: pd.Series, 
    cat_cols: List[str], 
    ncv: int = 3, 
    score: Union[_PredictScorer, str] = 'roc_auc') -> Dict[str, Any]:
    """
    Target function for optimization
    """
    
    model = get_estimator(hps=hps, cat_cols=cat_cols)
    print(model)
    cv_res = cross_val_score(model, X, y, cv=ncv, 
                             scoring=score,  n_jobs=-1)
    print(cv_res.mean())
    
    return {
        'loss': -cv_res.mean(), 
        'cv_std': cv_res.std(), 
        'status': STATUS_OK
    }

def train(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    cat_cols: List, 
    max_evals: int = 20, 
    score: Union[_PredictScorer, str] = 'roc_auc') -> Pipeline:
    """
    Retorna o melhor estimador computado usando cross-validation.
    """
    trials = Trials()
    best = fmin(partial(objective, X=X_train, y=y_train, cat_cols=cat_cols, score=score), 
                    get_search_space(), algo=tpe.suggest, max_evals=max_evals, 
                    trials=trials)
    best_hyperparams = space_eval(get_search_space(), best)
    model = get_estimator(hps=best_hyperparams, cat_cols=cat_cols)
    model.fit(X_train, y_train)
    return model