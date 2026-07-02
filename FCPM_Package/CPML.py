import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import logging

class CPML:
    def __init__(self, **xgb_kwargs):
        """
        Initializes the CPML wrapper using XGBoost.
        Acts exactly like the FCPM class for continuous prediction.
        """
        self.feature_columns = None
        
        # Use XGBClassifier configured to output probabilities
        # 'binary:logistic' ensures the output is a probability between 0 and 1
        self.ml_model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            **xgb_kwargs
        )
        
        # Columns that are metadata and should NOT be used as ML features.
        self.metadata_cols = [
            'EntityID', 'instance_ID', 'instance_start_time', 'TFS', 
            'current_time', 'outcome_class', 'event_time', 'TTE'
        ]

    def fit(self, X_train_df, y_train_series):
        """
        Fits the XGBoost model on the flattened continuous feature matrix.
        Automatically filters out metadata columns so only predictive features are used.
        """
        # 1. Isolate only the actual features (exclude metadata)
        feature_cols = [col for col in X_train_df.columns if col not in self.metadata_cols]
        self.feature_columns = feature_cols
        
        X_train_clean = X_train_df[self.feature_columns]
        
        # 2. Train the XGBoost model
        self.ml_model.fit(X_train_clean, y_train_series)

        logging.info(f"CPML model trained with {len(self.feature_columns)} features.")

    def fit_matrix(self, X, y, feature_names):
        """
        Fits the XGBoost model directly on a compact NumPy feature matrix (no DataFrame).

        Use this with CPM_Feature_Matrix.Create_feature_matrix.build_cpml_training_arrays,
        which returns (X, y, feature_names) as integer arrays. This avoids the large
        list-of-dicts/DataFrame that the DataFrame `fit` path builds, so training stays
        memory-lean on big TIRPs.

        :param X: np.ndarray (n_rows, n_features) integer feature matrix.
        :param y: np.ndarray (n_rows,) binary outcome labels.
        :param feature_names: ordered list of the columns in X (stored for prediction).
        """
        self.feature_columns = list(feature_names)
        self.ml_model.fit(X, y)
        logging.info(
            f"CPML model trained from array with {len(self.feature_columns)} features, "
            f"{X.shape[0]} rows."
        )

    def predict(self, timestamp, entity_id, df_per_timestamp, epsilon=None):
        """
        Predicts the probability of the event occurring.
        Signature matches FCPM EXACTLY.
        """
        if df_per_timestamp is None or df_per_timestamp.empty:
            return []
            
        # 1. Align features
        features_df = df_per_timestamp.copy()

        # Ensure all trained features exist in the current timestamp data.
        # If a TIEP feature is missing, it means it hasn't occurred yet, so we fill with 0.0
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0.0
                
        # Keep only the trained features in the exact same order.
        # Convert to a NumPy array so a model trained on a nameless array (fit_matrix)
        # does not raise an XGBoost feature-name mismatch against a named DataFrame.
        X_infer = features_df[self.feature_columns].to_numpy(dtype=np.int32)

        # 2. Predict probability (returns array of shape [n_samples, 2])
        # We extract the second column (index 1) which is the probability of class 1
        probs = self.ml_model.predict_proba(X_infer)[:, 1]
        
        # 3. Format output exactly like FCPM
        result = []
        # FIX: Use enumerate to get a 0-based counter (i) that matches the 'probs' array
        for i, (index, row) in enumerate(df_per_timestamp.iterrows()):
            inst_id = row.get('instance_ID', -1)
            tfs = row.get('TFS', timestamp)
            
            result.append({
                'EntityID': entity_id,
                'instance_ID': inst_id,
                'TFS': tfs,
                'FCPM_Prediction': probs[i]  # Use 'i' instead of 'index'
            })
            
        return result