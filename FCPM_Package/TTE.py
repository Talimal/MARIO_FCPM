import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from sklearn.linear_model import GammaRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error # These aren't used here but kept imports

class TTE:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-6):
        """
        Initializes the Time-to-Event (TTE) model using Gamma Regressor
        and includes feature scaling.

        Parameters:
        - alpha: Regularization strength (default: 1.0). Recommend using 0 or small value.
        - max_iter: Maximum iterations for convergence (default: 1000).
        - tol: Convergence tolerance (default: 1e-6).
        """
        self.model = GammaRegressor(alpha=alpha, max_iter=max_iter, tol=tol)
        # Initialize the scaler
        self.scaler = StandardScaler()
        self.fitted = False
        # Optional: Store feature names if input is DataFrame during fit
        self.feature_names_in_ = None

    def fit(self, X, y):
        """
        Fits the StandardScaler and then the Gamma Regressor model.

        Parameters:
        - X: Feature matrix (pandas DataFrame or numpy array).
        - y: Target variable (time-to-event values).

        Returns:
        - self: The fitted TTE object instance.
        """
        try:
            # Store feature names if X is a DataFrame
            if isinstance(X, pd.DataFrame):
                self.feature_names_in_ = X.columns.tolist()
                X_input = X.to_numpy() # Convert to numpy for scaler
            else:
                X_input = np.asarray(X)
                # If X is numpy, we don't get feature names directly
                # self.feature_names_in_ could be set based on X_input.shape[1] if needed

            print("Scaling features using StandardScaler...")
            # Fit the scaler to the training data and transform it
            X_scaled = self.scaler.fit_transform(X_input)
            print(f"Scaled X shape: {X_scaled.shape}")

            print(f"Fitting GammaRegressor (alpha={self.model.alpha}) on SCALED data...")
            # Fit the GammaRegressor model on the scaled data
            self.model.fit(X_scaled, y)
            self.fitted = True
            print("Model fitting complete.")

            # You can check the coefficients here (they relate to scaled features)
            if hasattr(self.model, 'coef_'):
                print(f"Fitted coefficients (on scaled features): {self.model.coef_}")
            if hasattr(self.model, 'intercept_'):
                print(f"Fitted intercept (on scaled features): {self.model.intercept_}")

        except Exception as e:
            print(f"Error during TTE fitting/scaling: {e}")
            import traceback
            traceback.print_exc()
            self.fitted = False # Ensure fitting status is False on error

        # Return the TTE instance itself, allowing chaining or inspection
        return self

    def predict(self, X):
        """
        Scales the input features and predicts the time-to-event.

        Parameters:
        - X: Feature matrix (pandas DataFrame or numpy array).

        Returns:
        - Predicted time-to-event values (numpy array).
        """
        if not self.fitted:
            raise ValueError("Model is not trained. Call `fit` first.")
        try:
            # Ensure X is in the right format (numpy array)
            if isinstance(X, pd.DataFrame):
                # Optional: Check if prediction columns match training columns
                if self.feature_names_in_ and X.columns.tolist() != self.feature_names_in_:
                     print("Warning: Prediction feature names differ from training names.")
                X_input = X.to_numpy()
            else:
                X_input = np.asarray(X)

            # Apply the SAME scaling transformation learned during fit
            # print("Scaling prediction data using fitted scaler...") # Optional print
            X_scaled = self.scaler.transform(X_input) # <<< Use transform()

            # Predict using the scaled data
            # print("Predicting with GammaRegressor on scaled data...") # Optional print
            predictions = self.model.predict(X_scaled)
            return predictions

        except Exception as e:
             print(f"Error during TTE prediction/scaling: {e}")
             import traceback
             traceback.print_exc()
             # Return NaNs or raise error
             return np.full(X.shape[0] if hasattr(X, 'shape') else 1, np.nan)