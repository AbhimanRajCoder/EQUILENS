import pandas as pd
import numpy as np
import shap
from typing import List, Dict, Any

class BiasExplainer:
    def __init__(self, model: Any, X_train: pd.DataFrame, feature_names: List[str], df: pd.DataFrame, sensitive_col: str):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.df = df
        self.sensitive_col = sensitive_col

    def get_explanations(self) -> Dict[str, Any]:
        # Initialize SHAP explainer
        # Since we're using LogisticRegression, LinearExplainer is appropriate
        explainer = shap.LinearExplainer(self.model, self.X_train)
        shap_values = explainer.shap_values(self.X_train)

        # Handle various SHAP output formats
        if isinstance(shap_values, list):
            # If it's a list, it's likely [neg_class_values, pos_class_values]
            # Take values for the positive class (usually index 1)
            # If list has only one element, take that
            shap_v = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif hasattr(shap_values, "values"):
            # Newer SHAP versions return an Explanation object
            shap_v = shap_values.values
            if len(shap_v.shape) == 3: # (samples, features, classes)
                shap_v = shap_v[:, :, 1]
        else:
            shap_v = shap_values

        # Ensure shap_v is a numpy array and handle multi-dimensional outputs
        shap_v = np.array(shap_v)
        if len(shap_v.shape) == 3:
            shap_v = shap_v[:, :, 1]
        
        # Compute mean absolute SHAP values for feature importance
        # Ensure we're taking the mean across samples (axis 0)
        mean_abs_shap = np.abs(shap_v).mean(axis=0)
        
        # In case mean_abs_shap is still multi-dimensional (shouldn't be)
        if hasattr(mean_abs_shap, "flatten"):
            mean_abs_shap = mean_abs_shap.flatten()

        # Create list of {feature, importance} dicts
        feature_importance = []
        for name, importance in zip(self.feature_names, mean_abs_shap):
            try:
                # Use np.asarray().item() for safer scalar conversion if it's a numpy type
                val = float(importance.item()) if hasattr(importance, "item") else float(importance)
                feature_importance.append({"feature": name, "importance": val})
            except Exception:
                # Fallback for unexpected types
                feature_importance.append({"feature": name, "importance": 0.0})
        
        # Sort and take top 5
        top_5_features = sorted(feature_importance, key=lambda x: x["importance"], reverse=True)[:5]

        # Compute group comparison: mean prediction probability per group
        y_prob = self.model.predict_proba(self.X_train)[:, 1]
        
        temp_df = pd.DataFrame({
            'prob': y_prob,
            'sensitive': self.df.loc[self.X_train.index, self.sensitive_col]
        })
        
        # Calculate per-group SHAP importance
        per_group_shap = {}
        unique_groups = temp_df['sensitive'].unique()
        
        for group in unique_groups:
            group_indices = temp_df[temp_df['sensitive'] == group].index
            # Map original df indices to X_train integer positions
            # X_train.index is the original indices.
            # We need to find the integer positions of group_indices in X_train.index
            pos = [self.X_train.index.get_loc(idx) for idx in group_indices]
            group_shap_v = shap_v[pos]
            
            mean_abs_group_shap = np.abs(group_shap_v).mean(axis=0)
            if hasattr(mean_abs_group_shap, "flatten"):
                mean_abs_group_shap = mean_abs_group_shap.flatten()
            
            group_importance = []
            for name, importance in zip(self.feature_names, mean_abs_group_shap):
                try:
                    val = float(importance.item()) if hasattr(importance, "item") else float(importance)
                    group_importance.append({"feature": name, "importance": val})
                except Exception:
                    group_importance.append({"feature": name, "importance": 0.0})
            
            per_group_shap[str(group)] = sorted(group_importance, key=lambda x: x["importance"], reverse=True)[:5]

        group_comparison_series = temp_df.groupby('sensitive')['prob'].mean()
        group_comparison = {}
        for k, v in group_comparison_series.items():
            try:
                val = float(v.item()) if hasattr(v, "item") else float(v)
                group_comparison[str(k)] = val
            except Exception:
                group_comparison[str(k)] = 0.0

        return {
            "shap_features": top_5_features,
            "per_group_shap": per_group_shap,
            "group_comparison": group_comparison
        }
