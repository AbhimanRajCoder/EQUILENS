import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, 
    precision_recall_curve, auc
)
from sklearn.calibration import calibration_curve
from scipy.stats import chi2_contingency
from typing import List, Optional, Dict, Any

class BiasDetector:
    def __init__(self, df: pd.DataFrame, target_col: str, sensitive_col: str):
        self.df = df
        self.target_col = target_col
        self.sensitive_col = sensitive_col
        self._validate_columns()
        self.model = LogisticRegression(max_iter=1000)
        # Populated by run_detection() for reuse in downstream analyses
        self._test_df = None
        self._X_test = None
        self._X_train = None
        self._feature_names = None
        self._per_group_approval_rates = None

    def _validate_columns(self):
        """Check for mixed data types and non-binary target in target and sensitive columns."""
        # 1. Check for mixed types
        for col_name in [self.target_col, self.sensitive_col]:
            col = self.df[col_name]
            # Check for mixed types (int and str)
            # We use .map(type) instead of apply for better performance on large DFs
            types = col.map(type).unique()
            type_names = [t.__name__ for t in types]
            
            if 'int' in type_names and 'str' in type_names:
                 role = "Target Column (Y)" if col_name == self.target_col else "Sensitive Attribute (S)"
                 raise TypeError(
                     f"'<' not supported between instances of 'int' and 'str' in {role} '{col_name}'. "
                     f"This usually happens when a column contains mixed numeric and string values (e.g., 1 and '1'). "
                     f"Suggestion: Convert '{col_name}' to a consistent type. "
                     f"Click 'Fix Automatically' to convert this column to strings."
                 )
        
        # 2. Check that target column is binary (0/1) or has only 2 unique values
        target_values = self.df[self.target_col].unique()
        if len(target_values) > 2:
            raise ValueError(
                f"Target Column (Y) '{self.target_col}' has {len(target_values)} unique values. "
                f"BiasGuard currently only supports binary classification (e.g., 0 and 1). "
                f"Suggestion: Map your target values to 0 and 1 before uploading."
            )

    def run_detection(self):
        # Prepare data
        # Handle categorical columns by dummy encoding except for target and sensitive
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Simple encoding for demonstration: convert all non-numeric to dummies
        X = pd.get_dummies(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Metrics calculation
        # Re-attach sensitive attribute and target to test set for fairness analysis
        test_df = X_test.copy()
        test_df[self.target_col] = y_test
        test_df['prediction'] = y_pred
        test_df['probability'] = y_prob
        
        # We need the original sensitive column values. 
        # If it was dummy encoded, we'll try to find it in the original df using index
        test_df[self.sensitive_col] = self.df.loc[X_test.index, self.sensitive_col]

        # Store for downstream reuse (intersectional + counterfactual)
        self._test_df = test_df
        self._X_test = X_test
        self._X_train = X_train
        self._feature_names = X.columns.tolist()
        
        groups = test_df[self.sensitive_col].unique()
        per_group_approval_rates = {}
        tpr_per_group = {} # True Positive Rate for Equal Opportunity
        fpr_per_group = {} # False Positive Rate
        
        for group in groups:
            group_data = test_df[test_df[self.sensitive_col] == group]
            # Approval rate: P(Y_pred = 1 | Sensitive = group)
            approval_rate = (group_data['prediction'] == 1).mean()
            
            # Robust scalar conversion
            try:
                val = float(approval_rate.item()) if hasattr(approval_rate, "item") else float(approval_rate)
            except Exception:
                val = float(approval_rate)
            per_group_approval_rates[str(group)] = val
            
            # TPR: P(Y_pred = 1 | Y = 1, Sensitive = group)
            positives = group_data[group_data[self.target_col] == 1]
            if len(positives) > 0:
                tpr = (positives['prediction'] == 1).mean()
            else:
                tpr = 0.0
            
            # FPR: P(Y_pred = 1 | Y = 0, Sensitive = group)
            negatives = group_data[group_data[self.target_col] == 0]
            if len(negatives) > 0:
                fpr = (negatives['prediction'] == 1).mean()
            else:
                fpr = 0.0

            try:
                tpr_val = float(tpr.item()) if hasattr(tpr, "item") else float(tpr)
                fpr_val = float(fpr.item()) if hasattr(fpr, "item") else float(fpr)
            except Exception:
                tpr_val = float(tpr)
                fpr_val = float(fpr)
            
            tpr_per_group[str(group)] = tpr_val
            fpr_per_group[str(group)] = fpr_val

        self._per_group_approval_rates = per_group_approval_rates

        # Calculate Fairness Scores
        # 1. Demographic Parity Difference: max(approval_rate) - min(approval_rate)
        rates = list(per_group_approval_rates.values())
        dpd = max(rates) - min(rates) if rates else 0.0
        
        # 2. Equal Opportunity Difference: max(tpr) - min(tpr)
        tprs = list(tpr_per_group.values())
        eod = max(tprs) - min(tprs) if tprs else 0.0
        
        # 3. Equalized Odds Difference: max(max(tpr_diff), max(fpr_diff))
        fprs = list(fpr_per_group.values())
        max_tpr_diff = max(tprs) - min(tprs) if tprs else 0.0
        max_fpr_diff = max(fprs) - min(fprs) if fprs else 0.0
        equalized_odds_diff = max(max_tpr_diff, max_fpr_diff)

        # 4. Average Odds Difference: 0.5 * ( (TPR_diff) + (FPR_diff) )
        avg_odds_diff = 0.5 * (max_tpr_diff + max_fpr_diff)

        # 5. Disparate Impact Ratio: min(approval_rate) / max(approval_rate)
        max_rate = max(rates) if rates else 0.0
        min_rate = min(rates) if rates else 0.0
        dir_score = min_rate / max_rate if max_rate > 0 else 1.0
        
        # 6. Statistical Significance (P-value for Demographic Parity using Chi-squared)
        p_values = {}
        try:
            contingency_table = pd.crosstab(test_df[self.sensitive_col], test_df['prediction'])
            chi2, p_val, _, _ = chi2_contingency(contingency_table)
            p_values["demographic_parity"] = float(p_val)
        except Exception:
            p_values["demographic_parity"] = 1.0

        # Return results along with components needed for explainer
        return {
            "accuracy": float(accuracy.item()) if hasattr(accuracy, "item") else float(accuracy),
            "per_group_approval_rates": per_group_approval_rates,
            "fairness_metrics": {
                "demographic_parity_difference": float(dpd),
                "equal_opportunity_difference": float(eod),
                "equalized_odds_difference": float(equalized_odds_diff),
                "average_odds_difference": float(avg_odds_diff),
                "disparate_impact_ratio": float(dir_score),
                "p_values": p_values
            },
            "model": self.model,
            "X_train": X_train,
            "feature_names": X.columns.tolist(),
            "advanced_metrics": self.get_advanced_metrics()
        }

    def get_advanced_metrics(self) -> Dict[str, Any]:
        """Calculate researcher-level metrics like per-group ROC/PR curves and Confusion Matrices."""
        if self._test_df is None:
            return {}

        test_df = self._test_df
        groups = test_df[self.sensitive_col].unique()
        per_group_metrics = {}
        
        # Representation bias calculation
        counts = test_df[self.sensitive_col].value_counts()
        total = len(test_df)
        representation_bias = {str(k): float(v/total) for k, v in counts.items()}

        for group in groups:
            group_data = test_df[test_df[self.sensitive_col] == group]
            y_true = group_data[self.target_col]
            y_pred = group_data['prediction']
            y_prob = group_data['probability']

            # 1. Confusion Matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            # 2. ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_data = [{"x": float(f), "y": float(t)} for f, t in zip(fpr, tpr)]

            # 3. Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_data = [{"x": float(r), "y": float(p)} for p, r in zip(precision, recall)]

            # 4. Calibration Curve
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            cal_data = [{"x": float(pp), "y": float(pt)} for pt, pp in zip(prob_true, prob_pred)]

            # 5. Score Distribution (for histograms)
            score_dist = y_prob.tolist()

            per_group_metrics[str(group)] = {
                "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
                "roc_curve": roc_data,
                "pr_curve": pr_data,
                "calibration_curve": cal_data,
                "score_distribution": score_dist
            }

        return {
            "per_group_metrics": per_group_metrics,
            "representation_bias": representation_bias
        }

    # ------------------------------------------------------------------
    # Counterfactual explanation
    # ------------------------------------------------------------------

    def run_counterfactual_analysis(
        self,
        top_shap_features: List[str],
        max_examples: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations by flipping the sensitive
        attribute from the disadvantaged group to the advantaged group.

        Must be called **after** ``run_detection()``.

        Parameters
        ----------
        top_shap_features : list[str]
            The top-N SHAP feature names (from the explainer) to include in
            ``original_features`` for each example.
        max_examples : int
            Return at most this many examples where the outcome changed.

        Returns
        -------
        list[dict]
            Each dict contains:
            - original_features  (dict of top-SHAP feature values)
            - original_prediction (int, 0)
            - counterfactual_prediction (int, 1)
            - sensitive_attr_original (str)
            - sensitive_attr_flipped (str)
        """
        if self._test_df is None or self._per_group_approval_rates is None:
            raise RuntimeError(
                "run_detection() must be called before run_counterfactual_analysis()"
            )

        approval = self._per_group_approval_rates  # {"Male": 0.4, "Female": 0.12}

        # Identify disadvantaged (lowest rate) and advantaged (highest rate) groups
        disadvantaged_group = min(approval, key=approval.get)
        advantaged_group = max(approval, key=approval.get)

        if disadvantaged_group == advantaged_group:
            return []  # no disparity → nothing to show

        # --- 1. Sample up to 5 negative-outcome rows from the disadvantaged group ---
        test_df = self._test_df
        candidates = test_df[
            (test_df[self.sensitive_col].astype(str) == str(disadvantaged_group))
            & (test_df["prediction"] == 0)
        ]

        if candidates.empty:
            return []

        sample_n = min(5, len(candidates))
        sampled = candidates.sample(n=sample_n, random_state=42)

        # --- 2. For each row, flip the sensitive attr and re-predict ---
        results: List[Dict[str, Any]] = []
        feature_cols = self._feature_names  # dummy-encoded column list

        for idx in sampled.index:
            # Grab original raw row from the source dataframe
            orig_raw = self.df.loc[idx].copy()

            # Build the counterfactual raw row with the sensitive attr flipped
            cf_raw = orig_raw.copy()
            cf_raw[self.sensitive_col] = advantaged_group

            # Encode both through the same dummy pipeline
            orig_row_df = pd.DataFrame([orig_raw]).drop(columns=[self.target_col])
            cf_row_df = pd.DataFrame([cf_raw]).drop(columns=[self.target_col])

            orig_encoded = pd.get_dummies(orig_row_df).reindex(
                columns=feature_cols, fill_value=0
            )
            cf_encoded = pd.get_dummies(cf_row_df).reindex(
                columns=feature_cols, fill_value=0
            )

            orig_pred = int(self.model.predict(orig_encoded)[0])
            cf_pred = int(self.model.predict(cf_encoded)[0])

            # We only care about flips from 0 → 1
            if orig_pred == 0 and cf_pred == 1:
                # Build the display-friendly feature dict (top SHAP features only)
                original_features = {}
                for feat in top_shap_features[:5]:
                    # SHAP feature names may be dummy-encoded (e.g. "sex_Female").
                    # Try to map back to the raw column value for readability.
                    raw_col = feat.split("_")[0] if feat not in orig_raw.index else feat
                    if raw_col in orig_raw.index:
                        val = orig_raw[raw_col]
                        original_features[raw_col] = (
                            int(val) if isinstance(val, (np.integer,)) else
                            float(val) if isinstance(val, (np.floating, float)) else
                            str(val)
                        )
                    elif feat in orig_encoded.columns:
                        original_features[feat] = int(orig_encoded[feat].iloc[0])

                results.append({
                    "original_features": original_features,
                    "original_prediction": 0,
                    "counterfactual_prediction": 1,
                    "sensitive_attr_original": str(disadvantaged_group),
                    "sensitive_attr_flipped": str(advantaged_group),
                })

                if len(results) >= max_examples:
                    break

        return results

    # ------------------------------------------------------------------
    # Intersectional analysis
    # ------------------------------------------------------------------

    def run_intersectional_analysis(
        self,
        intersectional_cols: List[str],
        min_sample_size: int = 30
    ) -> List[dict]:
        """Compute positive-prediction rates for every unique combination of
        values across *intersectional_cols*.

        Must be called **after** ``run_detection()`` so that the trained model
        and test-set predictions are available.

        Groups with fewer than *min_sample_size* samples in the test set are
        filtered out to avoid spurious / noisy results.

        Returns a list of dicts:
            [{ "group": "Female + Black", "approval_rate": 0.12, "sample_size": 42 }, ...]
        """
        if self._test_df is None:
            raise RuntimeError(
                "run_detection() must be called before run_intersectional_analysis()"
            )

        test_df = self._test_df.copy()

        # Re-attach the original (pre-dummy-encoded) intersectional columns
        # from the source dataframe using the test index.
        for col in intersectional_cols:
            if col not in test_df.columns:
                test_df[col] = self.df.loc[self._X_test.index, col].values

        # Build a combined group label, e.g. "Female + Black"
        test_df["_intersect_group"] = test_df[intersectional_cols].astype(str).agg(
            " + ".join, axis=1
        )

        results: List[dict] = []
        for group_label, group_df in test_df.groupby("_intersect_group"):
            n = len(group_df)
            if n < min_sample_size:
                continue

            rate = (group_df["prediction"] == 1).mean()
            # Robust scalar conversion
            try:
                rate_val = float(rate.item()) if hasattr(rate, "item") else float(rate)
            except Exception:
                rate_val = float(rate)

            results.append({
                "group": str(group_label),
                "approval_rate": rate_val,
                "sample_size": int(n),
            })

        # Sort by approval rate ascending so the chart reads bottom-to-top
        results.sort(key=lambda r: r["approval_rate"])
        return results
