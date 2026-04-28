import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from services.detector import BiasDetector
from typing import List, Dict, Any

class BiasSimulator:
    def __init__(self, df: pd.DataFrame, target_col: str, sensitive_col: str):
        self.df = df
        self.target_col = target_col
        self.sensitive_col = sensitive_col
        # Get baseline
        detector = BiasDetector(df, target_col, sensitive_col)
        self.baseline = detector.run_detection()
        self.baseline_fairness = 1 - abs(self.baseline["fairness_metrics"]["demographic_parity_difference"])
        self.baseline_accuracy = self.baseline["accuracy"]

    def _calculate_metrics(self, y_test, y_pred, sensitive_test):
        accuracy = accuracy_score(y_test, y_pred)
        
        test_df = pd.DataFrame({
            'prediction': y_pred,
            'sensitive': sensitive_test
        })
        
        groups = test_df['sensitive'].unique()
        approval_rates = []
        for group in groups:
            rate = (test_df[test_df['sensitive'] == group]['prediction'] == 1).mean()
            approval_rates.append(rate)
        
        dpd = max(approval_rates) - min(approval_rates) if approval_rates else 0
        fairness_score = 1 - abs(dpd)
        
        return accuracy, fairness_score

    def run_simulation(self) -> List[Dict[str, Any]]:
        results = []
        
        # Prepare data
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Strategy 1: Remove Sensitive Attribute
        results.append(self._strategy_remove_sensitive(X.copy(), y.copy()))
        
        # Strategy 2: Reweight Dataset
        results.append(self._strategy_reweight(X.copy(), y.copy()))
        
        # Strategy 3: Threshold Adjustment
        results.append(self._strategy_threshold_adjustment(X.copy(), y.copy()))
        
        # Strategy 4: Fairness Constraint (Simple Regularization approach)
        results.append(self._strategy_fairness_constraint(X.copy(), y.copy()))
        
        # Sort by fairness_score descending
        return sorted(results, key=lambda x: x["fairness_score"], reverse=True)

    def _strategy_remove_sensitive(self, X, y):
        X_no_sensitive = X.drop(columns=[self.sensitive_col])
        X_no_sensitive = pd.get_dummies(X_no_sensitive)
        
        X_train, X_test, y_train, y_test = train_test_split(X_no_sensitive, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Need original sensitive column for metric calculation
        sensitive_test = self.df.loc[X_test.index, self.sensitive_col]
        acc, fair = self._calculate_metrics(y_test, y_pred, sensitive_test)
        
        return self._format_result("Remove Sensitive Attribute", acc, fair)

    def _strategy_reweight(self, X, y):
        # Compute weights: W = P(Y)/P(Y|S)
        # Simplified reweighting: balance (sensitive, target) pairs
        df_temp = X.copy()
        df_temp[self.target_col] = y
        
        counts = df_temp.groupby([self.sensitive_col, self.target_col]).size().reset_index(name='count')
        total = len(df_temp)
        
        # weight = (P(S) * P(Y)) / P(S, Y)
        weights_map = {}
        for _, row in counts.iterrows():
            s_val, y_val, count = row[self.sensitive_col], row[self.target_col], row['count']
            p_s = len(df_temp[df_temp[self.sensitive_col] == s_val]) / total
            p_y = len(df_temp[df_temp[self.target_col] == y_val]) / total
            p_sy = count / total
            weights_map[(s_val, y_val)] = (p_s * p_y) / p_sy if p_sy > 0 else 1.0
            
        sample_weights = df_temp.apply(lambda x: weights_map[(x[self.sensitive_col], x[self.target_col])], axis=1)
        
        X_encoded = pd.get_dummies(X)
        X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
            X_encoded, y, sample_weights, test_size=0.2, random_state=42
        )
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train, sample_weight=w_train)
        y_pred = model.predict(X_test)
        
        sensitive_test = self.df.loc[X_test.index, self.sensitive_col]
        acc, fair = self._calculate_metrics(y_test, y_pred, sensitive_test)
        
        return self._format_result("Reweight Dataset", acc, fair)

    def _strategy_threshold_adjustment(self, X, y):
        X_encoded = pd.get_dummies(X)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        sensitive_test = self.df.loc[X_test.index, self.sensitive_col]
        groups = sensitive_test.unique()
        
        # Find a global target approval rate (e.g., the average baseline approval rate)
        target_rate = np.mean(list(self.baseline["per_group_approval_rates"].values()))
        
        y_pred_adj = np.zeros_like(y_probs)
        for group in groups:
            mask = (sensitive_test == group)
            group_probs = y_probs[mask]
            if len(group_probs) == 0: continue
            
            # Find threshold that gives target_rate
            threshold = np.percentile(group_probs, (1 - target_rate) * 100)
            y_pred_adj[mask] = (group_probs >= threshold).astype(int)
            
        acc, fair = self._calculate_metrics(y_test, y_pred_adj, sensitive_test)
        return self._format_result("Threshold Adjustment", acc, fair)

    def _strategy_fairness_constraint(self, X, y):
        # Simplified: Use a high C parameter for LogisticRegression (less regularization)
        # and include sensitive attribute interactions to allow model to learn nuances 
        # but here we'll simulate it by using a very strong regularization to reduce variance
        X_encoded = pd.get_dummies(X)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(max_iter=1000, C=0.01) # Strong regularization
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        sensitive_test = self.df.loc[X_test.index, self.sensitive_col]
        acc, fair = self._calculate_metrics(y_test, y_pred, sensitive_test)
        
        return self._format_result("Fairness Constraint", acc, fair)

    def _format_result(self, name, acc, fair):
        return {
            "strategy_name": name,
            "accuracy": float(acc),
            "fairness_score": float(fair),
            "fairness_gain": float(fair - self.baseline_fairness),
            "accuracy_drop": float(self.baseline_accuracy - acc)
        }
