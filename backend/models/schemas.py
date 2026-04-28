from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class BiasMetrics(BaseModel):
    demographic_parity_difference: float
    equal_opportunity_difference: float
    equalized_odds_difference: float
    average_odds_difference: float
    disparate_impact_ratio: float
    p_values: Dict[str, float]

class IntersectionalGroup(BaseModel):
    group: str
    approval_rate: float
    sample_size: int

class CounterfactualExample(BaseModel):
    original_features: Dict[str, Any]
    original_prediction: int
    counterfactual_prediction: int
    sensitive_attr_original: str
    sensitive_attr_flipped: str

class CurvePoint(BaseModel):
    x: float
    y: float

class ConfusionMatrix(BaseModel):
    tp: int
    tn: int
    fp: int
    fn: int

class GroupAdvancedMetrics(BaseModel):
    confusion_matrix: ConfusionMatrix
    roc_curve: List[CurvePoint]
    pr_curve: List[CurvePoint]
    calibration_curve: List[CurvePoint]
    score_distribution: List[float]

class AdvancedMetrics(BaseModel):
    per_group_metrics: Dict[str, GroupAdvancedMetrics]
    representation_bias: Dict[str, float] # Percentage of each group

class BiasDetectionResponse(BaseModel):
    accuracy: float
    per_group_approval_rates: Dict[str, float]
    fairness_metrics: BiasMetrics
    shap_features: List[FeatureImportance]
    per_group_shap: Optional[Dict[str, List[FeatureImportance]]] = None
    group_comparison: Dict[str, float]
    intersectional_bias: Optional[List[IntersectionalGroup]] = None
    counterfactual_examples: Optional[List[CounterfactualExample]] = None
    groq_narrative: Optional[str] = None
    groq_shap_insight: Optional[str] = None
    advanced_metrics: Optional[AdvancedMetrics] = None

class SimulationResult(BaseModel):
    strategy_name: str
    accuracy: float
    fairness_score: float
    fairness_gain: float
    accuracy_drop: float

class SimulationResponse(BaseModel):
    strategies: List[SimulationResult]

class RecommendationRequest(BaseModel):
    bias_score: float
    accuracy: float

class StrategyScore(BaseModel):
    strategy_name: str
    score: float

class RunnerUp(BaseModel):
    strategy_name: str
    score: float
    reason: str

class RecommendationResponse(BaseModel):
    recommended_strategy: str
    expected_fairness_gain: float
    expected_accuracy_drop: float
    reason: str
    all_scores: List[StrategyScore]
    runner_up: RunnerUp

class DatasetMetadata(BaseModel):
    filename: str
    target_col: str
    sensitive_col: str

class AuditExportRequest(BaseModel):
    timestamp: str
    metadata: DatasetMetadata
    baseline_metrics: BiasDetectionResponse
    simulation_results: Optional[List[SimulationResult]] = None
    recommendation: Optional[RecommendationResponse] = None
    counterfactual_examples: Optional[List[CounterfactualExample]] = None
    include_groq_narrative: Optional[bool] = False
