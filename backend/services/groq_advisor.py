import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


class GroqAdvisor:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[float, str]] = {}

    def is_configured(self) -> bool:
        return bool(os.getenv("GROQ_API_KEY"))

    @staticmethod
    def _sanitize_text(value: str) -> str:
        if value is None:
            return ""
        return re.sub(r"[^a-zA-Z0-9 _\-]", "", str(value)).strip()

    def _cache_key(self, method: str, system_prompt: str, user_message: str, max_tokens: int) -> str:
        payload = f"{method}|{max_tokens}|{system_prompt}|{user_message}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_cached(self, key: str) -> Optional[str]:
        hit = self._cache.get(key)
        if not hit:
            return None
        ts, value = hit
        if time.time() - ts > self.ttl_seconds:
            self._cache.pop(key, None)
            return None
        return value

    def _set_cached(self, key: str, value: str) -> None:
        self._cache[key] = (time.time(), value)

    async def _call_groq(self, system_prompt: str, user_message: str, max_tokens: int = 600, method: str = "generic") -> str:
        cache_key = self._cache_key(method, system_prompt, user_message, max_tokens)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        headers = {
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GROQ_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
            self._set_cached(cache_key, text)
            return text

    async def generate_bias_narrative(self, metrics: dict, sensitive_col: str, dataset_name: str) -> str:
        safe_sensitive_col = self._sanitize_text(sensitive_col) or "sensitive attribute"
        safe_dataset = self._sanitize_text(dataset_name) or "uploaded dataset"
        dp = metrics.get("demographic_parity_difference")
        eo = metrics.get("equal_opportunity_difference")
        di = metrics.get("disparate_impact_ratio")

        system_prompt = (
            "You are an AI fairness expert writing for C-suite executives and compliance officers "
            "who are not data scientists. Be precise, urgent, and use concrete analogies. "
            "Never use jargon without immediate plain-English explanation."
        )
        user_message = (
            f"Dataset: {safe_dataset}\n"
            f"Sensitive attribute: {safe_sensitive_col}\n"
            f"Demographic Parity Difference: {dp}\n"
            f"Equal Opportunity Difference: {eo}\n"
            f"Disparate Impact Ratio: {di}\n\n"
            "Write exactly 3 short paragraphs:\n"
            "1) Explain what these numbers mean in human terms.\n"
            "2) Identify the most alarming metric and reference the 80% rule for disparate impact.\n"
            "3) Explain urgency and real-world consequences of deploying this model now."
        )
        return await self._call_groq(system_prompt, user_message, max_tokens=400, method="bias_narrative")

    async def explain_shap_features(self, shap_data: List[dict], sensitive_col: str) -> str:
        safe_sensitive_col = self._sanitize_text(sensitive_col) or "sensitive attribute"
        top_features = []
        for item in shap_data[:8]:
            feat = self._sanitize_text(item.get("feature", ""))
            imp = item.get("importance", 0)
            top_features.append({"feature": feat, "importance": imp})

        system_prompt = (
            "You are an AI fairness specialist explaining model behavior to mixed technical and non-technical teams. "
            "Explain proxy discrimination clearly and concretely."
        )
        user_message = (
            f"Sensitive attribute: {safe_sensitive_col}\n"
            f"Top SHAP features: {json.dumps(top_features)}\n\n"
            "Write 2 concise paragraphs:\n"
            "1) Explain which features likely proxy the sensitive attribute.\n"
            "2) Flag if zipcode/zip code, education, job title/occupation style variables appear, and why risky."
        )
        return await self._call_groq(system_prompt, user_message, max_tokens=300, method="shap_insight")

    async def generate_counterfactual_story(self, counterfactual_examples: List[dict], sensitive_col: str) -> str:
        safe_sensitive_col = self._sanitize_text(sensitive_col) or "sensitive attribute"
        trimmed = counterfactual_examples[:3]
        system_prompt = (
            "You are writing human-centered fairness case vignettes for ethics and compliance readers. "
            "Be emotionally resonant but factual. Do not invent attributes not present in the data."
        )
        user_message = (
            f"Sensitive attribute: {safe_sensitive_col}\n"
            f"Counterfactual examples: {json.dumps(trimmed)}\n\n"
            "For each example, write 2-3 short sentences from the perspective of the affected individual. "
            "Keep details grounded in provided fields only."
        )
        return await self._call_groq(system_prompt, user_message, max_tokens=400, method="counterfactual_story")

    async def recommend_mitigation_strategy(self, simulation_results: List[dict], rl_recommendation: str, domain: str) -> str:
        safe_domain = self._sanitize_text(domain) or "General"
        safe_rec = self._sanitize_text(rl_recommendation)
        system_prompt = (
            "You are an AI governance advisor. Return practical action plans that engineering leaders can execute."
        )
        user_message = (
            f"Domain: {safe_domain}\n"
            f"RL recommended strategy: {safe_rec}\n"
            f"Simulation results: {json.dumps(simulation_results)}\n\n"
            "Return exactly 4 bullet points:\n"
            "- Immediate fix\n"
            "- 30-day plan\n"
            "- Compliance considerations\n"
            "- Monitoring cadence"
        )
        return await self._call_groq(system_prompt, user_message, max_tokens=500, method="mitigation_advice")

    async def generate_intersectional_insight(self, intersectional_data: List[dict]) -> str:
        system_prompt = (
            "You are an AI fairness analyst. Explain compound discrimination clearly for policy and legal audiences."
        )
        user_message = (
            f"Intersectional group approval data: {json.dumps(intersectional_data)}\n\n"
            "Identify the worst-performing subgroup and explain why this is compound discrimination "
            "(double-bind effect) in plain language."
        )
        return await self._call_groq(system_prompt, user_message, max_tokens=350, method="intersectional_insight")

    async def generate_audit_report_narrative(self, full_audit_payload: Dict[str, Any]) -> str:
        system_prompt = (
            "You are an AI fairness report writer producing executive-grade narratives for compliance documentation."
        )
        user_message = (
            f"Full audit payload: {json.dumps(full_audit_payload)}\n\n"
            "Write a 500-700 word report with these section headers:\n"
            "Overview\nKey Findings\nRisk Assessment\nRecommended Actions\nCompliance Notes\n"
            "Use concrete numbers from the payload and no hallucinated metrics."
        )
        return await self._call_groq(system_prompt, user_message, max_tokens=800, method="full_report")
