#!/usr/bin/env python3
"""
smoke_test.py — End-to-end smoke test for the EquiLens AI pipeline.

Validates /api/detect, /api/simulate, and /api/recommend against the
bundled adult_sample.csv.  Requires the backend to be running on port 8000.

Usage:
    cd backend/
    python scripts/smoke_test.py
"""

import os
import sys
import time
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CSV = os.path.join(BACKEND_DIR, "data", "adult_sample.csv")

TARGET_COL = "income"
SENSITIVE_COL = "sex"

# Timing thresholds (seconds)
DETECT_TIMEOUT = 5
SIMULATE_TIMEOUT = 10
RECOMMEND_TIMEOUT = 3

# Fairness thresholds
DEMO_DPD_THRESHOLD = 0.15          # demographic parity diff strong enough to demo
MIN_FAIRNESS_IMPROVEMENT = 0.10    # 10% minimum improvement from recommendation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

def ok(msg):
    print(f"  {Colors.GREEN}✓{Colors.RESET} {msg}")

def fail(msg):
    print(f"  {Colors.RED}✗{Colors.RESET} {msg}")

def warn(msg):
    print(f"  {Colors.YELLOW}⚠{Colors.RESET} {msg}")

def header(title):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}{Colors.RESET}")

def timed_request(method, url, timeout_sec, **kwargs):
    """Make a request, return (response, elapsed_seconds, passed_timing)."""
    t0 = time.time()
    resp = method(url, **kwargs)
    elapsed = time.time() - t0
    passed = elapsed < timeout_sec
    return resp, elapsed, passed


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
def preflight():
    header("Pre-flight")

    if not os.path.exists(SAMPLE_CSV):
        fail(f"Sample CSV not found: {SAMPLE_CSV}")
        sys.exit(1)
    ok(f"Sample CSV exists ({SAMPLE_CSV})")

    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        r.raise_for_status()
        ok(f"Backend reachable at {API_BASE}")
    except Exception as e:
        fail(f"Cannot reach backend at {API_BASE}: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1 — /api/detect
# ---------------------------------------------------------------------------
def test_detect():
    header("Step 1 · POST /api/detect")

    with open(SAMPLE_CSV, "rb") as f:
        resp, elapsed, timing_ok = timed_request(
            requests.post,
            f"{API_BASE}/api/detect",
            DETECT_TIMEOUT,
            files={"file": ("adult_sample.csv", f, "text/csv")},
            data={
                "target_col": TARGET_COL,
                "sensitive_col": SENSITIVE_COL,
                "intersectional_cols": "sex,race",
            },
        )

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    ok(f"Status 200  ({elapsed:.2f}s)")

    if timing_ok:
        ok(f"Response time {elapsed:.2f}s < {DETECT_TIMEOUT}s")
    else:
        fail(f"Response time {elapsed:.2f}s ≥ {DETECT_TIMEOUT}s")

    data = resp.json()

    # Fairness metrics
    fm = data["fairness_metrics"]
    dpd = fm["demographic_parity_difference"]
    eod = fm["equal_opportunity_difference"]
    di  = fm["disparate_impact_ratio"]

    print(f"\n  {'Metric':<35} {'Value':>8}")
    print(f"  {'─' * 45}")
    print(f"  {'Demographic Parity Difference':<35} {dpd:>8.4f}")
    print(f"  {'Equal Opportunity Difference':<35} {eod:>8.4f}")
    print(f"  {'Disparate Impact Ratio':<35} {di:>8.4f}")
    print(f"  {'Model Accuracy':<35} {data['accuracy']:>8.4f}")

    if dpd > DEMO_DPD_THRESHOLD:
        ok(f"DPD {dpd:.4f} > {DEMO_DPD_THRESHOLD} — strong enough for demo")
    else:
        warn(f"DPD {dpd:.4f} ≤ {DEMO_DPD_THRESHOLD} — bias may be too subtle for a live demo")

    # Counterfactual examples
    cf = data.get("counterfactual_examples", [])
    if cf:
        ok(f"{len(cf)} counterfactual example(s) returned")
    else:
        warn("No counterfactual examples returned")

    # Intersectional
    ib = data.get("intersectional_bias", [])
    if ib:
        ok(f"{len(ib)} intersectional groups returned")
    else:
        warn("No intersectional bias data returned")

    return data, elapsed, timing_ok


# ---------------------------------------------------------------------------
# Step 2 — /api/simulate
# ---------------------------------------------------------------------------
def test_simulate():
    header("Step 2 · POST /api/simulate")

    with open(SAMPLE_CSV, "rb") as f:
        resp, elapsed, timing_ok = timed_request(
            requests.post,
            f"{API_BASE}/api/simulate",
            SIMULATE_TIMEOUT,
            files={"file": ("adult_sample.csv", f, "text/csv")},
            data={
                "target_col": TARGET_COL,
                "sensitive_col": SENSITIVE_COL,
            },
        )

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    ok(f"Status 200  ({elapsed:.2f}s)")

    if timing_ok:
        ok(f"Response time {elapsed:.2f}s < {SIMULATE_TIMEOUT}s")
    else:
        fail(f"Response time {elapsed:.2f}s ≥ {SIMULATE_TIMEOUT}s")

    data = resp.json()
    strategies = data["strategies"]

    print(f"\n  {'Strategy':<30} {'Fairness Gain':>15} {'Accuracy Drop':>15} {'Final Score':>12}")
    print(f"  {'─' * 74}")
    for s in strategies:
        gain_str = f"+{s['fairness_gain']:.3f}"
        drop_str = f"-{s['accuracy_drop']:.3f}"
        print(
            f"  {s['strategy_name']:<30} "
            f"{gain_str:>15} "
            f"{drop_str:>15} "
            f"{s['fairness_score']:>12.3f}"
        )

    ok(f"{len(strategies)} strategies returned")
    return data, elapsed, timing_ok


# ---------------------------------------------------------------------------
# Step 3 — /api/recommend
# ---------------------------------------------------------------------------
def test_recommend(detect_data):
    header("Step 3 · POST /api/recommend")

    payload = {
        "bias_score": detect_data["fairness_metrics"]["demographic_parity_difference"],
        "accuracy": detect_data["accuracy"],
    }

    resp, elapsed, timing_ok = timed_request(
        requests.post,
        f"{API_BASE}/api/recommend",
        RECOMMEND_TIMEOUT,
        json=payload,
    )

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    ok(f"Status 200  ({elapsed:.2f}s)")

    if timing_ok:
        ok(f"Response time {elapsed:.2f}s < {RECOMMEND_TIMEOUT}s")
    else:
        fail(f"Response time {elapsed:.2f}s ≥ {RECOMMEND_TIMEOUT}s")

    data = resp.json()

    print(f"\n  {'Recommended Strategy':<25} {data['recommended_strategy']}")
    print(f"  {'Expected Fairness Gain':<25} +{data['expected_fairness_gain']:.3f}")
    print(f"  {'Expected Accuracy Drop':<25} -{data['expected_accuracy_drop']:.3f}")
    print(f"  {'Reason':<25} {data['reason'][:80]}...")

    return data, elapsed, timing_ok


# ---------------------------------------------------------------------------
# Step 4 — Fairness improvement assertion
# ---------------------------------------------------------------------------
def test_improvement(recommend_data):
    header("Step 4 · Fairness Improvement Check")

    gain = recommend_data["expected_fairness_gain"]

    if gain >= MIN_FAIRNESS_IMPROVEMENT:
        ok(f"Fairness gain {gain:.3f} ≥ {MIN_FAIRNESS_IMPROVEMENT} — recommendation is meaningful")
        return True
    else:
        warn(
            f"Fairness gain {gain:.3f} < {MIN_FAIRNESS_IMPROVEMENT} — "
            "Simulation strategies may need tuning — check reweighting sample weights"
        )
        return False


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
def summary(results):
    header("Summary")

    all_pass = True
    for name, elapsed, timing_ok, extra_pass in results:
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if (timing_ok and extra_pass) else f"{Colors.RED}FAIL{Colors.RESET}"
        if not (timing_ok and extra_pass):
            all_pass = False
        print(f"  {name:<30} {elapsed:>6.2f}s   {status}")

    total = sum(r[1] for r in results)
    print(f"\n  {'Total pipeline time':<30} {total:>6.2f}s")

    if all_pass:
        print(f"\n  {Colors.BOLD}{Colors.GREEN}▶ ALL CHECKS PASSED{Colors.RESET}\n")
    else:
        print(f"\n  {Colors.BOLD}{Colors.RED}▶ SOME CHECKS FAILED — review output above{Colors.RESET}\n")

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{Colors.BOLD}EquiLens AI — Smoke Test{Colors.RESET}")
    print(f"{Colors.DIM}Backend: {API_BASE}  |  Dataset: adult_sample.csv{Colors.RESET}")

    preflight()

    detect_data,    t1, p1 = test_detect()
    simulate_data,  t2, p2 = test_simulate()
    recommend_data, t3, p3 = test_recommend(detect_data)
    improvement_ok         = test_improvement(recommend_data)

    results = [
        ("/api/detect",    t1, p1, True),
        ("/api/simulate",  t2, p2, True),
        ("/api/recommend", t3, p3, True),
        ("fairness check", 0,  improvement_ok, improvement_ok),
    ]

    passed = summary(results)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
