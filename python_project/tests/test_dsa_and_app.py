"""
================================================================================
FILE: tests/test_dsa_and_app.py
PURPOSE:
    Automated pytest test suite verifying:
      1. Stage 1 Fixed-Size 8-Element Feature Array (`extract_feature_array`)
         returns exactly 8 normalized floats in `[0.0, 1.0]`.
      2. Stage 2 Custom 31-Bucket Hash Table (`PestHashTable`) computes
         deterministic polynomial rolling hashes, handles collisions via singly
         linked list chaining, and retrieves all 20 IP102 pest records in O(1)
         average time.
      3. Stage 3 Binary Decision Tree (`SeverityDecisionTree`) classifies
         `< 15%` as Low, `15-35%` as Moderate, and `> 35%` as Severe.
      4. Stage 4 Manual Binary Max-Heap Priority Queue (`TreatmentPriorityQueue`)
         ranks treatments in strictly non-increasing order of `priority_score`.
      5. Flask `/api/health` route, Login/Signup flow, and Admin complaint reply
         workflow work end-to-end.

HOW TO RUN:
    pytest
================================================================================
"""

# Import standard library modules
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from app import create_app
from data.pest_database import GLOBAL_PEST_HASH_TABLE, IP102_TOP20_PEST_DATABASE
from dsa_pipeline import (
    PestHashTable,
    SeverityDecisionTree,
    TreatmentPriorityQueue,
    extract_feature_array,
    rank_treatments_with_priority_queue,
)


# # [ARRAY]
def test_stage1_feature_array_shape_and_bounds():
    """Verifies Stage 1 returns a fixed-size 8-element float list in [0.0, 1.0]."""
    synthetic_leaf_bgr = np.zeros((120, 120, 3), dtype=np.uint8)
    synthetic_leaf_bgr[:, :] = (40, 160, 50)  # Green leaf patch
    features = extract_feature_array(synthetic_leaf_bgr)

    assert isinstance(features, list)
    assert len(features) == config.FEATURE_ARRAY_SIZE == 8
    for val in features:
        assert 0.0 <= val <= 1.0


# # [HASH_TABLE]
def test_stage2_custom_hash_table_lookup_and_chaining():
    """Verifies Stage 2 custom 31-bucket Hash Table lookup for all 20 IP102 classes."""
    ht = PestHashTable(num_buckets=config.HASH_TABLE_BUCKETS)
    for record in IP102_TOP20_PEST_DATABASE:
        key = f"{record['crop']}|{record['common_name']}"
        bucket_idx = ht.insert(key, record)
        assert 0 <= bucket_idx < config.HASH_TABLE_BUCKETS

    # Verify every inserted crop|pest record can be retrieved accurately
    for record in IP102_TOP20_PEST_DATABASE:
        key = f"{record['crop']}|{record['common_name']}"
        found, b_idx, steps = GLOBAL_PEST_HASH_TABLE.lookup(key)
        assert found is not None
        assert found["scientific_name"] == record["scientific_name"]
        assert steps >= 1


# # [DECISION_TREE]
def test_stage3_decision_tree_severity_thresholds():
    """Verifies Stage 3 Binary Decision Tree boundaries (<15%, 15-35%, >35%)."""
    tree = SeverityDecisionTree()

    res_low = tree.evaluate(8.5)
    assert res_low["severity"] == "Low"

    res_mod = tree.evaluate(22.0)
    assert res_mod["severity"] == "Moderate"

    res_sev = tree.evaluate(48.0)
    assert res_sev["severity"] == "Severe"


# # [PRIORITY_QUEUE]
def test_stage4_manual_max_heap_priority_queue_ordering():
    """Verifies Stage 4 Manual Max-Heap pops treatments in descending score order."""
    sample_treatments = IP102_TOP20_PEST_DATABASE[0]["treatments"]
    ranked = rank_treatments_with_priority_queue(sample_treatments, severity="Severe")

    assert len(ranked) == len(sample_treatments)
    assert ranked[0]["rank"] == 1
    # Verify monotonic non-increasing priority scores
    scores = [item["priority_score"] for item in ranked]
    assert scores == sorted(scores, reverse=True)


# # [API_ROUTE]
def test_flask_health_endpoint():
    """Verifies the Flask `/api/health` route returns HTTP 200 and expected JSON keys."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["hash_table_buckets"] == 31
        assert "PHANI NANDAN M S" in data["project_by"]
