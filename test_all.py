"""
================================================================================
FILE: test_all.py
PURPOSE:
    Automated test suite compatible with both `pytest` (`pytest test_all.py`)
    and standard Python (`python test_all.py`).

    Tests:
      1. Dynamic crop & pest counts in `config.py` (`NUM_CROPS = len(CROPS)`
         and `NUM_PESTS = len(PEST_SHORTLIST)`), `DEMO_MODE == False`, and
         environment variable configuration.
      2. Stage 1 (`# [ARRAY]`): 8-element normalized feature array in [0.0, 1.0].
      3. Stage 2 (`# [HASH_TABLE]`): 31-bucket Polynomial Rolling Hash Table with
         Separate Chaining (no `dict` used) across all `NUM_PESTS` pests.
      4. Stage 3 (`# [DECISION_TREE]`): `<15%` Low, `15-35%` Moderate, `>35%` Severe.
      5. Stage 4 (`# [PRIORITY_QUEUE]`): Manual Binary Max-Heap ranking treatments
         by efficacy, PHI safety, and INR cost.
      6. `prepare_data.py` (`# [DATASET]`): Space/hyphen/bracket normalization
         (`"armyworm"` == `"army worm"`) and stopping on zero or multiple matches.
      7. `database.py` & `app.py` (`# [DATABASE]`, `# [LOGIN]`, `# [FEEDBACK]`,
         `# [ADMIN]`): Real evaluation accuracy (`"Not evaluated yet"` until
         `evaluate.py` saves metrics), measured DSA pipeline speed (`"No scans yet"`
         until a scan is saved), and user complaint + admin response workflow.
================================================================================
"""

# Import standard library modules for temporary files and test assertions
import os
import tempfile
import unittest

# Import all project modules under test
import app
import config
import database
import dsa
from pest_data import PEST_DATABASE
import prepare_data


# ==============================================================================
# 1. CONFIG & DYNAMIC SCOPE TESTS (# [CONFIG] # [DATASET])
# ==============================================================================

# # [CONFIG] # [DATASET]
def test_config_counts_and_demo_mode() -> None:
    """
    Verifies that `NUM_CROPS` and `NUM_PESTS` are dynamically computed with `len()`,
    that every pest in `PEST_SHORTLIST` matches `PEST_DATABASE`, and that
    `DEMO_MODE` defaults to `False`.
    """
    assert config.NUM_CROPS == len(config.CROPS)
    assert config.NUM_PESTS == len(config.PEST_SHORTLIST)
    assert len(PEST_DATABASE) == config.NUM_PESTS
    assert config.DEMO_MODE is False

    for idx, item in enumerate(config.PEST_SHORTLIST):
        assert item["class_id"] == idx
        assert item["crop"] in config.CROPS
        assert PEST_DATABASE[idx]["class_id"] == idx
        assert PEST_DATABASE[idx]["crop"] == item["crop"]


# ==============================================================================
# 2. FOUR DATA STRUCTURE STAGES TESTS (# [ARRAY] # [HASH_TABLE] # [DECISION_TREE] # [PRIORITY_QUEUE])
# ==============================================================================

# # [ARRAY]
def test_stage1_feature_array() -> None:
    """
    Verifies that `dsa.extract_feature_array` always returns a fixed-size
    8-element array of floats in `[0.0, 1.0]`.
    """
    # Create a synthetic 16x16 BGR leaf image (nested list or numpy array)
    synthetic_bgr = [
        [[40, 160, 50] if (r + c) % 2 == 0 else [30, 90, 120] for c in range(16)]
        for r in range(16)
    ]
    features = dsa.extract_feature_array(synthetic_bgr)
    assert len(features) == config.FEATURE_ARRAY_SIZE
    for val in features:
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0


# # [HASH_TABLE]
def test_stage2_hash_table_all_pests_and_chaining() -> None:
    """
    Verifies that `PestHashTable` uses a 31-bucket list (not a dict), supports
    collision chaining, and retrieves every pest in `PEST_DATABASE`.
    """
    ht = dsa.PestHashTable(num_buckets=config.HASH_TABLE_BUCKETS)
    assert isinstance(ht.buckets, list)
    assert len(ht.buckets) == config.HASH_TABLE_BUCKETS

    for rec in PEST_DATABASE:
        key = dsa.PestHashTable.make_key(rec["crop"], rec["common_name"])
        bucket_idx = ht.insert(key, rec)
        assert 0 <= bucket_idx < config.HASH_TABLE_BUCKETS

    assert ht.size == config.NUM_PESTS

    for rec in PEST_DATABASE:
        key = dsa.PestHashTable.make_key(rec["crop"], rec["common_name"])
        found, b_idx, steps = ht.lookup(key)
        assert found is not None
        assert found["class_id"] == rec["class_id"]
        assert 0 <= b_idx < config.HASH_TABLE_BUCKETS
        assert steps >= 1


# # [DECISION_TREE] # [SEVERITY]
def test_stage3_severity_decision_tree() -> None:
    """
    Verifies exact Decision Tree severity boundaries:
      - `< 15.0` -> `"Low"`
      - `15.0 .. 35.0` -> `"Moderate"`
      - `> 35.0` -> `"Severe"`
    """
    tree = dsa.SeverityDecisionTree()
    assert tree.classify(0.0)["severity"] == "Low"
    assert tree.classify(14.9)["severity"] == "Low"
    assert tree.classify(15.0)["severity"] == "Moderate"
    assert tree.classify(35.0)["severity"] == "Moderate"
    assert tree.classify(35.1)["severity"] == "Severe"
    assert tree.classify(85.0)["severity"] == "Severe"


# # [PRIORITY_QUEUE] # [TREATMENT]
def test_stage4_manual_max_heap_priority_queue() -> None:
    """
    Verifies that `ManualMaxHeapPriorityQueue` pops treatments in descending
    `priority_score` order.
    """
    sample_treatments = PEST_DATABASE[0]["treatments"]
    for sev in ("Low", "Moderate", "Severe"):
        ranked = dsa.rank_treatments(sample_treatments, sev)
        assert len(ranked) == len(sample_treatments)
        scores = [tx["priority_score"] for tx in ranked]
        assert scores == sorted(scores, reverse=True)
        assert [tx["rank"] for tx in ranked] == list(
            range(1, len(sample_treatments) + 1)
        )


# ==============================================================================
# 3. DATASET NAME NORMALIZATION & AMBIGUITY TESTS (# [DATASET])
# ==============================================================================

# # [DATASET]
def test_prepare_data_normalization_and_matching() -> None:
    """
    Verifies that:
      1. `normalize_pest_name` removes spaces, hyphens, and brackets and
         lowercases both sides so `"armyworm"` matches `"army worm"`.
      2. `match_shortlist_to_ip102` stops (`ValueError`) when a name matches
         zero classes or more than one class (such as `"aphids"` matching two
         aphid species).
    """
    assert prepare_data.normalize_pest_name("armyworm") == prepare_data.normalize_pest_name(
        "army worm"
    )
    assert prepare_data.normalize_pest_name(
        "panonychus citri (citrus red mite)"
    ) == "panonychuscitricitrusredmite"

    mock_ip102_classes = [
        {"ip102_id": 0, "raw_name": "army worm", "norm_name": "armyworm"},
        {"ip102_id": 1, "raw_name": "corn aphids", "norm_name": "cornaphids"},
        {"ip102_id": 2, "raw_name": "wheat aphids", "norm_name": "wheataphids"},
    ]

    # Exact match between "armyworm" and "army worm"
    matched = prepare_data.match_shortlist_to_ip102(
        [{"class_id": 0, "crop": "Corn", "name": "armyworm", "match_keyword": "armyworm"}],
        mock_ip102_classes,
    )
    assert len(matched) == 1
    assert matched[0]["ip102_raw_name"] == "army worm"

    # Ambiguous match ("aphids" matches both "corn aphids" and "wheat aphids") must STOP
    ambiguous_raised = False
    try:
        prepare_data.match_shortlist_to_ip102(
            [{"class_id": 1, "crop": "Corn", "name": "aphids", "match_keyword": "aphids"}],
            mock_ip102_classes,
        )
    except ValueError:
        ambiguous_raised = True
    assert ambiguous_raised is True

    # Zero match must STOP
    zero_raised = False
    try:
        prepare_data.match_shortlist_to_ip102(
            [{"class_id": 2, "crop": "Rice", "name": "unknown beetle", "match_keyword": "unknown beetle"}],
            mock_ip102_classes,
        )
    except ValueError:
        zero_raised = True
    assert zero_raised is True


# ==============================================================================
# 4. SQLITE DATABASE, LIVE STATS & ADMIN COMPLAINT REPLY TESTS (# [DATABASE])
# ==============================================================================

# # [DATABASE] # [LOGIN] # [FEEDBACK] # [ADMIN]
def test_database_live_stats_and_feedback_workflow() -> None:
    """
    Verifies:
      - Before evaluation or scans, `app.get_live_dashboard_stats` returns
        `"Not evaluated yet"` and `"No scans yet"`.
      - After `database.save_evaluation_metrics` and `database.save_scan`,
        `app.get_live_dashboard_stats` returns the real stored numbers.
      - User registration, login, complaint submission, and admin reply work
        end-to-end.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_db = os.path.join(tmpdir, "test_pest.db")
        database.init_db(db_path=temp_db)

        # 1. Verify initial live stats have NO fake numbers
        initial_stats = app.get_live_dashboard_stats(db_path=temp_db)
        assert initial_stats["crops_text"] == f"{config.NUM_CROPS} crops"
        assert initial_stats["pests_text"] == f"{config.NUM_PESTS} pests"
        assert initial_stats["accuracy_text"] == "Not evaluated yet"
        assert initial_stats["dsa_speed_text"] == "No scans yet"

        # 2. Register a user and an admin
        ok_u, _, user = database.register_user(
            "Test Farmer", "farmer_test@example.com", "Pass1234", db_path=temp_db
        )
        assert ok_u and user is not None and user["role"] == "user"

        ok_a, _, admin = database.register_user(
            "Test Admin",
            "admin_test@example.com",
            "Admin1234",
            role_override="admin",
            db_path=temp_db,
        )
        assert ok_a and admin is not None and admin["role"] == "admin"

        # 3. Save a scan with measured DSA speed and check updated stats
        database.save_scan(
            user_id=user["id"],
            user_name=user["name"],
            user_email=user["email"],
            crop=config.CROPS[0],
            pest_name=config.PEST_SHORTLIST[0]["name"],
            scientific_name=PEST_DATABASE[0]["scientific_name"],
            cnn_confidence=0.912,
            yolo_confidence=0.845,
            affected_area_pct=18.5,
            severity="Moderate",
            top_treatment=PEST_DATABASE[0]["treatments"][0]["name"],
            dsa_speed_ms=1.42,
            db_path=temp_db,
        )

        # 4. Save real evaluation metrics from evaluate.py and verify live stats
        database.save_evaluation_metrics(
            cnn_test_accuracy=89.35,
            cnn_macro_f1=87.10,
            yolo_map50=82.40,
            db_path=temp_db,
        )

        updated_stats = app.get_live_dashboard_stats(db_path=temp_db)
        assert updated_stats["accuracy_text"] == "89.35%"
        assert updated_stats["dsa_speed_text"] == "1.42 ms"

        # 5. Submit user complaint and send Admin reply
        ok_fb, _, fb_id = database.submit_feedback(
            user_id=user["id"],
            user_name=user["name"],
            user_email=user["email"],
            subject="Dosage question",
            message="Can I spray in the evening?",
            crop=config.CROPS[0],
            pest_name=config.PEST_SHORTLIST[0]["name"],
            db_path=temp_db,
        )
        assert ok_fb and fb_id is not None

        ok_rep, _ = database.reply_to_feedback(
            feedback_id=fb_id,
            admin_reply="Yes, spray after 4 PM for best results.",
            admin_name=admin["name"],
            new_status="Resolved",
            db_path=temp_db,
        )
        assert ok_rep is True

        user_complaints = database.get_feedback_list(
            user_id=user["id"], db_path=temp_db
        )
        assert len(user_complaints) == 1
        assert user_complaints[0]["user_name"] == "Test Farmer"
        assert user_complaints[0]["status"] == "Resolved"
        assert (
            user_complaints[0]["admin_reply"]
            == "Yes, spray after 4 PM for best results."
        )


# Allow running directly via `python test_all.py` as well as `pytest`
class AllProjectTests(unittest.TestCase):
    def test_all(self) -> None:
        test_config_counts_and_demo_mode()
        test_stage1_feature_array()
        test_stage2_hash_table_all_pests_and_chaining()
        test_stage3_severity_decision_tree()
        test_stage4_manual_max_heap_priority_queue()
        test_prepare_data_normalization_and_matching()
        test_database_live_stats_and_feedback_workflow()


if __name__ == "__main__":
    unittest.main(verbosity=2)
