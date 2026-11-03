"""
================================================================================
FILE: dsa.py
PURPOSE:
    Implements the 4 required Data Structure & Algorithm (DSA) stages completely
    from scratch in pure Python (NO built-in `dict` inside the Hash Table, and
    NO `heapq` module inside the Priority Queue):
      - Stage 1 (`# [ARRAY]`): Fixed-size 8-element normalized visual feature
        array extracted from a leaf image using OpenCV.
      - Stage 2 (`# [HASH_TABLE]`): Custom 31-bucket Hash Table using a
        Polynomial Rolling Hash and Singly Linked List Separate Chaining to map
        `"crop|pest"` keys to their database record in `pest_data.py`.
      - Stage 3 (`# [DECISION_TREE]`): Binary Decision Tree classifying pest
        severity from affected leaf area % (`< 15` Low, `15-35` Moderate,
        `> 35` Severe).
      - Stage 4 (`# [PRIORITY_QUEUE]`): Manual Binary Max-Heap Priority Queue
        ranking treatments by a composite score of efficacy %, PHI safety, and
        cost per acre in INR.

    Every function includes its exact Big-O Time and Space Complexity in comments.

HOW IT CONNECTS TO OTHER FILES:
    - Uses constants from `config.py` and pest records from `pest_data.py`.
    - Called by `detect.py` during leaf scanning and displayed live on the
      "DSA Guide" page in `app.py`.
    - Tested by `test_all.py`.
================================================================================
"""

# Import math for log2 entropy calculation and typing for clear signatures
import math
from typing import Any, Dict, List, Optional, Tuple

# Import OpenCV and NumPy for Stage 1 visual feature extraction
import cv2
import numpy as np

# Import configuration constants and the 20-pest database
import config
from pest_data import PEST_DATABASE


# ==============================================================================
# STAGE 1: FIXED-SIZE 8-ELEMENT FEATURE ARRAY (# [ARRAY] # [PREPROCESS])
# ==============================================================================

# # [ARRAY] # [PREPROCESS]
def extract_feature_array(image_bgr: np.ndarray) -> List[float]:
    """
    Extracts a fixed-size 8-element normalized visual feature array in [0.0, 1.0]
    from a BGR leaf image using OpenCV:
      Index [0]: Green Healthy Leaf Ratio
      Index [1]: Brown Necrotic Lesion Ratio
      Index [2]: Yellow Chlorosis Ratio
      Index [3]: Canny Edge Density
      Index [4]: Grayscale Texture Entropy
      Index [5]: Dark Spot Ratio
      Index [6]: HSV Saturation Spread
      Index [7]: Lab Colour Contrast

    Big-O Complexity:
      - Time Complexity: O(H * W) where H and W are the resized image dimensions
        (256 x 256), and O(1) for reading/writing any fixed array index `arr[i]`.
      - Space Complexity: O(1) output space because the feature array always has
        exactly `config.FEATURE_ARRAY_SIZE = 8` float slots.
    """
    # Pre-allocate our fixed-size 8-slot array filled with 0.0
    features: List[float] = [0.0] * config.FEATURE_ARRAY_SIZE

    # Guard against empty or invalid images
    if image_bgr is None or image_bgr.size == 0:
        return features

    # Resize image to fixed dimensions (256x256) for consistent pixel statistics
    resized = cv2.resize(
        image_bgr, (config.OPENCV_FEATURE_SIZE, config.OPENCV_FEATURE_SIZE)
    )
    total_pixels = float(config.OPENCV_FEATURE_SIZE * config.OPENCV_FEATURE_SIZE)

    # Convert BGR image to HSV, Grayscale, and CIE Lab color spaces
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)

    # Slot [0]: Green Healthy Leaf Ratio (Hue 35..85 in OpenCV HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    features[0] = round(float(np.count_nonzero(green_mask)) / total_pixels, 4)

    # Slot [1]: Brown Necrotic Lesion Ratio (Hue 5..22 in OpenCV HSV)
    brown_mask = cv2.inRange(hsv, np.array([5, 40, 20]), np.array([22, 255, 200]))
    features[1] = round(float(np.count_nonzero(brown_mask)) / total_pixels, 4)

    # Slot [2]: Yellow Chlorosis Ratio (Hue 23..34 in OpenCV HSV)
    yellow_mask = cv2.inRange(hsv, np.array([23, 40, 60]), np.array([34, 255, 255]))
    features[2] = round(float(np.count_nonzero(yellow_mask)) / total_pixels, 4)

    # Slot [3]: Canny Edge Density (measures chewing margins and insect outlines)
    edges = cv2.Canny(gray, 80, 180)
    features[3] = round(float(np.count_nonzero(edges)) / total_pixels, 4)

    # Slot [4]: Grayscale Texture Entropy (Shannon entropy normalized by 8 bits)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probs = hist / total_pixels
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= float(p) * math.log2(float(p))
    features[4] = round(min(1.0, max(0.0, entropy / 8.0)), 4)

    # Slot [5]: Dark Spot Ratio (Value channel < 45 indicating frass/sooty mould)
    dark_mask = hsv[:, :, 2] < 45
    features[5] = round(float(np.count_nonzero(dark_mask)) / total_pixels, 4)

    # Slot [6]: HSV Saturation Spread (std dev of Saturation normalized by 128)
    sat_std = float(np.std(hsv[:, :, 1])) / 128.0
    features[6] = round(min(1.0, max(0.0, sat_std)), 4)

    # Slot [7]: Lab Colour Contrast (A-B chrominance spread normalized by 64)
    lab_contrast = float(np.std(lab[:, :, 1]) + np.std(lab[:, :, 2])) / 64.0
    features[7] = round(min(1.0, max(0.0, lab_contrast)), 4)

    return features


# ==============================================================================
# STAGE 2: 31-BUCKET HASH TABLE WITH SEPARATE CHAINING (# [HASH_TABLE])
# ==============================================================================

# # [HASH_TABLE]
class HashNode:
    """
    Singly linked list node used for Separate Chaining inside a Hash Table bucket.
    Stores `key` ("crop|pest"), `value` (the pest record), and `next` pointer.

    Big-O Complexity:
      - Time Complexity: O(1) node creation.
      - Space Complexity: O(1) per node.
    """

    def __init__(self, key: str, value: Any) -> None:
        #Normalized lookup key string ("crop|pest")
        self.key: str = key
        # Associated pest record dictionary
        self.value: Any = value
        # Pointer to the next node in the same bucket chain (or None)
        self.next: Optional["HashNode"] = None


# # [HASH_TABLE]
class PestHashTable:
    """
    Custom Hash Table implemented from scratch WITHOUT using a Python `dict`
    for storage. Uses a fixed-size Python list of 31 bucket heads, a Polynomial
    Rolling Hash function (base 31), and Separate Chaining for collisions.
    """

    def __init__(self, num_buckets: int = config.HASH_TABLE_BUCKETS) -> None:
        """
        Initializes a contiguous list of `num_buckets` (31) empty bucket pointers.

        Big-O Complexity:
          - Time Complexity: O(B) where B = 31 buckets.
          - Space Complexity: O(B) to allocate the bucket array.
        """
        self.num_buckets: int = num_buckets
        # Fixed list of 31 bucket heads (no built-in dict used!)
        self.buckets: List[Optional[HashNode]] = [None] * num_buckets
        self.size: int = 0

    @staticmethod
    def make_key(crop: str, pest_name: str) -> str:
        """
        Builds a lowercase normalized `"crop|pest"` key string.

        Big-O Complexity:
          - Time Complexity: O(L) where L is the string length.
          - Space Complexity: O(L).
        """
        return f"{crop.strip().lower()}|{pest_name.strip().lower()}"

    def compute_hash(self, key: str) -> int:
        """
        Computes the Polynomial Rolling Hash of `key`:
            H(key) = (Sum_{i=0..L-1} ord(key[i]) * 31^i) mod 31

        Big-O Complexity:
          - Time Complexity: O(L) where L is the number of characters in `key`.
          - Space Complexity: O(1) auxiliary variables.
        """
        hash_sum = 0
        power = 1
        for ch in key.strip().lower():
            hash_sum = (hash_sum + ord(ch) * power) % self.num_buckets
            power = (power * config.HASH_POLYNOMIAL_BASE) % self.num_buckets
        return hash_sum

    def insert(self, key: str, value: Any) -> int:
        """
        Inserts or updates `(key, value)` inside the hash table bucket chain.
        Returns the bucket index (0..30).

        Big-O Complexity:
          - Time Complexity: O(L) average case; O(L + K) where K is the chain
            length in the target bucket.
          - Space Complexity: O(1) for one new `HashNode`.
        """
        norm_key = key.strip().lower()
        bucket_idx = self.compute_hash(norm_key)
        head = self.buckets[bucket_idx]

        # Traverse the chain to check if key already exists
        curr = head
        while curr is not None:
            if curr.key == norm_key:
                curr.value = value
                return bucket_idx
            curr = curr.next

        # Prepend a new HashNode at the head of this bucket's linked list
        new_node = HashNode(norm_key, value)
        new_node.next = head
        self.buckets[bucket_idx] = new_node
        self.size += 1
        return bucket_idx

    def lookup(self, key: str) -> Tuple[Optional[Any], int, int]:
        """
        Looks up `key` ("crop|pest") in the 31-bucket hash table.
        Returns `(record_or_None, bucket_index, chain_steps_traversed)`.

        Big-O Complexity:
          - Time Complexity: O(L) average case (O(1) bucket jump + 1-2 chain
            steps); O(L + N) worst case if all N items collide in one bucket.
          - Space Complexity: O(1).
        """
        norm_key = key.strip().lower()
        bucket_idx = self.compute_hash(norm_key)
        curr = self.buckets[bucket_idx]
        steps = 0

        while curr is not None:
            steps += 1
            if curr.key == norm_key:
                return curr.value, bucket_idx, steps
            curr = curr.next

        return None, bucket_idx, steps

    def get_bucket_summary(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all 31 buckets and their linked-list keys for the
        DSA Guide page.

        Big-O Complexity:
          - Time Complexity: O(B + N) where B = 31 buckets and N = 20 pests.
          - Space Complexity: O(B + N).
        """
        summary: List[Dict[str, Any]] = []
        for idx in range(self.num_buckets):
            keys_in_chain: List[str] = []
            curr = self.buckets[idx]
            while curr is not None:
                keys_in_chain.append(curr.key)
                curr = curr.next
            summary.append(
                {
                    "bucket": idx,
                    "chain_length": len(keys_in_chain),
                    "keys": " -> ".join(keys_in_chain) if keys_in_chain else "Empty",
                }
            )
        return summary


# Build and populate the global 31-bucket Hash Table with all 20 pests at import time
GLOBAL_HASH_TABLE = PestHashTable(num_buckets=config.HASH_TABLE_BUCKETS)
for _pest_rec in PEST_DATABASE:
    _key = PestHashTable.make_key(_pest_rec["crop"], _pest_rec["common_name"])
    GLOBAL_HASH_TABLE.insert(_key, _pest_rec)


# ==============================================================================
# STAGE 3: DECISION TREE FOR SEVERITY CLASSIFICATION (# [DECISION_TREE] # [SEVERITY])
# ==============================================================================

# # [DECISION_TREE] # [SEVERITY]
class DecisionNode:
    """
    Node in our binary Decision Tree for severity classification.
    Internal nodes test `affected_area < threshold`; leaf nodes hold the
    final severity label (`Low`, `Moderate`, or `Severe`) and strategy text.

    Big-O Complexity:
      - Time Complexity: O(1) node creation.
      - Space Complexity: O(1).
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        left: Optional["DecisionNode"] = None,
        right: Optional["DecisionNode"] = None,
        severity: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> None:
        self.threshold = threshold
        self.left = left
        self.right = right
        self.severity = severity
        self.strategy = strategy

    def is_leaf(self) -> bool:
        """Returns True if this node is a leaf node. Big-O: O(1) time, O(1) space."""
        return self.severity is not None


# # [DECISION_TREE] # [SEVERITY] # [THRESHOLD]
class SeverityDecisionTree:
    """
    Binary Decision Tree classifying leaf damage severity from `affected_area_pct`:
      - Under 15%   -> "Low"
      - 15% to 35%  -> "Moderate"
      - Over 35%    -> "Severe"
    """

    def __init__(self) -> None:
        """
        Constructs the 2-level binary decision tree using `config.SEVERITY_LOW_MAX`
        (15.0%) and `config.SEVERITY_MODERATE_MAX` (35.0%).

        Big-O Complexity:
          - Time Complexity: O(1) to build 5 tree nodes.
          - Space Complexity: O(1).
        """
        leaf_low = DecisionNode(
            severity="Low",
            strategy="Low Severity (<15% area): Prioritize Organic/Botanical sprays and field traps.",
        )
        leaf_moderate = DecisionNode(
            severity="Moderate",
            strategy="Moderate Severity (15-35% area): Prioritize Biological biocontrol or selective IPM sprays.",
        )
        leaf_severe = DecisionNode(
            severity="Severe",
            strategy="Severe Infestation (>35% area): Immediate high-efficacy control required to save the crop.",
        )

        # Level 1 right-subtree node checking <= 35% vs > 35%
        node_moderate_check = DecisionNode(
            threshold=config.SEVERITY_MODERATE_MAX,
            left=leaf_moderate,
            right=leaf_severe,
        )

        # Root node checking < 15% vs >= 15%
        self.root = DecisionNode(
            threshold=config.SEVERITY_LOW_MAX,
            left=leaf_low,
            right=node_moderate_check,
        )

    def classify(self, affected_area_pct: float) -> Dict[str, Any]:
        """
        Traverses the decision tree from root to leaf for `affected_area_pct`.

        Big-O Complexity:
          - Time Complexity: O(H) where H = 2 is the tree height, which is O(1).
          - Space Complexity: O(H) = O(1) to store the path list.
        """
        area = max(0.0, min(100.0, float(affected_area_pct)))
        path: List[str] = []

        # Root check: under 15% is Low
        if area < float(self.root.threshold):
            path.append(f"Root: {area:.1f}% < {self.root.threshold}% -> Left (Low)")
            leaf = self.root.left
        else:
            path.append(f"Root: {area:.1f}% >= {self.root.threshold}% -> Right")
            right_node = self.root.right
            # Second check: 15% to 35% is Moderate; over 35% is Severe
            if area <= float(right_node.threshold):
                path.append(
                    f"Node 2: {area:.1f}% <= {right_node.threshold}% -> Left (Moderate)"
                )
                leaf = right_node.left
            else:
                path.append(
                    f"Node 2: {area:.1f}% > {right_node.threshold}% -> Right (Severe)"
                )
                leaf = right_node.right

        return {
            "severity": leaf.severity,
            "strategy": leaf.strategy,
            "path": " | ".join(path),
        }


# ==============================================================================
# STAGE 4: MANUAL MAX-HEAP PRIORITY QUEUE (# [PRIORITY_QUEUE] # [TREATMENT])
# ==============================================================================

# # [PRIORITY_QUEUE] # [TREATMENT]
def compute_treatment_score(treatment: Dict[str, Any], severity: str) -> float:
    """
    Computes the composite priority score (0..100) for a treatment option based on:
      - Efficacy % (weight 0.50)
      - Pre-Harvest Interval (PHI) safety (weight 0.30, shorter PHI = higher score)
      - Cost per acre in INR (weight 0.20, lower cost = higher score)
      - Small strategy bonus (+5.0) matching the Decision Tree severity level

    Big-O Complexity:
      - Time Complexity: O(1) arithmetic operations.
      - Space Complexity: O(1).
    """
    efficacy_score = float(treatment.get("efficacy", 75.0))
    phi_days = float(treatment.get("phi_days", 7.0))
    cost_inr = float(treatment.get("cost_inr", 600.0))

    # Shorter PHI days -> higher safety score (0..100)
    phi_score = max(0.0, (1.0 - (phi_days / config.MAX_PHI_DAYS_CAP)) * 100.0)
    # Lower cost in INR -> higher cost efficiency score (0..100)
    cost_score = max(0.0, (1.0 - (cost_inr / config.MAX_COST_INR_CAP)) * 100.0)

    base_score = (
        config.WEIGHT_EFFICACY * efficacy_score
        + config.WEIGHT_PHI_SAFETY * phi_score
        + config.WEIGHT_COST * cost_score
    )

    # Align with Stage 3 Decision Tree severity strategy
    tx_type = str(treatment.get("type", "")).lower()
    bonus = 0.0
    if severity == "Low" and tx_type == "organic":
        bonus = 5.0
    elif severity == "Moderate" and tx_type == "biological":
        bonus = 5.0
    elif severity == "Severe" and tx_type == "chemical":
        bonus = 5.0

    return round(base_score + bonus, 2)


# # [PRIORITY_QUEUE] # [TREATMENT]
class ManualMaxHeapPriorityQueue:
    """
    Manual Binary Max-Heap Priority Queue implemented from scratch WITHOUT
    Python's `heapq` library. Elements with the highest `priority_score` bubble
    to index 0 (`heap[0]`).
    """

    def __init__(self) -> None:
        """
        Initializes an empty array-backed binary max-heap.

        Big-O Complexity:
          - Time Complexity: O(1).
          - Space Complexity: O(1).
        """
        self.heap: List[Dict[str, Any]] = []

    def push(self, item: Dict[str, Any]) -> None:
        """
        Appends `item` to the end of the heap array and restores the max-heap
        property via `_sift_up`.

        Big-O Complexity:
          - Time Complexity: O(log T) where T is the number of treatments in the heap.
          - Space Complexity: O(1) auxiliary space.
        """
        self.heap.append(item)
        self._sift_up(len(self.heap) - 1)

    def pop_max(self) -> Optional[Dict[str, Any]]:
        """
        Removes and returns the highest-scoring treatment at `self.heap[0]` and
        restores the max-heap property via `_sift_down`.

        Big-O Complexity:
          - Time Complexity: O(log T) where T is the number of treatments in the heap.
          - Space Complexity: O(1) auxiliary space.
        """
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()

        top_item = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return top_item

    def _sift_up(self, idx: int) -> None:
        """
        Bubbles the element at `idx` upward while its `priority_score` is greater
        than its parent's `priority_score`.

        Big-O Complexity:
          - Time Complexity: O(log T).
          - Space Complexity: O(1).
        """
        while idx > 0:
            parent = (idx - 1) // 2
            if self.heap[idx]["priority_score"] > self.heap[parent]["priority_score"]:
                self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
                idx = parent
            else:
                break

    def _sift_down(self, idx: int) -> None:
        """
        Pushes the element at `idx` downward while either child has a higher
        `priority_score`.

        Big-O Complexity:
          - Time Complexity: O(log T).
          - Space Complexity: O(1).
        """
        n = len(self.heap)
        while True:
            largest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2

            if (
                left < n
                and self.heap[left]["priority_score"]
                > self.heap[largest]["priority_score"]
            ):
                largest = left
            if (
                right < n
                and self.heap[right]["priority_score"]
                > self.heap[largest]["priority_score"]
            ):
                largest = right

            if largest != idx:
                self.heap[idx], self.heap[largest] = (
                    self.heap[largest],
                    self.heap[idx],
                )
                idx = largest
            else:
                break


# # [PRIORITY_QUEUE] # [TREATMENT]
def rank_treatments(
    treatments: List[Dict[str, Any]], severity: str
) -> List[Dict[str, Any]]:
    """
    Scores all candidate treatments for a pest, pushes them into our manual
    `ManualMaxHeapPriorityQueue`, and pops them in descending order (Rank #1 first).

    Big-O Complexity:
      - Time Complexity: O(T log T) where T is the number of candidate treatments.
      - Space Complexity: O(T) to hold the heap items and ranked output list.
    """
    pq = ManualMaxHeapPriorityQueue()
    for tx in treatments:
        scored = dict(tx)
        scored["priority_score"] = compute_treatment_score(scored, severity)
        pq.push(scored)

    ranked: List[Dict[str, Any]] = []
    rank_num = 1
    while len(pq.heap) > 0:
        best = pq.pop_max()
        if best is not None:
            best["rank"] = rank_num
            ranked.append(best)
            rank_num += 1
    return ranked


# # [ARRAY] # [HASH_TABLE] # [DECISION_TREE] # [PRIORITY_QUEUE]
def run_dsa_pipeline(
    image_bgr: np.ndarray,
    crop: str,
    pest_name: str,
    affected_area_pct: float,
) -> Dict[str, Any]:
    """
    Executes all 4 DSA stages in sequence and returns a single dictionary with
    every intermediate value for the Scanner page and the DSA Guide page.

    Big-O Complexity:
      - Time Complexity: O(H * W + L + T log T).
      - Space Complexity: O(B + N + T).
    """
    # Stage 1: Fixed-size 8-element OpenCV visual feature array
    feature_array = extract_feature_array(image_bgr)

    # Stage 2: 31-bucket Polynomial Rolling Hash Table lookup
    hash_key = PestHashTable.make_key(crop, pest_name)
    pest_record, bucket_idx, chain_steps = GLOBAL_HASH_TABLE.lookup(hash_key)

    # Stage 3: Binary Decision Tree severity classification
    tree = SeverityDecisionTree()
    tree_res = tree.classify(affected_area_pct)

    # Stage 4: Manual Max-Heap Priority Queue treatment ranking
    treatments = pest_record["treatments"] if pest_record else []
    ranked_treatments = rank_treatments(treatments, tree_res["severity"])

    return {
        "feature_array": feature_array,
        "hash_key": hash_key,
        "hash_bucket": bucket_idx,
        "hash_steps": chain_steps,
        "pest_record": pest_record,
        "severity": tree_res["severity"],
        "severity_strategy": tree_res["strategy"],
        "decision_path": tree_res["path"],
        "ranked_treatments": ranked_treatments,
    }
