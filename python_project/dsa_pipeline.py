"""
================================================================================
FILE: dsa_pipeline.py
PURPOSE:
    Implements the 4 Core Data Structures & Algorithms (DSA) from scratch in ONE
    self-contained Python module for our 5th-Semester B.Tech Design Project:
      - Stage 1 (# [ARRAY]): Fixed-size 1D Feature Array of 8 normalized visual
        metrics computed from a leaf image using OpenCV and NumPy.
      - Stage 2 (# [HASH_TABLE]): Custom Hash Table with 31 buckets, Polynomial
        Rolling Hash function, and Singly Linked List Separate Chaining (built
        from scratch without using Python's built-in dict for the table).
      - Stage 3 (# [DECISION_TREE]): Explicit node-based Decision Tree Classifier
        that evaluates the affected leaf area % (<15% Low, 15-35% Moderate,
        >35% Severe) and selects the matching IPM treatment strategy.
      - Stage 4 (# [PRIORITY_QUEUE]): Manual Binary Max-Heap Priority Queue
        (built from scratch without using Python's heapq module) that ranks
        candidate treatments by a composite score of Efficacy %, Pre-Harvest
        Interval (PHI) safety, and Cost per Acre in INR.

CONCEPTS USED:
    - 1D Fixed-Size Array & Image Signal Processing (# [ARRAY], # [PREPROCESS])
    - Polynomial Rolling Hash & Linked-List Chaining (# [HASH_TABLE])
    - Binary Decision Tree Traversal (# [DECISION_TREE], # [SEVERITY])
    - Binary Max-Heap Sift-Up / Sift-Down (# [PRIORITY_QUEUE], # [TREATMENT])

HOW IT CONNECTS TO OTHER FILES:
    - Uses constants from `config.py` and pest records from `data/pest_database.py`.
    - Called by `inference.py` after the real CNN and YOLOv8 models finish their
      predictions on an uploaded leaf photo.
    - Tested thoroughly by `tests/test_dsa_and_app.py`.
================================================================================
"""

# Import time to measure execution latency (in milliseconds) of each DSA stage
import time

# Import typing annotations for clear, beginner-friendly function signatures
from typing import Any, Dict, List, Optional, Tuple

# Import OpenCV for computer vision feature extraction in Stage 1
import cv2

# Import NumPy for pixel array math in Stage 1
import numpy as np

# Import all tunable DSA constants from our central config.py file
import config


# ==============================================================================
# STAGE 1: 1D FEATURE ARRAY (OPENCV VISUAL FEATURE EXTRACTION)
# ==============================================================================

# # [ARRAY] # [PREPROCESS]
def extract_feature_array(image_bgr: np.ndarray) -> List[float]:
    """
    STAGE 1 — 1D FEATURE ARRAY EXTRACTION
    Extracts a fixed-size 1D array of 8 normalized visual metrics (each in [0.0, 1.0])
    from an OpenCV BGR leaf image.

    Big-O Time Complexity:
        O(W * H) where W is image width and H is image height (linear in pixel count),
        and O(1) constant time to index any of the 8 extracted features afterwards.
    Big-O Space Complexity:
        O(1) auxiliary space for the 8-element fixed-size feature array.
    """
    # Check if the input image is valid; if empty or None, return 8 zeros safely
    if image_bgr is None or image_bgr.size == 0:
        # Allocate a fixed-size array of 8 float zeros
        return [0.0] * config.FEATURE_ARRAY_SIZE

    # Resize image to a standard 224x224 grid so feature extraction speed is consistent
    resized_bgr = cv2.resize(
        image_bgr,
        (config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    # Convert BGR image to float32 in range [0.0, 1.0] for normalized channel math
    bgr_float = resized_bgr.astype(np.float32) / 255.0

    # Split the 3 color channels: Blue (index 0), Green (index 1), Red (index 2)
    blue_channel = bgr_float[:, :, 0]
    green_channel = bgr_float[:, :, 1]
    red_channel = bgr_float[:, :, 2]

    # Convert BGR image to HSV (Hue, Saturation, Value) color space for spot/yellowing analysis
    hsv_image = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]         # Hue ranges from 0 to 179 in OpenCV
    sat_channel = hsv_image[:, :, 1] / 255.0 # Normalized Saturation in [0.0, 1.0]
    val_channel = hsv_image[:, :, 2] / 255.0 # Normalized Value (Brightness) in [0.0, 1.0]

    # Convert BGR image to Grayscale for Canny edge detection and Laplacian texture variance
    gray_image = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    # # [ARRAY] Pre-allocate a fixed-size Python list of length 8 (FEATURE_ARRAY_SIZE = 8)
    feature_array: List[float] = [0.0] * config.FEATURE_ARRAY_SIZE

    # Feature [0]: Green Chrominance Ratio = G / (R + G + B + epsilon)
    # Measures how much healthy green chlorophyll is present in the leaf
    total_rgb = red_channel + green_channel + blue_channel + 1e-6
    feature_array[0] = round(float(np.mean(green_channel / total_rgb)), 4)

    # Feature [1]: Lesion Spot Density
    # Counts pixels with brownish/reddish-brown pest lesion hues (Hue < 22 and moderate brightness)
    lesion_mask = (hue_channel < 22) & (sat_channel > 0.25) & (val_channel < 0.75)
    feature_array[1] = round(float(np.mean(lesion_mask.astype(np.float32))), 4)

    # Feature [2]: Yellow Chlorosis Ratio
    # Counts yellowing pixels caused by sap-sucking pests (Hue between 20 and 36 in OpenCV HSV)
    yellow_mask = (hue_channel >= 20) & (hue_channel <= 36) & (sat_channel > 0.20)
    feature_array[2] = round(float(np.mean(yellow_mask.astype(np.float32))), 4)

    # Feature [3]: Canny Edge Sharpness Density
    # Detects chewed leaf margins, boreholes, and lesion borders using Canny edge detector
    edges = cv2.Canny(gray_image, threshold1=60, threshold2=160)
    feature_array[3] = round(float(np.mean(edges > 0)), 4)

    # Feature [4]: HSV Saturation Mean
    # Measures overall color richness of the leaf surface
    feature_array[4] = round(float(np.mean(sat_channel)), 4)

    # Feature [5]: Necrotic Dark Ratio
    # Measures severely dead/blackened leaf tissue or dark insect bodies (Brightness < 0.22)
    necrotic_mask = val_channel < 0.22
    feature_array[5] = round(float(np.mean(necrotic_mask.astype(np.float32))), 4)

    # Feature [6]: Laplacian Texture Variance (normalized to [0.0, 1.0])
    # Measures surface roughness caused by pest feeding tunnels or mines
    laplacian_var = float(cv2.Laplacian(gray_image, cv2.CV_64F).var())
    feature_array[6] = round(min(1.0, laplacian_var / 2500.0), 4)

    # Feature [7]: Normalized Brightness Mean
    # Measures overall lighting / exposure level of the leaf photo
    feature_array[7] = round(float(np.mean(val_channel)), 4)

    # Return the fixed-size 8-element feature array
    return feature_array


# ==============================================================================
# STAGE 2: CUSTOM HASH TABLE (POLYNOMIAL ROLLING HASH + SEPARATE CHAINING)
# ==============================================================================

# # [HASH_TABLE]
class HashBucketNode:
    """
    Singly Linked List Node used for Separate Chaining inside a Hash Table bucket.
    When two keys hash to the same bucket index (a collision), they are chained
    together using the `next` pointer.
    """

    def __init__(self, key: str, value: Any) -> None:
        # Store the normalized lookup key string (e.g., "rice|rice leaf roller")
        self.key: str = key
        # Store the associated pest database record object
        self.value: Any = value
        # Pointer to the next node in the bucket chain (None if end of chain)
        self.next: Optional["HashBucketNode"] = None


# # [HASH_TABLE]
class CustomPestHashTable:
    """
    STAGE 2 — CUSTOM HASH TABLE FROM SCRATCH (NO BUILT-IN PYTHON DICT FOR STORAGE)
    Uses a fixed-size array of 31 buckets (`config.HASH_TABLE_BUCKETS`) and a
    Polynomial Rolling Hash function to map `"crop|pest"` keys directly to their
    agricultural pest & treatment records.
    """

    def __init__(self, num_buckets: int = config.HASH_TABLE_BUCKETS) -> None:
        """
        Initializes an empty array of `num_buckets` slots, each set to None.
        Big-O Time Complexity: O(B) where B = num_buckets (31).
        """
        # # [HASH_TABLE] # [CHANGE_HERE] Number of hash buckets comes from config.py
        self.num_buckets: int = num_buckets
        # Fixed-size list representing the underlying hash table bucket array
        self.buckets: List[Optional[HashBucketNode]] = [None] * self.num_buckets
        # Counter tracking total number of key-value pairs stored
        self.size: int = 0

    def compute_polynomial_hash(self, key: str) -> int:
        """
        Computes the bucket index (0 to num_buckets - 1) for a string key using
        the Polynomial Rolling Hash formula:
            hash = sum(ord(char) * P^i) mod num_buckets

        Big-O Time Complexity:
            O(L) where L is the number of characters in the key string (effectively O(1)
            constant time since crop/pest names are short strings).
        """
        # Normalize the key by stripping spaces and converting to lowercase
        clean_key: str = key.strip().lower()
        # Accumulator for the rolling hash value
        hash_value: int = 0
        # Prime multiplier P = 31 from config.py
        base_p: int = config.HASH_POLYNOMIAL_BASE

        # Loop over every character in the key to update the rolling hash
        for char in clean_key:
            hash_value = (hash_value * base_p + ord(char)) % self.num_buckets

        # Return the computed bucket index in [0, num_buckets - 1]
        return hash_value

    def insert(self, key: str, value: Any) -> int:
        """
        Inserts a `(key, value)` pair into the Hash Table using Separate Chaining
        to handle any bucket collisions. Returns the bucket index used.

        Big-O Time Complexity:
            O(1) average time; O(C) worst-case where C is the chain length in that bucket.
        """
        # Normalize the key string for case-insensitive matching
        clean_key: str = key.strip().lower()
        # Step 1: Compute the target bucket index using Polynomial Rolling Hash
        bucket_index: int = self.compute_polynomial_hash(clean_key)

        # Step 2: Check if the bucket is currently empty
        head_node: Optional[HashBucketNode] = self.buckets[bucket_index]
        if head_node is None:
            # No collision: place a new HashBucketNode directly at this bucket slot
            self.buckets[bucket_index] = HashBucketNode(clean_key, value)
            self.size += 1
            return bucket_index

        # Step 3: Bucket is non-empty; traverse the linked list chain
        current: Optional[HashBucketNode] = head_node
        prev: Optional[HashBucketNode] = None
        while current is not None:
            # If the exact key already exists in the chain, update its value in place
            if current.key == clean_key:
                current.value = value
                return bucket_index
            prev = current
            current = current.next

        # Step 4: Append the new node to the end of the bucket's linked list chain
        if prev is not None:
            prev.next = HashBucketNode(clean_key, value)
            self.size += 1

        return bucket_index

    def lookup(self, key: str) -> Tuple[Optional[Any], int, int]:
        """
        Looks up `key` in the Hash Table and returns a 3-tuple:
            `(matched_value_or_None, bucket_index, chain_steps_taken)`

        Big-O Time Complexity:
            O(1) average time; O(C) worst-case where C is the chain length in that bucket.
        """
        # Normalize the key string
        clean_key: str = key.strip().lower()
        # Compute the bucket index in O(1) time
        bucket_index: int = self.compute_polynomial_hash(clean_key)

        # Start at the head of the linked list at `bucket_index`
        current: Optional[HashBucketNode] = self.buckets[bucket_index]
        steps: int = 0

        # Walk the chain until we find the matching key or reach None
        while current is not None:
            steps += 1
            if current.key == clean_key:
                return current.value, bucket_index, steps
            current = current.next

        # Key was not found in the bucket chain
        return None, bucket_index, steps

    def get_all_bucket_stats(self) -> List[Dict[str, Any]]:
        """
        Returns a summary list of all 31 buckets and the keys chained inside each bucket.
        Useful for displaying the Hash Table structure in the Admin Portal and DSA Guide.

        Big-O Time Complexity: O(B + N) where B is bucket count and N is total stored keys.
        """
        stats: List[Dict[str, Any]] = []
        # Iterate through every bucket slot from 0 to num_buckets - 1
        for idx in range(self.num_buckets):
            chain_keys: List[str] = []
            current = self.buckets[idx]
            while current is not None:
                chain_keys.append(current.key)
                current = current.next
            stats.append(
                {
                    "bucket_index": idx,
                    "chain_length": len(chain_keys),
                    "keys": chain_keys,
                }
            )
        return stats


# ==============================================================================
# STAGE 3: DECISION TREE CLASSIFIER (SEVERITY & TREATMENT STRATEGY)
# ==============================================================================

# # [DECISION_TREE] # [SEVERITY]
class DecisionTreeNode:
    """
    Represents a single node in our Binary Decision Tree for Severity Classification.
    - Internal split nodes hold a `threshold`, a `left_child` (<= threshold),
      and a `right_child` (> threshold).
    - Leaf nodes hold the final `severity_label`, `strategy`, and `rule_template`.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        question: str = "",
        left_child: Optional["DecisionTreeNode"] = None,
        right_child: Optional["DecisionTreeNode"] = None,
        is_leaf: bool = False,
        severity_label: str = "",
        strategy: str = "",
        rule_template: str = "",
    ) -> None:
        self.threshold = threshold
        self.question = question
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf
        self.severity_label = severity_label
        self.strategy = strategy
        self.rule_template = rule_template


# # [DECISION_TREE] # [SEVERITY]
class SeverityDecisionTree:
    """
    STAGE 3 — DECISION TREE CLASSIFIER FOR INFESTATION SEVERITY
    Evaluates the affected leaf area percentage (computed from YOLOv8 bounding
    boxes) through a 2-level binary decision tree:
      - Root Split: Is affected_area_percent < 15.0% (`SEVERITY_LOW_MAX_PERCENT`)?
          * YES -> Leaf Node: 'Low' Severity (Preventive Organic / Biological care)
          * NO  -> Second Split: Is affected_area_percent <= 35.0% (`SEVERITY_MODERATE_MAX_PERCENT`)?
              - YES -> Leaf Node: 'Moderate' Severity (Targeted Biological + IPM intervention)
              - NO  -> Leaf Node: 'Severe' Severity (Immediate high-efficacy control)
    """

    def __init__(self) -> None:
        """
        Builds the 3-leaf Decision Tree structure using thresholds from `config.py`.
        Big-O Time Complexity: O(1) constant time to construct the 5 nodes.
        """
        # Leaf Node 1: Low Severity (< 15% leaf area affected)
        low_leaf = DecisionTreeNode(
            is_leaf=True,
            severity_label="Low",
            strategy=(
                "Preventive Organic & Biological Strategy: Use neem/botanical extracts "
                "or bio-control agents to halt early pest establishment with zero chemical residue."
            ),
            rule_template=(
                "Affected leaf area ({area:.1f}%) is below {low_t:.1f}% threshold "
                "-> Classified as Low Severity."
            ),
        )

        # Leaf Node 2: Moderate Severity (15% to 35% leaf area affected)
        moderate_leaf = DecisionTreeNode(
            is_leaf=True,
            severity_label="Moderate",
            strategy=(
                "Integrated Pest Management (IPM) Strategy: Combine biological control "
                "with selective systemic treatment to stop active pest spread."
            ),
            rule_template=(
                "Affected leaf area ({area:.1f}%) is between {low_t:.1f}% and {mod_t:.1f}% "
                "-> Classified as Moderate Severity."
            ),
        )

        # Leaf Node 3: Severe Outbreak (> 35% leaf area affected)
        severe_leaf = DecisionTreeNode(
            is_leaf=True,
            severity_label="Severe",
            strategy=(
                "Immediate High-Efficacy Control Strategy: Apply targeted high-efficacy "
                "formulations immediately while strictly observing the Pre-Harvest Interval (PHI)."
            ),
            rule_template=(
                "Affected leaf area ({area:.1f}%) exceeds {mod_t:.1f}% critical threshold "
                "-> Classified as Severe Infestation."
            ),
        )

        # Internal Node Level 2: Splits between Moderate (<= 35%) and Severe (> 35%)
        level2_node = DecisionTreeNode(
            threshold=config.SEVERITY_MODERATE_MAX_PERCENT,
            question=f"Is affected_area_percent <= {config.SEVERITY_MODERATE_MAX_PERCENT}%?",
            left_child=moderate_leaf,
            right_child=severe_leaf,
            is_leaf=False,
        )

        # Root Node Level 1: Splits between Low (< 15%) and Level 2 (>= 15%)
        self.root = DecisionTreeNode(
            threshold=config.SEVERITY_LOW_MAX_PERCENT,
            question=f"Is affected_area_percent < {config.SEVERITY_LOW_MAX_PERCENT}%?",
            left_child=low_leaf,
            right_child=level2_node,
            is_leaf=False,
        )

    def evaluate(self, affected_area_percent: float) -> Dict[str, Any]:
        """
        Traverses the Decision Tree from `self.root` down to a leaf node based on
        `affected_area_percent`.

        Big-O Time Complexity:
            O(H) where H = 2 is the height of the decision tree (O(1) constant time).
        Big-O Space Complexity:
            O(H) to record the decision path steps.
        """
        # Clamp input percentage to valid range [0.0, 100.0]
        clamped_area: float = max(0.0, min(100.0, float(affected_area_percent)))
        # List recording each comparison step taken during tree traversal
        path_steps: List[str] = []

        # Start traversal at the root node
        current_node: DecisionTreeNode = self.root

        # Walk down the tree until we reach a leaf node
        while not current_node.is_leaf:
            threshold = current_node.threshold or 0.0
            # At root node we check `< 15.0%`; at level-2 node we check `<= 35.0%`
            if current_node is self.root:
                goes_left = clamped_area < threshold
            else:
                goes_left = clamped_area <= threshold

            if goes_left:
                path_steps.append(f"{current_node.question} -> YES ({clamped_area:.1f}%)")
                current_node = current_node.left_child  # type: ignore[assignment]
            else:
                path_steps.append(f"{current_node.question} -> NO ({clamped_area:.1f}%)")
                current_node = current_node.right_child  # type: ignore[assignment]

        # Format the human-readable rule explanation at the leaf node
        rule_explanation = current_node.rule_template.format(
            area=clamped_area,
            low_t=config.SEVERITY_LOW_MAX_PERCENT,
            mod_t=config.SEVERITY_MODERATE_MAX_PERCENT,
        )

        return {
            "severity": current_node.severity_label,
            "strategy": current_node.strategy,
            "rule_explanation": rule_explanation,
            "traversal_path": path_steps,
        }


# ==============================================================================
# STAGE 4: PRIORITY QUEUE (MANUAL BINARY MAX-HEAP FROM SCRATCH, NO HEAPQ)
# ==============================================================================

# # [PRIORITY_QUEUE] # [TREATMENT]
class ManualMaxHeapPriorityQueue:
    """
    STAGE 4 — MANUAL BINARY MAX-HEAP PRIORITY QUEUE (NO BUILT-IN `heapq`)
    Stores candidate treatments in a complete binary tree represented as a dynamic
    array (`self.heap`), where every parent node has a `priority_score` greater
    than or equal to its children. Thus, the #1 best treatment is always at index 0.
    """

    def __init__(self) -> None:
        # Underlying array storing heap elements (each element is a treatment dict)
        self.heap: List[Dict[str, Any]] = []

    def _parent_index(self, index: int) -> int:
        """Returns the parent index of node `index` in O(1) time."""
        return (index - 1) // 2

    def _left_child_index(self, index: int) -> int:
        """Returns the left child index of node `index` in O(1) time."""
        return 2 * index + 1

    def _right_child_index(self, index: int) -> int:
        """Returns the right child index of node `index` in O(1) time."""
        return 2 * index + 2

    def _swap(self, i: int, j: int) -> None:
        """Swaps two elements in the heap array at indices `i` and `j` in O(1) time."""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _sift_up(self, index: int) -> None:
        """
        Restores the Max-Heap property upwards from `index` to the root after insertion.
        Big-O Time Complexity: O(log K) where K is the number of treatments in the heap.
        """
        current = index
        # While current node is not the root and has a higher score than its parent
        while current > 0:
            parent = self._parent_index(current)
            if self.heap[current]["priority_score"] > self.heap[parent]["priority_score"]:
                self._swap(current, parent)
                current = parent
            else:
                break

    def _sift_down(self, index: int) -> None:
        """
        Restores the Max-Heap property downwards from `index` after extracting the root.
        Big-O Time Complexity: O(log K) where K is the number of treatments in the heap.
        """
        size = len(self.heap)
        current = index

        while True:
            largest = current
            left = self._left_child_index(current)
            right = self._right_child_index(current)

            # Check if left child exists and has a higher priority_score than current largest
            if left < size and self.heap[left]["priority_score"] > self.heap[largest]["priority_score"]:
                largest = left

            # Check if right child exists and has a higher priority_score than current largest
            if right < size and self.heap[right]["priority_score"] > self.heap[largest]["priority_score"]:
                largest = right

            # If either child was larger, swap and continue sifting down
            if largest != current:
                self._swap(current, largest)
                current = largest
            else:
                break

    def insert(self, treatment_item: Dict[str, Any]) -> None:
        """
        Inserts a scored treatment item into the Max-Heap and sifts it up.
        Big-O Time Complexity: O(log K) where K is the number of items in the heap.
        """
        # Append new item at the end of the array
        self.heap.append(treatment_item)
        # Bubble it up to its proper max-heap position
        self._sift_up(len(self.heap) - 1)

    def extract_max(self) -> Optional[Dict[str, Any]]:
        """
        Removes and returns the treatment with the highest `priority_score` (root at index 0).
        Big-O Time Complexity: O(log K) where K is the number of items in the heap.
        """
        if not self.heap:
            return None

        # If only one element remains, pop and return it directly
        if len(self.heap) == 1:
            return self.heap.pop()

        # Save the maximum element at the root (index 0)
        max_item = self.heap[0]
        # Move the last element to the root and restore the Max-Heap property downwards
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return max_item

    def is_empty(self) -> bool:
        """Returns True if the Priority Queue is empty in O(1) time."""
        return len(self.heap) == 0


# # [PRIORITY_QUEUE] # [TREATMENT]
def compute_treatment_priority_score(
    treatment: Dict[str, Any], severity: str
) -> float:
    """
    Computes the composite Priority Score for a single treatment option based on:
      1. Efficacy % (higher is better)
      2. Severity-aware category bonus (Organic/Biological preferred in Low/Moderate;
         high-efficacy Chemical gets a bonus in Severe outbreaks)
      3. Cost per Acre in INR penalty (`cost_per_acre_inr / COST_PENALTY_DIVISOR`)
      4. Pre-Harvest Interval (PHI) waiting days penalty (`phi_days * PHI_PENALTY_MULTIPLIER`)

    Big-O Time Complexity: O(1) constant time arithmetic.
    """
    efficacy: float = float(treatment.get("efficacy_percent", 75.0))
    cost_inr: float = float(treatment.get("cost_per_acre_inr", 500.0))
    phi_days: float = float(treatment.get("phi_days", 7.0))
    category: str = str(treatment.get("category", "Organic")).lower()

    # Determine category bonus based on current infestation severity from Stage 3
    category_bonus: float = 0.0
    if "organic" in category or "biological" in category:
        if severity == "Low":
            category_bonus = config.ORGANIC_BONUS_LOW_SEVERITY
        elif severity == "Moderate":
            category_bonus = config.ORGANIC_BONUS_MODERATE_SEVERITY
    elif "chemical" in category and severity == "Severe":
        category_bonus = config.CHEMICAL_BONUS_SEVERE

    # Composite formula using weights from config.py
    raw_score: float = (
        efficacy
        + category_bonus
        - (cost_inr / config.COST_PENALTY_DIVISOR)
        - (phi_days * config.PHI_PENALTY_MULTIPLIER)
    )

    # Clamp score into [PRIORITY_SCORE_MIN, PRIORITY_SCORE_MAX] and round to 1 decimal place
    clamped_score: float = max(
        config.PRIORITY_SCORE_MIN, min(config.PRIORITY_SCORE_MAX, raw_score)
    )
    return round(clamped_score, 1)


# # [PRIORITY_QUEUE] # [TREATMENT]
def rank_treatments_with_priority_queue(
    treatments: List[Dict[str, Any]], severity: str
) -> List[Dict[str, Any]]:
    """
    Scores all candidate treatments for a detected pest, inserts them into our
    `ManualMaxHeapPriorityQueue`, and extracts them in descending order so
    Rank #1 is the optimal recommendation.

    Big-O Time Complexity:
        O(K log K) where K is the number of candidate treatments (typically 3 to 5).
    Big-O Space Complexity:
        O(K) to store the heap elements and ranked output list.
    """
    pq = ManualMaxHeapPriorityQueue()

    # Step 1: Compute priority score for each treatment and insert into Max-Heap
    for item in treatments:
        scored_copy = dict(item)
        scored_copy["priority_score"] = compute_treatment_priority_score(
            scored_copy, severity
        )
        pq.insert(scored_copy)

    # Step 2: Repeatedly extract the maximum element to build the sorted ranking
    ranked_list: List[Dict[str, Any]] = []
    rank_number: int = 1
    while not pq.is_empty():
        best_item = pq.extract_max()
        if best_item is not None:
            best_item["rank"] = rank_number
            ranked_list.append(best_item)
            rank_number += 1

    return ranked_list


# ==============================================================================
# END-TO-END 4-STAGE DSA PIPELINE RUNNER
# ==============================================================================

def run_four_stage_dsa_pipeline(
    image_bgr: np.ndarray,
    crop_name: str,
    pest_name: str,
    affected_area_percent: float,
    hash_table: CustomPestHashTable,
) -> Dict[str, Any]:
    """
    Executes all 4 DSA stages sequentially after CNN + YOLOv8 inference and
    measures the execution time (in milliseconds) of each stage:
      - Stage 1: OpenCV 8-element Feature Array (`extract_feature_array`)
      - Stage 2: Custom 31-Bucket Hash Table O(1) Lookup (`hash_table.lookup`)
      - Stage 3: Decision Tree Severity Classifier (`SeverityDecisionTree.evaluate`)
      - Stage 4: Manual Max-Heap Priority Queue Treatment Ranker

    Big-O Time Complexity:
        O(W * H + 1 + H_tree + K log K) where W*H is image pixels and K is treatment count.
    """
    # --- Stage 1: Feature Array ---
    t0 = time.perf_counter()
    feature_array = extract_feature_array(image_bgr)
    t1 = time.perf_counter()

    # --- Stage 2: Hash Table Lookup ---
    composite_key = f"{crop_name.strip().lower()}|{pest_name.strip().lower()}"
    pest_record, bucket_index, chain_steps = hash_table.lookup(composite_key)
    # Fallback: if crop|pest composite key wasn't matched, try pest_name directly
    if pest_record is None:
        pest_record, bucket_index, chain_steps = hash_table.lookup(pest_name.strip().lower())
    t2 = time.perf_counter()

    # --- Stage 3: Decision Tree Severity Evaluation ---
    tree = SeverityDecisionTree()
    tree_result = tree.evaluate(affected_area_percent)
    severity = tree_result["severity"]
    t3 = time.perf_counter()

    # --- Stage 4: Priority Queue Max-Heap Treatment Ranking ---
    raw_treatments = pest_record.get("treatments", []) if pest_record else []
    ranked_treatments = rank_treatments_with_priority_queue(raw_treatments, severity)
    t4 = time.perf_counter()

    # Calculate per-stage execution times in milliseconds
    stage1_ms = round((t1 - t0) * 1000.0, 3)
    stage2_ms = round((t2 - t1) * 1000.0, 3)
    stage3_ms = round((t3 - t2) * 1000.0, 3)
    stage4_ms = round((t4 - t3) * 1000.0, 3)
    total_ms = round((t4 - t0) * 1000.0, 3)

    return {
        "feature_array": feature_array,
        "feature_names": config.FEATURE_NAMES,
        "hash_key_used": composite_key,
        "hash_bucket_index": bucket_index,
        "hash_chain_steps": chain_steps,
        "pest_record": pest_record,
        "severity": severity,
        "severity_strategy": tree_result["strategy"],
        "decision_rule": tree_result["rule_explanation"],
        "decision_path": tree_result["traversal_path"],
        "ranked_treatments": ranked_treatments,
        "telemetry_ms": {
            "stage1_array_ms": stage1_ms,
            "stage2_hash_ms": stage2_ms,
            "stage3_tree_ms": stage3_ms,
            "stage4_heap_ms": stage4_ms,
            "total_dsa_ms": total_ms,
        },
    }
