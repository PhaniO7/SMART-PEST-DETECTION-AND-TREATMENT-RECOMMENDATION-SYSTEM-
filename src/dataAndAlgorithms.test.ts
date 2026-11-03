/**
 * ============================================================================
 * UNIT TESTS FOR SIMPLIFIED 4-STAGE PIPELINE & IP102 DATASET ALIGNMENT
 * Project By: PHANI NANDAN M S, VEDHESHA PARTHIBAN, KHADGA LOHITH K, SHIVANI BALLANI
 * ============================================================================
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  extractFeatureArray,
  pestHashTable,
  classifySeverityDecisionTree,
  rankTreatmentsPriorityQueue,
  runCompleteDetectionPipeline,
  ALL_CROPS,
  PEST_DATABASE,
  TOTAL_CROPS,
  TOTAL_PESTS,
  PESTS_PER_CROP,
  TOTAL_TREATMENTS
} from './dataAndAlgorithms.ts';

// Summary: Test IP102 dataset alignment (8 crops, >= 3 pests per crop, no crops outside the 8)
test('IP102 Dataset: has exactly 8 crops, at least 3 pests per crop, and no pest outside the 8 crops', () => {
  assert.equal(ALL_CROPS.length, 8);
  assert.equal(TOTAL_CROPS, 8);
  assert.equal(TOTAL_PESTS, PEST_DATABASE.length);
  assert.equal(
    TOTAL_TREATMENTS,
    PEST_DATABASE.reduce((sum, p) => sum + p.treatments.length, 0)
  );

  // Every one of the 8 IP102 crops must have at least 3 common pests
  for (const crop of ALL_CROPS) {
    assert.ok(
      PESTS_PER_CROP[crop] >= 3,
      `Expected ${crop} to have at least 3 pests, got ${PESTS_PER_CROP[crop]}`
    );
  }

  // Every pest must have a valid ip102Class and only reference crops within ALL_CROPS
  for (const pest of PEST_DATABASE) {
    assert.ok(pest.ip102Class && pest.ip102Class.trim().length > 0);
    for (const crop of pest.affectedCrops) {
      assert.ok(
        ALL_CROPS.includes(crop),
        `Pest ${pest.commonName} references crop ${crop} outside the 8 IP102 crops`
      );
    }
  }
});

// Summary: Test Stage 1 - Feature Array returns 8 normalized numbers
test('Stage 1 (Feature Array): extracts 8 normalized feature values', () => {
  const arr = extractFeatureArray('Rice_Leaf_Sample');
  assert.equal(arr.length, 8);
  assert.ok(arr.every((val) => val >= 0.1 && val <= 1.0));
});

// Summary: Test Stage 2 - Hash Table O(1) lookup finds pests for a crop
test('Stage 2 (Hash Table): O(1) lookup retrieves matching pests and bucket index', () => {
  const result = pestHashTable.lookupByCrop('Rice');
  assert.ok(result.pests.length >= 3);
  assert.equal(result.pests[0].commonName, 'Brown Planthopper');
  assert.equal(result.pests[0].ip102Class, 'brown plant hopper');
  assert.equal(result.bucketIndex, 16);
  assert.ok(result.bucketIndex >= 0 && result.bucketIndex < 31);
});

// Summary: Test Stage 3 - Decision Tree branches correctly on 15% and 35% thresholds
test('Stage 3 (Decision Tree): classifies Low, Moderate, and Severe accurately', () => {
  assert.equal(classifySeverityDecisionTree(10).severity, 'Low');
  assert.equal(classifySeverityDecisionTree(25).severity, 'Moderate');
  assert.equal(classifySeverityDecisionTree(45).severity, 'Severe');
});

// Summary: Test Stage 4 - Priority Queue sorts treatments in descending score order
test('Stage 4 (Priority Queue): ranks treatments from highest score to lowest', () => {
  const sampleTreatments = PEST_DATABASE[0].treatments;
  const ranked = rankTreatmentsPriorityQueue(sampleTreatments, 'Severe');
  assert.equal(ranked.length, 3);
  assert.ok((ranked[0].priorityScore || 0) >= (ranked[1].priorityScore || 0));
  assert.ok((ranked[1].priorityScore || 0) >= (ranked[2].priorityScore || 0));
});

// Summary: Test complete end-to-end pipeline
test('Complete Pipeline: executes all 4 stages and returns a structured ScanResult', () => {
  const scan = runCompleteDetectionPipeline({
    cropInput: 'Rice',
    affectedAreaPercent: 42
  });
  assert.equal(scan.crop, 'Rice');
  assert.equal(scan.severity, 'Severe');
  assert.ok(scan.rankedTreatments.length > 0);
});
