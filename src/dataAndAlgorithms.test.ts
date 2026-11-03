/**
 * ============================================================================
 * UNIT TESTS FOR SIMPLIFIED 4-STAGE PIPELINE
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
  PEST_DATABASE
} from './dataAndAlgorithms.ts';

// Summary: Test Stage 1 - Feature Array returns 8 normalized numbers
test('Stage 1 (Feature Array): extracts 8 normalized feature values', () => {
  const arr = extractFeatureArray('Tomato_Leaf_Sample');
  assert.equal(arr.length, 8);
  assert.ok(arr.every((val) => val >= 0.1 && val <= 1.0));
});

// Summary: Test Stage 2 - Hash Table O(1) lookup finds pests for a crop
test('Stage 2 (Hash Table): O(1) lookup retrieves matching pests and bucket index', () => {
  const result = pestHashTable.lookupByCrop('Rice');
  assert.ok(result.pests.length >= 1);
  assert.equal(result.pests[0].commonName, 'Brown Planthopper');
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
    cropInput: 'Tomato',
    affectedAreaPercent: 42
  });
  assert.equal(scan.crop, 'Tomato');
  assert.equal(scan.severity, 'Severe');
  assert.ok(scan.rankedTreatments.length > 0);
});
