# Anomaly Detection API - Performance Test Report

**Generated:** 2025-11-02T22:41:23.708784

**API URL:** http://127.0.0.1:8000


## Test Configuration

- Training Series: 1000
- Concurrent Predictions: 1000


## Training Performance

- **Total Requests:** 1000
- **Success Rate:** 1000/1000 (100.0%)
- **Total Time:** 0.78s
- **Throughput:** 1285.30 req/s

**Client-Side End-to-End Latency (includes network + queueing + processing):**
- Min: 106.96ms
- Avg: 341.97ms
- Median: 344.97ms
- P95: 567.50ms
- P99: 597.70ms
- Max: 607.36ms


## Inference Performance

- **Total Requests:** 1000
- **Success Rate:** 1000/1000 (100.0%)
- **Total Time:** 0.58s
- **Throughput:** 1723.13 req/s

**Client-Side End-to-End Latency (includes network + queueing + processing):**
- Min: 41.10ms
- Avg: 258.20ms
- Median: 257.83ms
- P95: 470.89ms
- P99: 489.54ms
- Max: 495.00ms


## Validation Results

- **Prediction Accuracy:** 1000/1000 (100.0%)
- Correct Predictions: 1000
- Incorrect Predictions: 0


## Determinism Test

- **Is Deterministic:** ✓ YES
- Number of Tests: 5
- All Predictions Identical: True


## Post-Test Metrics

- Series Trained: 1000
- Inference Latency:
  - Avg: 0.3311542615498582ms
  - P95: 0.4984855651855469ms
- Training Latency:
  - Avg: 0.7598506605654733ms
  - P95: 1.0830163955688477ms


## Summary

The API successfully handled:
- 1000 concurrent training requests with 1285.30 req/s throughput
- 1000 concurrent inference requests with 1723.13 req/s throughput
- Maintained 100.0% prediction accuracy under load
- Demonstrated deterministic behavior