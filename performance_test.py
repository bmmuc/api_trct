"""
Performance Testing Script for Anomaly Detection API

This script performs comprehensive performance testing including:
- Concurrent training of multiple models
- Concurrent inference requests (100-500 parallel)
- Result validation (determinism, correctness, consistency)
- Metrics comparison before/after load
- Detailed benchmark report generation
"""

import asyncio
import aiohttp
import numpy as np
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_seconds: float
    min_latency_ms: float
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    throughput_rps: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestConfig:
    """Configuration for performance tests"""
    api_base_url: str = "http://127.0.0.1:8000"
    num_training_series: int = 10
    training_points_per_series: int = 100
    num_concurrent_predictions: int = 200
    num_determinism_tests: int = 5
    timeout_seconds: int = 30


class PerformanceTester:
    """Main performance testing class"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.session = None
        self.trained_series: Dict[str, Dict[str, Any]] = {}
        self.latencies: List[float] = []

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def generate_training_data(self, series_id: str, mean: float = 100.0, std: float = 10.0,
                              num_points: int = 100) -> Tuple[List[int], List[float]]:
        """Generate synthetic training data with normal distribution"""
        base_timestamp = int(time.time()) - num_points * 3600
        timestamps = [base_timestamp + i * 3600 for i in range(num_points)]
        values = np.random.normal(mean, std, num_points).tolist()
        return timestamps, values

    def generate_test_point(self, series_id: str, is_anomaly: bool = False) -> Tuple[int, float]:
        """Generate a single test point (normal or anomalous)"""
        series_info = self.trained_series.get(series_id, {})
        mean = series_info.get('mean', 100.0)
        std = series_info.get('std', 10.0)

        timestamp = int(time.time())

        if is_anomaly:
            value = mean + 3.5 * std + np.random.uniform(0, std)
        else:
            value = np.random.normal(mean, std)

        return timestamp, value

    async def get_healthcheck(self) -> Dict[str, Any]:
        """Get current healthcheck metrics"""
        url = f"{self.config.api_base_url}/healthcheck"
        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"Status {response.status}"}

    async def train_model(self, series_id: str, timestamps: List[int],
                         values: List[float]) -> Tuple[bool, float, Dict[str, Any]]:
        """Train a single model and return success status, latency, and response"""
        url = f"{self.config.api_base_url}/fit/{series_id}"
        payload = {
            "timestamps": timestamps,
            "values": values
        }

        start_time = time.time()
        try:
            async with self.session.post(url, json=payload) as response:
                latency_ms = (time.time() - start_time) * 1000
                if response.status == 200:
                    result = await response.json()
                    return True, latency_ms, result
                else:
                    return False, latency_ms, {"error": f"Status {response.status}"}
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return False, latency_ms, {"error": str(e)}

    async def predict(self, series_id: str, timestamp: int, value: float,
                     version: str = None) -> Tuple[bool, float, Dict[str, Any]]:
        """Make a single prediction and return success status, latency, and response"""
        url = f"{self.config.api_base_url}/predict/{series_id}"
        if version:
            url += f"?version={version}"

        payload = {
            "timestamp": str(timestamp),
            "value": value
        }

        start_time = time.time()
        try:
            async with self.session.post(url, json=payload) as response:
                latency_ms = (time.time() - start_time) * 1000
                if response.status == 200:
                    result = await response.json()
                    return True, latency_ms, result
                else:
                    return False, latency_ms, {"error": f"Status {response.status}"}
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return False, latency_ms, {"error": str(e)}

    async def test_concurrent_training(self) -> PerformanceMetrics:
        """Test concurrent training of multiple models"""
        print(f"\n{'='*80}")
        print(f"CONCURRENT TRAINING TEST - Training {self.config.num_training_series} models in parallel")
        print(f"{'='*80}")

        training_data = {}
        for i in range(self.config.num_training_series):
            series_id = f"sensor_{i:03d}"
            mean = 100.0 + i * 10  # Vary the mean for each series
            std = 10.0 + i * 2     # Vary the std for each series
            timestamps, values = self.generate_training_data(
                series_id, mean=mean, std=std, num_points=self.config.training_points_per_series
            )
            training_data[series_id] = {
                'timestamps': timestamps,
                'values': values,
                'mean': mean,
                'std': std
            }

        start_time = time.time()
        tasks = [
            self.train_model(series_id, data['timestamps'], data['values'])
            for series_id, data in training_data.items()
        ]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        latencies = []
        successful = 0
        failed = 0

        for i, (success, latency, response) in enumerate(results):
            series_id = f"sensor_{i:03d}"
            if success:
                successful += 1
                latencies.append(latency)
                self.trained_series[series_id] = {
                    **training_data[series_id],
                    'version': response.get('version'),
                    'points_used': response.get('points_used')
                }
                print(f"✓ {series_id}: {latency:.2f}ms - version {response.get('version')}")
            else:
                failed += 1
                print(f"✗ {series_id}: FAILED - {response.get('error', 'Unknown error')}")

        metrics = PerformanceMetrics(
            total_requests=len(results),
            successful_requests=successful,
            failed_requests=failed,
            total_time_seconds=total_time,
            min_latency_ms=min(latencies) if latencies else 0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            median_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            throughput_rps=successful / total_time if total_time > 0 else 0
        )

        print(f"\nTraining Results:")
        print(f"  Success Rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {metrics.throughput_rps:.2f} requests/sec")
        print(f"  Latency - Avg: {metrics.avg_latency_ms:.2f}ms, P95: {metrics.p95_latency_ms:.2f}ms, P99: {metrics.p99_latency_ms:.2f}ms")

        return metrics

    async def test_concurrent_inference(self) -> Tuple[PerformanceMetrics, Dict[str, Any]]:
        """Test concurrent inference with mixed normal and anomalous points"""
        print(f"\n{'='*80}")
        print(f"CONCURRENT INFERENCE TEST - {self.config.num_concurrent_predictions} parallel predictions")
        print(f"{'='*80}")

        if not self.trained_series:
            raise ValueError("No trained models available. Run training test first.")

        # Generate test cases (70% normal, 30% anomalous)
        test_cases = []
        series_ids = list(self.trained_series.keys())

        for i in range(self.config.num_concurrent_predictions):
            series_id = series_ids[i % len(series_ids)]
            is_anomaly = (i % 10) < 3  # 30% anomalous
            timestamp, value = self.generate_test_point(series_id, is_anomaly)
            test_cases.append({
                'series_id': series_id,
                'timestamp': timestamp,
                'value': value,
                'expected_anomaly': is_anomaly
            })

        # Make predictions concurrently
        start_time = time.time()
        tasks = [
            self.predict(case['series_id'], case['timestamp'], case['value'])
            for case in test_cases
        ]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Process results
        latencies = []
        successful = 0
        failed = 0
        correct_predictions = 0
        incorrect_predictions = 0

        for i, (success, latency, response) in enumerate(results):
            if success:
                successful += 1
                latencies.append(latency)

                # Check prediction correctness
                predicted_anomaly = response.get('anomaly', False)
                expected_anomaly = test_cases[i]['expected_anomaly']

                if predicted_anomaly == expected_anomaly:
                    correct_predictions += 1
                else:
                    incorrect_predictions += 1
            else:
                failed += 1

        # Calculate metrics
        metrics = PerformanceMetrics(
            total_requests=len(results),
            successful_requests=successful,
            failed_requests=failed,
            total_time_seconds=total_time,
            min_latency_ms=min(latencies) if latencies else 0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            median_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            throughput_rps=successful / total_time if total_time > 0 else 0
        )

        validation_results = {
            'correct_predictions': correct_predictions,
            'incorrect_predictions': incorrect_predictions,
            'accuracy': correct_predictions / successful if successful > 0 else 0
        }

        print(f"\nInference Results:")
        print(f"  Success Rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {metrics.throughput_rps:.2f} requests/sec")
        print(f"  Latency - Avg: {metrics.avg_latency_ms:.2f}ms, P95: {metrics.p95_latency_ms:.2f}ms, P99: {metrics.p99_latency_ms:.2f}ms")
        print(f"  Prediction Accuracy: {correct_predictions}/{successful} ({100*validation_results['accuracy']:.1f}%)")

        return metrics, validation_results

    async def test_determinism(self) -> Dict[str, Any]:
        """Test that same inputs produce same outputs"""
        print(f"\n{'='*80}")
        print(f"DETERMINISM TEST - Verifying consistent predictions")
        print(f"{'='*80}")

        if not self.trained_series:
            raise ValueError("No trained models available. Run training test first.")

        # Pick a random series and test point
        series_id = list(self.trained_series.keys())[0]
        timestamp, value = self.generate_test_point(series_id, is_anomaly=False)

        # Make the same prediction multiple times
        results = []
        for i in range(self.config.num_determinism_tests):
            success, latency, response = await self.predict(series_id, timestamp, value)
            if success:
                results.append(response.get('anomaly'))

        # Check if all results are identical
        is_deterministic = len(set(results)) == 1

        print(f"  Series: {series_id}")
        print(f"  Test Point: timestamp={timestamp}, value={value:.2f}")
        print(f"  Predictions: {results}")
        print(f"  Deterministic: {'✓ YES' if is_deterministic else '✗ NO'}")

        return {
            'is_deterministic': is_deterministic,
            'predictions': results,
            'num_tests': len(results)
        }

    async def run_full_test_suite(self) -> Dict[str, Any]:
        """Run the complete performance test suite"""
        print(f"\n{'#'*80}")
        print(f"# ANOMALY DETECTION API - PERFORMANCE TEST SUITE")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# Configuration:")
        print(f"#   API URL: {self.config.api_base_url}")
        print(f"#   Training Series: {self.config.num_training_series}")
        print(f"#   Concurrent Predictions: {self.config.num_concurrent_predictions}")
        print(f"{'#'*80}")

        # Get baseline metrics
        print("\nCollecting baseline metrics...")
        baseline_metrics = await self.get_healthcheck()
        print(f"Baseline - Series Trained: {baseline_metrics.get('series_trained', 0)}")

        # Run tests
        training_metrics = await self.test_concurrent_training()
        inference_metrics, validation_results = await self.test_concurrent_inference()
        determinism_results = await self.test_determinism()

        # Get post-test metrics
        print("\nCollecting post-test metrics...")
        post_test_metrics = await self.get_healthcheck()
        print(f"Post-Test - Series Trained: {post_test_metrics.get('series_trained', 0)}")

        # Compile full report
        report = {
            'test_config': {
                'api_base_url': self.config.api_base_url,
                'num_training_series': self.config.num_training_series,
                'num_concurrent_predictions': self.config.num_concurrent_predictions,
                'timestamp': datetime.now().isoformat()
            },
            'baseline_metrics': baseline_metrics,
            'training_performance': training_metrics.to_dict(),
            'inference_performance': inference_metrics.to_dict(),
            'validation': validation_results,
            'determinism': determinism_results,
            'post_test_metrics': post_test_metrics
        }

        return report


def generate_markdown_report(report: Dict[str, Any]) -> str:
    """Generate a formatted markdown report"""
    md = []
    md.append("# Anomaly Detection API - Performance Test Report\n")
    md.append(f"**Generated:** {report['test_config']['timestamp']}\n")
    md.append(f"**API URL:** {report['test_config']['api_base_url']}\n")

    md.append("\n## Test Configuration\n")
    md.append(f"- Training Series: {report['test_config']['num_training_series']}")
    md.append(f"- Concurrent Predictions: {report['test_config']['num_concurrent_predictions']}\n")

    md.append("\n## Training Performance\n")
    training = report['training_performance']
    md.append(f"- **Total Requests:** {training['total_requests']}")
    md.append(f"- **Success Rate:** {training['successful_requests']}/{training['total_requests']} ({100*training['successful_requests']/training['total_requests']:.1f}%)")
    md.append(f"- **Total Time:** {training['total_time_seconds']:.2f}s")
    md.append(f"- **Throughput:** {training['throughput_rps']:.2f} req/s")
    md.append(f"\n**Client-Side End-to-End Latency (includes network + queueing + processing):**")
    md.append(f"- Min: {training['min_latency_ms']:.2f}ms")
    md.append(f"- Avg: {training['avg_latency_ms']:.2f}ms")
    md.append(f"- Median: {training['median_latency_ms']:.2f}ms")
    md.append(f"- P95: {training['p95_latency_ms']:.2f}ms")
    md.append(f"- P99: {training['p99_latency_ms']:.2f}ms")
    md.append(f"- Max: {training['max_latency_ms']:.2f}ms\n")

    md.append("\n## Inference Performance\n")
    inference = report['inference_performance']
    md.append(f"- **Total Requests:** {inference['total_requests']}")
    md.append(f"- **Success Rate:** {inference['successful_requests']}/{inference['total_requests']} ({100*inference['successful_requests']/inference['total_requests']:.1f}%)")
    md.append(f"- **Total Time:** {inference['total_time_seconds']:.2f}s")
    md.append(f"- **Throughput:** {inference['throughput_rps']:.2f} req/s")
    md.append(f"\n**Client-Side End-to-End Latency (includes network + queueing + processing):**")
    md.append(f"- Min: {inference['min_latency_ms']:.2f}ms")
    md.append(f"- Avg: {inference['avg_latency_ms']:.2f}ms")
    md.append(f"- Median: {inference['median_latency_ms']:.2f}ms")
    md.append(f"- P95: {inference['p95_latency_ms']:.2f}ms")
    md.append(f"- P99: {inference['p99_latency_ms']:.2f}ms")
    md.append(f"- Max: {inference['max_latency_ms']:.2f}ms\n")

    md.append("\n## Validation Results\n")
    validation = report['validation']
    md.append(f"- **Prediction Accuracy:** {validation['correct_predictions']}/{validation['correct_predictions'] + validation['incorrect_predictions']} ({100*validation['accuracy']:.1f}%)")
    md.append(f"- Correct Predictions: {validation['correct_predictions']}")
    md.append(f"- Incorrect Predictions: {validation['incorrect_predictions']}\n")

    md.append("\n## Determinism Test\n")
    determinism = report['determinism']
    md.append(f"- **Is Deterministic:** {'✓ YES' if determinism['is_deterministic'] else '✗ NO'}")
    md.append(f"- Number of Tests: {determinism['num_tests']}")
    md.append(f"- All Predictions Identical: {determinism['is_deterministic']}\n")

    md.append("\n## Post-Test Metrics\n")
    post = report['post_test_metrics']
    md.append(f"- Series Trained: {post.get('series_trained', 0)}")
    if post.get('inference_latency_ms'):
        inf_lat = post['inference_latency_ms']
        md.append(f"- Inference Latency:")
        md.append(f"  - Avg: {inf_lat.get('avg', 'N/A')}ms")
        md.append(f"  - P95: {inf_lat.get('p95', 'N/A')}ms")
    if post.get('training_latency_ms'):
        train_lat = post['training_latency_ms']
        md.append(f"- Training Latency:")
        md.append(f"  - Avg: {train_lat.get('avg', 'N/A')}ms")
        md.append(f"  - P95: {train_lat.get('p95', 'N/A')}ms\n")

    md.append("\n## Summary\n")
    md.append(f"The API successfully handled:")
    md.append(f"- {training['successful_requests']} concurrent training requests with {training['throughput_rps']:.2f} req/s throughput")
    md.append(f"- {inference['successful_requests']} concurrent inference requests with {inference['throughput_rps']:.2f} req/s throughput")
    md.append(f"- Maintained {100*validation['accuracy']:.1f}% prediction accuracy under load")
    md.append(f"- Demonstrated {'deterministic' if determinism['is_deterministic'] else 'non-deterministic'} behavior")

    return '\n'.join(md)


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Performance test for Anomaly Detection API')
    parser.add_argument('--url', default='http://127.0.0.1:8000', help='API base URL')
    parser.add_argument('--training-series', type=int, default=1000, help='Number of series to train')
    parser.add_argument('--predictions', type=int, default=1000, help='Number of concurrent predictions')
    parser.add_argument('--output', default='performance_report.md', help='Output report file')
    parser.add_argument('--json-output', default='performance_report.json', help='JSON output file')

    args = parser.parse_args()

    config = TestConfig(
        api_base_url=args.url,
        num_training_series=args.training_series,
        num_concurrent_predictions=args.predictions
    )

    async with PerformanceTester(config) as tester:
        try:
            report = await tester.run_full_test_suite()

            # Generate and save markdown report
            md_report = generate_markdown_report(report)
            with open(args.output, 'w') as f:
                f.write(md_report)
            print(f"\n{'='*80}")
            print(f"Markdown report saved to: {args.output}")

            # Save JSON report
            with open(args.json_output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"JSON report saved to: {args.json_output}")
            print(f"{'='*80}\n")

            # Print summary
            print("\n" + md_report)

        except Exception as e:
            print(f"\n✗ Test suite failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1

    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)
