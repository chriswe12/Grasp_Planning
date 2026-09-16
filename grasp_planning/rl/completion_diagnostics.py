"""Streaming classification and calibration metrics for the policy stop head."""

from __future__ import annotations

import math
from collections.abc import Iterable


class CompletionDiagnostics:
    """Accumulate completion metrics without retaining every rendered step."""

    def __init__(self, *, threshold: float, bin_count: int = 10) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between zero and one")
        if bin_count < 2:
            raise ValueError("bin_count must be at least two")
        self.threshold = float(threshold)
        self.bin_count = int(bin_count)
        self.total_samples = 0
        self.supervised_samples = 0
        self.ambiguous_samples = 0
        self.positive_samples = 0
        self.negative_samples = 0
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0
        self.brier_sum = 0.0
        self.log_loss_sum = 0.0
        self.positive_probability_sum = 0.0
        self.negative_probability_sum = 0.0
        self.bin_counts = [0] * self.bin_count
        self.bin_probability_sums = [0.0] * self.bin_count
        self.bin_target_sums = [0.0] * self.bin_count

    def update(
        self,
        probabilities: Iterable[float],
        ready: Iterable[bool],
        supervised: Iterable[bool] | None = None,
    ) -> None:
        probabilities = list(probabilities)
        ready = list(ready)
        supervised = [True] * len(probabilities) if supervised is None else list(supervised)
        if not (len(probabilities) == len(ready) == len(supervised)):
            raise ValueError("probabilities, ready, and supervised must have equal lengths")
        self.total_samples += len(probabilities)
        for probability, target, use_sample in zip(probabilities, ready, supervised, strict=True):
            probability = min(max(float(probability), 0.0), 1.0)
            target = bool(target)
            if not use_sample:
                self.ambiguous_samples += 1
                continue
            self.supervised_samples += 1
            predicted = probability >= self.threshold
            if target:
                self.positive_samples += 1
                self.positive_probability_sum += probability
                if predicted:
                    self.true_positive += 1
                else:
                    self.false_negative += 1
            else:
                self.negative_samples += 1
                self.negative_probability_sum += probability
                if predicted:
                    self.false_positive += 1
                else:
                    self.true_negative += 1
            target_float = float(target)
            self.brier_sum += (probability - target_float) ** 2
            clipped = min(max(probability, 1.0e-7), 1.0 - 1.0e-7)
            self.log_loss_sum -= target_float * math.log(clipped) + (1.0 - target_float) * math.log(1.0 - clipped)
            bin_index = min(int(probability * self.bin_count), self.bin_count - 1)
            self.bin_counts[bin_index] += 1
            self.bin_probability_sums[bin_index] += probability
            self.bin_target_sums[bin_index] += target_float

    def summary(self) -> dict[str, object]:
        sample_count = self.supervised_samples
        predicted_positive = self.true_positive + self.false_positive
        expected_positive = self.true_positive + self.false_negative
        expected_negative = self.true_negative + self.false_positive
        bins: list[dict[str, object]] = []
        expected_calibration_error = 0.0
        for index, count in enumerate(self.bin_counts):
            if count:
                probability_mean = self.bin_probability_sums[index] / count
                ready_rate = self.bin_target_sums[index] / count
                expected_calibration_error += count / max(sample_count, 1) * abs(probability_mean - ready_rate)
            else:
                probability_mean = None
                ready_rate = None
            bins.append(
                {
                    "lower": index / self.bin_count,
                    "upper": (index + 1) / self.bin_count,
                    "count": count,
                    "probability_mean": probability_mean,
                    "ready_rate": ready_rate,
                }
            )
        return {
            "threshold": self.threshold,
            "total_samples": self.total_samples,
            "supervised_samples": sample_count,
            "ambiguous_samples": self.ambiguous_samples,
            "positive_samples": self.positive_samples,
            "negative_samples": self.negative_samples,
            "positive_rate": self.positive_samples / sample_count if sample_count else 0.0,
            "precision": self.true_positive / predicted_positive if predicted_positive else 0.0,
            "recall": self.true_positive / expected_positive if expected_positive else 0.0,
            "false_positive_rate": self.false_positive / expected_negative if expected_negative else 0.0,
            "brier_score": self.brier_sum / sample_count if sample_count else 0.0,
            "log_loss": self.log_loss_sum / sample_count if sample_count else 0.0,
            "expected_calibration_error": expected_calibration_error,
            "ready_probability_mean": (
                self.positive_probability_sum / self.positive_samples if self.positive_samples else 0.0
            ),
            "negative_probability_mean": (
                self.negative_probability_sum / self.negative_samples if self.negative_samples else 0.0
            ),
            "confusion": {
                "true_positive": self.true_positive,
                "false_positive": self.false_positive,
                "true_negative": self.true_negative,
                "false_negative": self.false_negative,
            },
            "calibration_bins": bins,
        }


__all__ = ["CompletionDiagnostics"]
