import collections
import logging as base_logging
import socket
import time
from typing import Any

import lightning.fabric as lightning_fabric
from absl import logging
from absl.logging import converter

from tts.core import constants


class _HostnameLogFormatter(logging.PythonFormatter):
    """Custom formatter that includes the hostname and local rank."""

    def __init__(self, global_rank: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._hostname = socket.gethostname()
        self._global_rank = global_rank

    def _is_non_absl_fatal_record(self, log_record: Any) -> bool:
        return log_record.levelno >= base_logging.FATAL and not log_record.__dict__.get(
            logging._ABSL_LOG_FATAL, False
        )

    def format(self, record: Any) -> Any:
        created_tuple = time.localtime(record.created)
        created_microsecond = int(record.created % 1.0 * 1e6)

        critical_prefix = ""
        level = record.levelno
        if self._is_non_absl_fatal_record(record):
            # When the level is FATAL, but not logged from absl, lower the level so
            # it's treated as ERROR.
            level = base_logging.ERROR
            critical_prefix = logging._CRITICAL_PREFIX
        severity = converter.get_initial_for_level(level)

        prefix = "%c%02d%02d %02d:%02d:%02d.%06d %s/%d %s:%d] %s" % (  # noqa: UP031
            severity,
            created_tuple.tm_mon,
            created_tuple.tm_mday,
            created_tuple.tm_hour,
            created_tuple.tm_min,
            created_tuple.tm_sec,
            created_microsecond,
            self._hostname,
            self._global_rank,
            record.filename,
            record.lineno,
            critical_prefix,
        )  # noqa: UP031

        return prefix + super(logging.PythonFormatter, self).format(record)


class Statistics:
    """Statistics for all sources.

    To make sure all ranks have same keys for each dict, one needs to pass the list of
    sources to the constructor and use only agreed upon metric names.
    """

    def __init__(self, sources: list[str]):
        if not sources:
            raise ValueError("|sources| cannot be empty.")

        # Supposed to be incremented by the owner of the class.
        self.step = 0
        self.stats_to_sum = collections.defaultdict(float)

        all_sources = sources
        if constants.TOTAL_SOURCE not in sources:
            all_sources.append(constants.TOTAL_SOURCE)

        self.curr_source_counter = dict.fromkeys(all_sources, 0)
        self.curr_metrics = {
            source: collections.defaultdict(float) for source in all_sources
        }

        self.accum_source_counter = dict.fromkeys(all_sources, 0)
        self.accum_metrics = {
            source: collections.defaultdict(float) for source in all_sources
        }
        self.sources = all_sources
        for source in all_sources:
            self.stats_to_sum[source] = 0

    def record(
        self,
        metrics: dict[str, float],
        sources: list[str],
        stats_to_sum: dict[str, float] | None = None,
    ):
        if stats_to_sum is None:
            stats_to_sum = {}
        if constants.TOTAL_SOURCE not in sources:
            raise ValueError("Total source must be present in the sources list.")

        for source in sources:
            if source not in self.sources:
                raise ValueError(
                    f"Source [{source}] not in known sources: {self.sources}."
                )
            # Update total source every time when a source is processed.
            if source != constants.TOTAL_SOURCE:
                self.stats_to_sum[source] += 1
                self.stats_to_sum[constants.TOTAL_SOURCE] += 1
            self.accum_source_counter[source] += 1
            self.curr_source_counter[source] += 1
            for k, v in metrics.items():
                self.accum_metrics[source][k] += v
                self.curr_metrics[source][k] += v

        for k, v in stats_to_sum.items():
            self.stats_to_sum[k] += v

        # Add missing sources with 0.0 values to keep the keys consistent across ranks.
        for missing_source in set(self.sources) - set(sources):
            self.stats_to_sum[missing_source] += 0
            for k in metrics:
                self.curr_metrics[missing_source][k] += 0.0
                self.accum_metrics[missing_source][k] += 0.0

    def start_micro_batch_training(self):
        for source in self.sources:
            self.curr_metrics[source] = collections.defaultdict(float)
            self.curr_source_counter[source] = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "stats_to_sum": dict(self.stats_to_sum),
            "curr_source_counter": dict(self.curr_source_counter),
            "curr_metrics": {
                source: dict(v) for source, v in self.curr_metrics.items()
            },
            "accum_source_counter": dict(self.accum_source_counter),
            "accum_metrics": {
                source: dict(v) for source, v in self.accum_metrics.items()
            },
            "sources": self.sources,
        }

    @staticmethod
    def from_dict(metric_stats_dict: dict[str, Any]) -> "Statistics":
        stats = Statistics(metric_stats_dict["sources"])
        stats.step = metric_stats_dict["step"]
        stats.stats_to_sum = collections.defaultdict(
            float, metric_stats_dict["stats_to_sum"]
        )

        stats.curr_source_counter = collections.Counter(
            metric_stats_dict["curr_source_counter"]
        )
        stats.curr_metrics = {
            source: collections.defaultdict(float)
            for source in metric_stats_dict["curr_metrics"]
        }
        for source, subdict in metric_stats_dict["curr_metrics"].items():
            stats.curr_metrics[source].update(subdict)

        stats.accum_source_counter = collections.Counter(
            metric_stats_dict["accum_source_counter"]
        )
        stats.accum_metrics = {
            source: collections.defaultdict(float)
            for source in metric_stats_dict["accum_metrics"]
        }
        for source, subdict in metric_stats_dict["accum_metrics"].items():
            stats.accum_metrics[source].update(subdict)

        return stats


class Timer:
    """Timer class for measuring time."""

    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs):
        self.time = time.perf_counter() - self.time

    def get_duration(self) -> float:
        """Returns time elapsed in seconds."""
        return self.time


def reconfigure_absl_logging_handler(global_rank: int = 0):
    """Reconfigures the absl logging handler to use the custom formatter."""
    logger = base_logging.getLogger()
    absl_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.ABSLHandler):
            absl_handler = handler
            break

    if absl_handler:
        # Set the formatter to the custom formatter
        absl_handler.setFormatter(_HostnameLogFormatter(global_rank))
    else:
        raise ValueError("No absl handler found in logger handlers.")


def rewrite_logs_for_wandb(logs: dict[str, float]) -> dict[str, float]:
    """Rewrites the logs to be compatible with the pipeline logger."""

    # Replace train_ with train/ and eval_ with eval/.
    new_d = {}
    for key, value in logs.items():
        if key.startswith("train_"):
            new_key = key.replace("train_", "train/")
        elif key.startswith("eval_"):
            new_key = key.replace("eval_", "eval/")
        else:
            new_key = key
        new_d[new_key] = value

    return new_d


def get_logging_stats(
    fabric: lightning_fabric.Fabric,
    statistics: Statistics,
    *,
    steps_per_epoch: int,
    total_samples_per_step: int,
    learning_rate: float,
    running_data_reading_time: float,
    running_step_time: float,
) -> dict[str, Any]:
    """Returns a dictionary with the statistics to be logged for the training set.

    It's assumed each rank calls this function having ensured all ranks have same keys
    for each dictionary used.
    """

    def _compute_avg(
        metric: float, counter: dict[str, float], source: str
    ) -> float | None:
        numerator = fabric.all_reduce(metric, reduce_op="sum").item()
        denominator = fabric.all_reduce(counter.get(source, 0), reduce_op="sum").item()
        if denominator == 0:
            return None
        return numerator / denominator

    if steps_per_epoch == 0:
        raise ValueError("steps_per_epoch cannot be 0.")
    if running_step_time == 0.0:
        raise ValueError("running_step_time cannot be 0.0")

    result = {
        "epoch": statistics.step / steps_per_epoch,
        "learning_rate": learning_rate,
        "step_time": running_step_time,
        "data_reading_time": running_data_reading_time,
        "total_samples_per_sec": total_samples_per_step / running_step_time,
    }

    # Total samples processed.
    for k, v in statistics.stats_to_sum.items():
        renamed_key = f"total_{k}"
        if k in statistics.sources:
            renamed_key = f"{k}_samples_processed"
        result[renamed_key] = fabric.all_reduce(v, reduce_op="sum").item()

    # Averaged metrics per step.
    avg_metric_per_step_reduced = {}
    for source, metrics in statistics.curr_metrics.items():
        for k, v in metrics.items():
            val = _compute_avg(v, statistics.curr_source_counter, source)
            if val is not None:
                avg_metric_per_step_reduced[f"{source}_{k}"] = val
    result.update(avg_metric_per_step_reduced)

    # Accumulated metrics.
    accum_metrics = {}
    for source, metrics in statistics.accum_metrics.items():
        for k, v in metrics.items():
            val = _compute_avg(v, statistics.accum_source_counter, source)
            if val is not None:
                accum_metrics[f"accum_{source}_{k}"] = val
    result.update(accum_metrics)

    # Mark that all computed metrics are for the training set.
    return {f"train_{k}": v for k, v in result.items()}
