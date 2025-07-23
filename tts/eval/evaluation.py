import collections

import lightning.fabric as lightning_fabric
import torch

from tts.core import constants
from tts.data import tts_datasets
from tts.utils import custom_logging


@torch.no_grad()
def _get_health_stats(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    fabric: lightning_fabric.Fabric,
) -> dict[str, float]:
    """Returns health statistics of the model and optimizer."""
    # Maximum absolute values for gradients and parameters.
    max_abs_param = torch.tensor(0.0, device=fabric.device, dtype=torch.float32)
    max_abs_grad = torch.tensor(0.0, device=fabric.device, dtype=torch.float32)

    # Average absolute values for gradients and parameters.
    avg_abs_param = torch.tensor(0.0, device=fabric.device, dtype=torch.float32)
    avg_abs_grad = torch.tensor(0.0, device=fabric.device, dtype=torch.float32)

    # Average values for gradients and parameters.
    avg_param = torch.tensor(0.0, device=fabric.device, dtype=torch.float32)
    avg_grad = torch.tensor(0.0, device=fabric.device, dtype=torch.float32)

    # Number of parameters and gradients.
    param_counter = torch.tensor(0, device=fabric.device, dtype=torch.long)
    grad_counter = torch.tensor(0, device=fabric.device, dtype=torch.long)

    for p in model.parameters():
        if p.requires_grad and p.data.numel():
            param_counter += p.data.numel()
            avg_param += torch.sum(p.data)
            abs_param = torch.abs(p.data)
            avg_abs_param += torch.sum(abs_param)
            max_abs_param = torch.max(torch.max(abs_param), max_abs_param)

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None and p.grad.data.numel() > 0:
                grad_counter += p.grad.data.numel()
                avg_grad += torch.sum(p.grad.data)
                abs_grad = torch.abs(p.grad.data)
                avg_abs_grad += torch.sum(abs_grad)
                max_abs_grad = torch.max(torch.max(abs_grad), max_abs_grad)

    max_abs_grad_reduced = fabric.all_reduce(max_abs_grad, reduce_op="max").item()
    max_abs_param_reduced = fabric.all_reduce(max_abs_param, reduce_op="max").item()
    avg_abs_grad_reduced = fabric.all_reduce(avg_abs_grad, reduce_op="sum").item()
    avg_abs_param_reduced = fabric.all_reduce(avg_abs_param, reduce_op="sum").item()
    avg_grad_reduced = fabric.all_reduce(avg_grad, reduce_op="sum").item()
    avg_param_reduced = fabric.all_reduce(avg_param, reduce_op="sum").item()
    param_counter_reduced = fabric.all_reduce(param_counter, reduce_op="sum").item()
    grad_counter_reduced = fabric.all_reduce(grad_counter, reduce_op="sum").item()
    if param_counter_reduced:
        avg_abs_param_reduced /= param_counter_reduced
        avg_param_reduced /= param_counter_reduced
    if grad_counter_reduced:
        avg_abs_grad_reduced /= grad_counter_reduced
        avg_grad_reduced /= grad_counter_reduced

    return {
        "eval_max_abs_grad": max_abs_grad_reduced,
        "eval_max_abs_param": max_abs_param_reduced,
        "eval_avg_abs_grad": avg_abs_grad_reduced,
        "eval_avg_abs_param": avg_abs_param_reduced,
        "eval_avg_grad": avg_grad_reduced,
        "eval_avg_param": avg_param_reduced,
    }


@torch.no_grad()
def _estimate_eval_loss(
    model: torch.nn.Module, val_data_loader: torch.utils.data.DataLoader
) -> dict[str, float]:
    """Computes average losses over the validation dataset."""
    per_source_loss = collections.defaultdict(lambda: torch.zeros(len(val_data_loader)))
    per_source_loss_count = collections.Counter()
    for batch_idx, batch in enumerate(val_data_loader):
        batch_sources = [constants.TOTAL_SOURCE] + batch.pop("source")
        batch = tts_datasets.prettify_data_sample(batch)
        batch_loss_value = model(**batch).loss.item()

        for source in batch_sources:
            per_source_loss[source][batch_idx] += batch_loss_value
            per_source_loss_count[source] += 1

    result = {}
    for k, v in per_source_loss.items():
        key = f"eval_{k}_loss"
        result[key] = 0.0
        denominator = per_source_loss_count[k]
        if denominator > 0:
            result[key] = v.sum().item() / denominator

    torch.cuda.empty_cache()
    return result


def compute_metrics(
    fabric: lightning_fabric.Fabric,
    model: torch.nn.Module,
    val_data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    collect_health_stats: bool,
) -> dict[str, float]:
    """Evaluates the model on specified evaluation datasets."""
    with custom_logging.Timer() as t:
        metrics = _estimate_eval_loss(model, val_data_loader)
        if collect_health_stats:
            metrics.update(_get_health_stats(model, optimizer, fabric))
    metrics.update({"eval_runtime": t.get_duration()})
    return metrics
