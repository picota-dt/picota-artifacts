from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch
from torch import nn

from kan.MetamorphicLoss import (
    Batch,
    Equal,
    Greater,
    GreaterOrEqual,
    Lower,
    LowerOrEqual,
    MetamorphicRelation,
    MetamorphicTest,
    Monotonic,
    Proportional,
    TransformSet,
)


def clone_batch(batch: Batch) -> Batch:
    cloned: Batch = {}
    for key, value in batch.items():
        cloned[key] = value.clone() if torch.is_tensor(value) else deepcopy(value)
    return cloned


def _clone_target(target: torch.Tensor) -> torch.Tensor:
    return target.clone() if torch.is_tensor(target) else deepcopy(target)


def _apply_target_transform(transform_spec, target: torch.Tensor, source_batch: Batch, transformed_batch: Batch):
    if getattr(transform_spec, "target_transform", None) is None:
        return target
    cloned_target = _clone_target(target)
    target_transform = transform_spec.target_transform
    try:
        return target_transform(cloned_target, source_batch, transformed_batch)
    except TypeError:
        return target_transform(cloned_target)


def compute_violation_report(
        model,
        data_loader,
        metamorphic_tests: Iterable[MetamorphicTest],
        atol: float = 1e-6,
        rtol: float = 1e-4,
) -> dict:
    model.eval()
    tests = list(metamorphic_tests)
    per_test: dict[str, dict[str, float | int]] = {}

    with torch.no_grad():
        for batch in data_loader:
            base_pred = model(batch).squeeze().reshape(-1)

            for i, test in enumerate(tests):
                test_name = test.name or f"{test.relation.kind.value}_{i}"
                transformed = test.transform(clone_batch(batch))
                transformed_pred = model(transformed).squeeze().reshape(-1)
                local_atol = atol if getattr(test, "violation_atol", None) is None else float(test.violation_atol)
                local_rtol = rtol if getattr(test, "violation_rtol", None) is None else float(test.violation_rtol)
                violations = violation_mask(test.relation, base_pred, transformed_pred, atol=local_atol,
                                            rtol=local_rtol)

                if test_name not in per_test:
                    per_test[test_name] = {"violations": 0, "total": 0}

                per_test[test_name]["violations"] += int(violations.sum().item())
                per_test[test_name]["total"] += int(violations.numel())

    total_violations = 0
    total_cases = 0
    for test_name, stats in per_test.items():
        violations = int(stats["violations"])
        total = int(stats["total"])
        stats["violation_rate"] = (violations / total) if total else float("nan")
        total_violations += violations
        total_cases += total

    return {
        "overall_violation_rate": (total_violations / total_cases) if total_cases else float("nan"),
        "total_violations": total_violations,
        "total_cases": total_cases,
        "by_test": per_test,
    }


def violation_mask(
        relation: MetamorphicRelation,
        base_pred: torch.Tensor,
        transformed_pred: torch.Tensor,
        atol: float,
        rtol: float,
) -> torch.Tensor:
    """
    Returns a boolean tensor where True means the relation is violated.

    Tolerance semantics:
    - non-strict relations (>=, <=): allow a tolerance band
    - strict relations (>, <): require exceeding the threshold by the tolerance band
    """

    def tol(reference: torch.Tensor):
        return atol + rtol * torch.abs(reference)

    def to_relation_space(prediction: torch.Tensor) -> torch.Tensor:
        raw_out_min = getattr(relation, "raw_out_min", None)
        raw_out_max = getattr(relation, "raw_out_max", None)
        if raw_out_min is None or raw_out_max is None or not raw_out_max > raw_out_min:
            return prediction
        span = float(raw_out_max) - float(raw_out_min)
        return prediction * span + float(raw_out_min)

    if isinstance(relation, Equal):
        diff = torch.abs(transformed_pred - base_pred)
        return diff > tol(base_pred)

    if isinstance(relation, Greater):
        threshold = base_pred + getattr(relation, "margin", 0.0)
        # Strict: require transformed > threshold + tol
        return transformed_pred <= (threshold + tol(threshold))

    if isinstance(relation, GreaterOrEqual):
        threshold = base_pred + getattr(relation, "margin", 0.0)
        # Non-strict: allow transformed >= threshold - tol
        return transformed_pred < (threshold - tol(threshold))

    if isinstance(relation, Lower):
        threshold = base_pred - getattr(relation, "margin", 0.0)
        # Strict: require transformed < threshold - tol
        return transformed_pred >= (threshold - tol(threshold))

    if isinstance(relation, LowerOrEqual):
        threshold = base_pred - getattr(relation, "margin", 0.0)
        # Non-strict: allow transformed <= threshold + tol
        return transformed_pred > (threshold + tol(threshold))

    if isinstance(relation, Monotonic):
        direction = getattr(relation, "direction", "increasing")
        margin = getattr(relation, "margin", 0.0)
        if direction == "increasing":
            threshold = base_pred + margin
            return transformed_pred < (threshold - tol(threshold))
        threshold = base_pred - margin
        return transformed_pred > (threshold + tol(threshold))

    if isinstance(relation, Proportional):
        base_relation = to_relation_space(base_pred)
        transformed_relation = to_relation_space(transformed_pred)
        expected = base_relation * getattr(relation, "factor", 1.0)
        diff = torch.abs(transformed_relation - expected)
        return diff > tol(expected)

    raise TypeError(f"Unsupported relation type for violation check: {type(relation).__name__}")


def evaluate_worst_case_over_T(
        model,
        data_loader,
        transform_set: TransformSet | None,
        tolerance: float | None = None,
        loss_fn: nn.Module | None = None,
) -> dict:
    """
    Worst-case robustness over T:
      sample-wise worst transformed loss/error aggregated over the dataset.
    """
    loss_fn = loss_fn or nn.MSELoss()
    if transform_set is None or len(transform_set) == 0:
        return {
            "available": False,
            "reason": "empty_transform_set",
            "num_transforms": 0,
        }

    model.eval()
    total = 0
    sum_worst_abs = 0.0
    sum_worst_sq = 0.0
    sum_worst_loss = 0.0
    acc_worst_hits = 0.0
    transform_worst_counter: dict[str, int] = {}

    with torch.no_grad():
        for batch in data_loader:
            target = batch["out"].reshape(-1)
            per_transform_abs = []
            per_transform_sq = []
            per_transform_loss = []
            names = []

            for idx, transform_spec in enumerate(transform_set):
                transformed_batch = transform_spec.transform(clone_batch(batch))
                transformed_pred = model(transformed_batch).squeeze().reshape(-1)
                transformed_target = _apply_target_transform(transform_spec, target, batch, transformed_batch).reshape(
                    -1)

                abs_err = torch.abs(transformed_pred - transformed_target)
                sq_err = (transformed_pred - transformed_target) ** 2
                per_transform_abs.append(abs_err)
                per_transform_sq.append(sq_err)
                per_transform_loss.append(loss_fn(transformed_pred, transformed_target))
                names.append(transform_spec.name or f"transform_{idx}")

            abs_stack = torch.stack(per_transform_abs, dim=0)  # [T, B]
            sq_stack = torch.stack(per_transform_sq, dim=0)  # [T, B]
            loss_stack = torch.stack(per_transform_loss, dim=0)  # [T]

            worst_abs, worst_idx_per_sample = torch.max(abs_stack, dim=0)
            worst_sq, _ = torch.max(sq_stack, dim=0)
            worst_loss, worst_loss_idx = torch.max(loss_stack, dim=0)

            batch_n = int(target.numel())
            total += batch_n
            sum_worst_abs += float(worst_abs.sum().item())
            sum_worst_sq += float(worst_sq.sum().item())
            sum_worst_loss += float(worst_loss.item()) * batch_n

            if tolerance is not None:
                acc_worst_hits += float((worst_abs <= tolerance).float().sum().item())

            # Track which transform wins most often at sample level.
            unique_ids, counts = torch.unique(worst_idx_per_sample, return_counts=True)
            for t_idx_tensor, count_tensor in zip(unique_ids, counts):
                name = names[int(t_idx_tensor.item())]
                transform_worst_counter[name] = transform_worst_counter.get(name, 0) + int(count_tensor.item())

            # Also count batch-level worst loss winner for reference.
            batch_worst_name = names[int(worst_loss_idx.item())]
            batch_key = f"batch_loss_winner::{batch_worst_name}"
            transform_worst_counter[batch_key] = transform_worst_counter.get(batch_key, 0) + 1

    if total == 0:
        return {
            "available": False,
            "reason": "empty_dataset",
            "num_transforms": len(transform_set),
        }

    report = {
        "available": True,
        "num_transforms": len(transform_set),
        "worst_case_mae_over_T": sum_worst_abs / total,
        "worst_case_rmse_over_T": (sum_worst_sq / total) ** 0.5,
        "worst_case_loss_over_T": sum_worst_loss / total,
        "worst_transform_frequency": transform_worst_counter,
    }
    if tolerance is not None:
        report["worst_case_acc@tol_over_T"] = acc_worst_hits / total
    return report


def compute_over_T_violation_report(
        model,
        data_loader,
        transform_set: TransformSet | None,
        tolerance: float | None = None,
        atol: float = 1e-6,
        rtol: float = 1e-4,
) -> dict:
    """
    Violation report for over-T transforms.

    A sample violates transform t when:
      |f(t(x)) - y_t| > threshold
    where y_t is target transformed by t.target_transform (or y if None).
    """
    if transform_set is None or len(transform_set) == 0:
        return {
            "available": False,
            "reason": "empty_transform_set",
            "num_transforms": 0,
            "overall_violation_rate": float("nan"),
            "total_violations": 0,
            "total_cases": 0,
            "by_transform": {},
        }

    model.eval()
    per_transform: dict[str, dict[str, float | int]] = {}

    with torch.no_grad():
        for batch in data_loader:
            target = batch["out"].reshape(-1)
            for idx, transform_spec in enumerate(transform_set):
                transform_name = getattr(transform_spec, "name", None) or f"transform_{idx}"
                transformed_batch = transform_spec.transform(clone_batch(batch))
                transformed_pred = model(transformed_batch).squeeze().reshape(-1)
                transformed_target = _apply_target_transform(transform_spec, target, batch, transformed_batch).reshape(
                    -1
                )

                abs_err = torch.abs(transformed_pred - transformed_target)
                if tolerance is not None:
                    threshold = torch.full_like(abs_err, float(tolerance))
                else:
                    threshold = atol + rtol * torch.abs(transformed_target)
                violations = abs_err > threshold

                if transform_name not in per_transform:
                    per_transform[transform_name] = {"violations": 0, "total": 0}
                per_transform[transform_name]["violations"] += int(violations.sum().item())
                per_transform[transform_name]["total"] += int(violations.numel())

    total_violations = 0
    total_cases = 0
    for transform_name, stats in per_transform.items():
        violations = int(stats["violations"])
        total = int(stats["total"])
        stats["violation_rate"] = (violations / total) if total else float("nan")
        total_violations += violations
        total_cases += total

    return {
        "available": True,
        "num_transforms": len(transform_set),
        "overall_violation_rate": (total_violations / total_cases) if total_cases else float("nan"),
        "total_violations": total_violations,
        "total_cases": total_cases,
        "by_transform": per_transform,
    }


def validate_metamorphic_transforms_on_batch(
        batch: Batch,
        relation_tests: Iterable[MetamorphicTest] | None = None,
        transform_set: TransformSet | Iterable | None = None,
) -> dict:
    """
    Heuristic validation for structural and physical consistency of follow-up inputs.
    Returns errors/warnings; does not raise.
    """
    relation_tests = list(relation_tests or [])
    if transform_set is None:
        transform_specs = []
    elif isinstance(transform_set, TransformSet):
        transform_specs = list(transform_set)
    else:
        transform_specs = list(transform_set)

    errors: list[str] = []
    warnings: list[str] = []
    checked = 0

    collections = []
    for i, test in enumerate(relation_tests):
        collections.append((test.name or f"relation_{i}", test.transform))
    for i, t in enumerate(transform_specs):
        collections.append((getattr(t, "name", None) or f"transform_{i}", t.transform))

    for name, transform in collections:
        checked += 1
        try:
            transformed = transform(clone_batch(batch))
        except Exception as exc:
            errors.append(f"{name}: transform raised {type(exc).__name__}: {exc}")
            continue

        _validate_batch_structure(batch, transformed, name, errors, warnings)
        _validate_heuristic_time_series_consistency(batch, transformed, name, warnings)

    return {
        "checked_transforms": checked,
        "errors": errors,
        "warnings": warnings,
        "is_valid": len(errors) == 0,
    }


def _validate_batch_structure(source: Batch, transformed: Batch, name: str, errors: list[str], warnings: list[str]):
    source_keys = set(source.keys())
    transformed_keys = set(transformed.keys())
    if source_keys != transformed_keys:
        errors.append(f"{name}: key set changed ({sorted(source_keys)} -> {sorted(transformed_keys)})")
        return

    for key in sorted(source_keys):
        src = source[key]
        dst = transformed[key]
        if torch.is_tensor(src) != torch.is_tensor(dst):
            errors.append(f"{name}: key '{key}' tensor/non-tensor type changed")
            continue
        if not torch.is_tensor(src):
            continue
        if src.shape != dst.shape:
            errors.append(f"{name}: key '{key}' shape changed {tuple(src.shape)} -> {tuple(dst.shape)}")
        if src.dtype != dst.dtype:
            errors.append(f"{name}: key '{key}' dtype changed {src.dtype} -> {dst.dtype}")
        if src.device != dst.device:
            errors.append(f"{name}: key '{key}' device changed {src.device} -> {dst.device}")
        if not torch.isfinite(dst).all():
            errors.append(f"{name}: key '{key}' contains non-finite values")

        if key == "categorical_t_features" and torch.is_tensor(dst):
            # Heuristic: categorical features are expected to be small integers/one-hots in this project.
            if torch.any(torch.abs(dst - torch.round(dst)) > 1e-5):
                warnings.append(f"{name}: categorical_t_features became non-integer-like")


def _validate_heuristic_time_series_consistency(source: Batch, transformed: Batch, name: str, warnings: list[str]):
    # If current timestamp encoding changed but lookback timestamps didn't, warn.
    if "t" in source and "lookback_t" in source and torch.is_tensor(source["t"]) and torch.is_tensor(
            source["lookback_t"]):
        if source["lookback_t"].numel() > 0:
            t_changed = not torch.equal(source["t"], transformed["t"])
            lookback_t_changed = not torch.equal(source["lookback_t"], transformed["lookback_t"])
            if t_changed and not lookback_t_changed:
                warnings.append(f"{name}: changed 't' without changing 'lookback_t' (possible temporal inconsistency)")

    # If current numerical features changed but lookback numerical features did not, warn.
    if (
            "numerical_t_features" in source
            and "numerical_lookback_features" in source
            and torch.is_tensor(source["numerical_t_features"])
            and torch.is_tensor(source["numerical_lookback_features"])
    ):
        if source["numerical_lookback_features"].numel() > 0:
            num_t_changed = not torch.equal(source["numerical_t_features"], transformed["numerical_t_features"])
            num_lb_changed = not torch.equal(source["numerical_lookback_features"],
                                             transformed["numerical_lookback_features"])
            if num_t_changed and not num_lb_changed:
                warnings.append(
                    f"{name}: changed current numerical features without changing lookback numerical features "
                    f"(possible history inconsistency)"
                )
