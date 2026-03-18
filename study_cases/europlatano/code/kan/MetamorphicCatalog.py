from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch

from kan.MetamorphicLoss import (
    Batch,
    BatchTransform,
    Equal,
    Greater,
    GreaterOrEqual,
    Lower,
    LowerOrEqual,
    MetamorphicTransform,
    MetamorphicTest,
    Monotonic,
    Proportional,
    TargetTransform,
    TransformSet,
)


def compose(*transforms: BatchTransform) -> BatchTransform:
    def _transform(batch: Batch) -> Batch:
        for transform in transforms:
            batch = transform(batch)
        return batch

    return _transform


def identity() -> BatchTransform:
    return lambda batch: batch


def add_scalar(field: str, delta: float) -> BatchTransform:
    def _transform(batch: Batch) -> Batch:
        batch[field] = batch[field] + delta
        return batch

    return _transform


def scale_field(field: str, factor: float) -> BatchTransform:
    def _transform(batch: Batch) -> Batch:
        batch[field] = batch[field] * factor
        return batch

    return _transform


def add_to_t(index: int, delta: float) -> BatchTransform:
    def _transform(batch: Batch) -> Batch:
        batch["t"] = _add_at_last_dim(batch["t"], index, delta)
        return batch

    return _transform


def add_to_lookback_t(index: int, delta: float, step: int | None = None) -> BatchTransform:
    def _transform(batch: Batch) -> Batch:
        tensor = batch["lookback_t"]
        if tensor.dim() == 2:
            # shape [B, F] when lookback=1 collapsed by collate
            batch["lookback_t"] = _add_at_last_dim(tensor, index, delta)
            return batch
        if step is None:
            tensor[..., index] = tensor[..., index] + delta
        else:
            tensor[:, step, index] = tensor[:, step, index] + delta
        batch["lookback_t"] = tensor
        return batch

    return _transform


def add_to_numerical_t_feature(index: int, delta: float) -> BatchTransform:
    def _transform(batch: Batch) -> Batch:
        batch["numerical_t_features"] = _add_at_last_dim(batch["numerical_t_features"], index, delta)
        return batch

    return _transform


def scale_numerical_t_feature(index: int, factor: float) -> BatchTransform:
    def _transform(batch: Batch) -> Batch:
        batch["numerical_t_features"] = _mul_at_last_dim(batch["numerical_t_features"], index, factor)
        return batch

    return _transform


def add_to_numerical_lookback_feature(index: int, delta: float, step: int | None = None) -> BatchTransform:
    def _transform(batch: Batch) -> Batch:
        tensor = batch["numerical_lookback_features"]
        if tensor.dim() == 2:
            batch["numerical_lookback_features"] = _add_at_last_dim(tensor, index, delta)
            return batch
        if step is None:
            tensor[..., index] = tensor[..., index] + delta
        else:
            tensor[:, step, index] = tensor[:, step, index] + delta
        batch["numerical_lookback_features"] = tensor
        return batch

    return _transform


def scale_numerical_lookback_feature(index: int, factor: float, step: int | None = None) -> BatchTransform:
    def _transform(batch: Batch) -> Batch:
        tensor = batch["numerical_lookback_features"]
        if tensor.dim() == 2:
            batch["numerical_lookback_features"] = _mul_at_last_dim(tensor, index, factor)
            return batch
        if step is None:
            tensor[..., index] = tensor[..., index] * factor
        else:
            tensor[:, step, index] = tensor[:, step, index] * factor
        batch["numerical_lookback_features"] = tensor
        return batch

    return _transform


def zero_categorical_t_features() -> BatchTransform:
    def _transform(batch: Batch) -> Batch:
        batch["categorical_t_features"] = torch.zeros_like(batch["categorical_t_features"])
        return batch

    return _transform


def zero_categorical_lookback_features() -> BatchTransform:
    def _transform(batch: Batch) -> Batch:
        batch["categorical_lookback_features"] = torch.zeros_like(batch["categorical_lookback_features"])
        return batch

    return _transform


def shift_target(delta: float) -> TargetTransform:
    def _transform(target: torch.Tensor, *_args) -> torch.Tensor:
        return target + delta

    return _transform


def scale_target(factor: float) -> TargetTransform:
    def _transform(target: torch.Tensor, *_args) -> torch.Tensor:
        return target * factor

    return _transform


def coupled_shift_numerical_feature(index: int, delta: float, lookback_step: int | None = None) -> BatchTransform:
    """
    Shift the same numerical feature in both current step and lookback to reduce
    physically inconsistent follow-ups for time-series data.
    """
    return compose(
        add_to_numerical_t_feature(index, delta),
        add_to_numerical_lookback_feature(index, delta, step=lookback_step),
    )


def coupled_scale_numerical_feature(index: int, factor: float, lookback_step: int | None = None) -> BatchTransform:
    return compose(
        scale_numerical_t_feature(index, factor),
        scale_numerical_lookback_feature(index, factor, step=lookback_step),
    )


def coupled_add_to_time_encoding(index: int, delta: float, lookback_step: int | None = None) -> BatchTransform:
    return compose(
        add_to_t(index, delta),
        add_to_lookback_t(index, delta, step=lookback_step),
    )


def make_transform(
        transform: BatchTransform,
        name: str | None = None,
        target_transform: TargetTransform | None = None,
        weight: float = 1.0,
) -> MetamorphicTransform:
    return MetamorphicTransform(transform=transform, target_transform=target_transform, name=name, weight=weight)


def make_transform_set(transforms: Iterable[MetamorphicTransform]) -> TransformSet:
    return TransformSet(transforms)


def make_equal_test(
        transform: BatchTransform,
        name: str | None = None,
        weight: float = 1.0,
        target_transform: TargetTransform | None = None,
        violation_atol: float | None = None,
        violation_rtol: float | None = None,
) -> MetamorphicTest:
    return MetamorphicTest(
        relation=Equal(weight=weight),
        transform=transform,
        name=name,
        target_transform=target_transform,
        violation_atol=violation_atol,
        violation_rtol=violation_rtol,
    )


def make_monotonic_test(
        transform: BatchTransform,
        direction: str = "increasing",
        margin: float = 0.0,
        name: str | None = None,
        weight: float = 1.0,
        target_transform: TargetTransform | None = None,
        violation_atol: float | None = None,
        violation_rtol: float | None = None,
) -> MetamorphicTest:
    return MetamorphicTest(
        relation=Monotonic(direction=direction, margin=margin, weight=weight),
        transform=transform,
        name=name,
        target_transform=target_transform,
        violation_atol=violation_atol,
        violation_rtol=violation_rtol,
    )


def make_greater_test(transform: BatchTransform, margin: float = 0.0, name: str | None = None,
                      weight: float = 1.0,
                      target_transform: TargetTransform | None = None,
                      violation_atol: float | None = None,
                      violation_rtol: float | None = None) -> MetamorphicTest:
    return MetamorphicTest(
        relation=Greater(margin=margin, weight=weight),
        transform=transform,
        name=name,
        target_transform=target_transform,
        violation_atol=violation_atol,
        violation_rtol=violation_rtol,
    )


def make_lower_test(transform: BatchTransform, margin: float = 0.0, name: str | None = None,
                    weight: float = 1.0,
                    target_transform: TargetTransform | None = None,
                    violation_atol: float | None = None,
                    violation_rtol: float | None = None) -> MetamorphicTest:
    return MetamorphicTest(
        relation=Lower(margin=margin, weight=weight),
        transform=transform,
        name=name,
        target_transform=target_transform,
        violation_atol=violation_atol,
        violation_rtol=violation_rtol,
    )


def make_greater_or_equal_test(transform: BatchTransform, margin: float = 0.0, name: str | None = None,
                               weight: float = 1.0,
                               target_transform: TargetTransform | None = None,
                               violation_atol: float | None = None,
                               violation_rtol: float | None = None) -> MetamorphicTest:
    return MetamorphicTest(
        relation=GreaterOrEqual(margin=margin, weight=weight),
        transform=transform,
        name=name,
        target_transform=target_transform,
        violation_atol=violation_atol,
        violation_rtol=violation_rtol,
    )


def make_lower_or_equal_test(transform: BatchTransform, margin: float = 0.0, name: str | None = None,
                             weight: float = 1.0,
                             target_transform: TargetTransform | None = None,
                             violation_atol: float | None = None,
                             violation_rtol: float | None = None) -> MetamorphicTest:
    return MetamorphicTest(
        relation=LowerOrEqual(margin=margin, weight=weight),
        transform=transform,
        name=name,
        target_transform=target_transform,
        violation_atol=violation_atol,
        violation_rtol=violation_rtol,
    )


def make_proportional_test(transform: BatchTransform, factor: float, name: str | None = None,
                           weight: float = 1.0,
                           target_transform: TargetTransform | None = None,
                           violation_atol: float | None = None,
                           violation_rtol: float | None = None) -> MetamorphicTest:
    return MetamorphicTest(
        relation=Proportional(factor=factor, weight=weight),
        transform=transform,
        name=name,
        target_transform=target_transform,
        violation_atol=violation_atol,
        violation_rtol=violation_rtol,
    )


class RuleCategory(str, Enum):
    INVARIANCE = "invariance"
    DIRECTIONAL_ORDINAL = "directional_ordinal"
    TARGET_MAPPED = "target_mapped"


@dataclass(frozen=True)
class CatalogRuleSpec:
    name: str
    category: RuleCategory
    relation_test: MetamorphicTest | None = None
    over_T_transform: MetamorphicTransform | None = None
    description: str | None = None
    consistency_profile: str | None = None

    def __post_init__(self):
        has_relation = self.relation_test is not None
        has_over_t = self.over_T_transform is not None
        if has_relation == has_over_t:
            raise ValueError(
                "CatalogRuleSpec must define exactly one branch: "
                "either relation_test or over_T_transform."
            )
        if has_relation and getattr(self.relation_test, "target_transform", None) is not None:
            raise ValueError(
                "Relation constraints cannot define target_transform. "
                "Use over_T_transform for target-mapped rules."
            )


def summarize_rule_specs(specs: Iterable[CatalogRuleSpec]) -> dict:
    counts = {category.value: 0 for category in RuleCategory}
    total_relation = 0
    total_over_T = 0
    for spec in specs:
        counts[spec.category.value] += 1
        if spec.relation_test is not None:
            total_relation += 1
        if spec.over_T_transform is not None:
            total_over_T += 1
    return {
        "num_specs": sum(counts.values()),
        "num_relation_tests": total_relation,
        "num_over_T_transforms": total_over_T,
        "by_category": counts,
    }


def _add_at_last_dim(tensor: torch.Tensor, index: int, delta: float) -> torch.Tensor:
    out = tensor.clone()
    out[..., index] = out[..., index] + delta
    return out


def _mul_at_last_dim(tensor: torch.Tensor, index: int, factor: float) -> torch.Tensor:
    out = tensor.clone()
    out[..., index] = out[..., index] * factor
    return out
