from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable

import torch
from torch import nn

Batch = dict[str, torch.Tensor]
BatchTransform = Callable[[Batch], Batch]
# Supported signatures:
#   target_transform(target)
#   target_transform(target, source_batch, transformed_batch)
TargetTransform = Callable[..., torch.Tensor]


class MetamorphicRelationKind(str, Enum):
    EQUAL = "Equal"
    MONOTONIC = "Monotonic"
    GREATER = "Greater"
    LOWER = "Lower"
    GREATER_OR_EQUAL = "GreaterOrEqual"
    LOWER_OR_EQUAL = "LowerOrEqual"
    PROPORTIONAL = "Proportional"


class MetamorphicRelation(ABC):
    def __init__(self, weight: float = 1.0):
        if weight < 0:
            raise ValueError("weight must be >= 0")
        self.weight = weight

    @property
    @abstractmethod
    def kind(self) -> MetamorphicRelationKind:
        raise NotImplementedError

    @abstractmethod
    def penalty(self, base_prediction: torch.Tensor, transformed_prediction: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Equal(MetamorphicRelation):
    @property
    def kind(self) -> MetamorphicRelationKind:
        return MetamorphicRelationKind.EQUAL

    def penalty(self, base_prediction: torch.Tensor, transformed_prediction: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(transformed_prediction - base_prediction))


class Greater(MetamorphicRelation):
    def __init__(self, margin: float = 0.0, weight: float = 1.0):
        super().__init__(weight=weight)
        self.margin = margin

    @property
    def kind(self) -> MetamorphicRelationKind:
        return MetamorphicRelationKind.GREATER

    def penalty(self, base_prediction: torch.Tensor, transformed_prediction: torch.Tensor) -> torch.Tensor:
        # Enforce transformed > base + margin (strictness is approximated with hinge + margin)
        return torch.relu((base_prediction + self.margin) - transformed_prediction).mean()


class Lower(MetamorphicRelation):
    def __init__(self, margin: float = 0.0, weight: float = 1.0):
        super().__init__(weight=weight)
        self.margin = margin

    @property
    def kind(self) -> MetamorphicRelationKind:
        return MetamorphicRelationKind.LOWER

    def penalty(self, base_prediction: torch.Tensor, transformed_prediction: torch.Tensor) -> torch.Tensor:
        # Enforce transformed < base - margin (strictness is approximated with hinge + margin)
        return torch.relu(transformed_prediction - (base_prediction - self.margin)).mean()


class GreaterOrEqual(MetamorphicRelation):
    def __init__(self, margin: float = 0.0, weight: float = 1.0):
        super().__init__(weight=weight)
        self.margin = margin

    @property
    def kind(self) -> MetamorphicRelationKind:
        return MetamorphicRelationKind.GREATER_OR_EQUAL

    def penalty(self, base_prediction: torch.Tensor, transformed_prediction: torch.Tensor) -> torch.Tensor:
        return torch.relu((base_prediction + self.margin) - transformed_prediction).mean()


class LowerOrEqual(MetamorphicRelation):
    def __init__(self, margin: float = 0.0, weight: float = 1.0):
        super().__init__(weight=weight)
        self.margin = margin

    @property
    def kind(self) -> MetamorphicRelationKind:
        return MetamorphicRelationKind.LOWER_OR_EQUAL

    def penalty(self, base_prediction: torch.Tensor, transformed_prediction: torch.Tensor) -> torch.Tensor:
        return torch.relu(transformed_prediction - (base_prediction - self.margin)).mean()


class Monotonic(MetamorphicRelation):
    def __init__(self, direction: str = "increasing", margin: float = 0.0, weight: float = 1.0):
        super().__init__(weight=weight)
        if direction not in ("increasing", "decreasing"):
            raise ValueError("direction must be 'increasing' or 'decreasing'")
        self.direction = direction
        self.margin = margin

    @property
    def kind(self) -> MetamorphicRelationKind:
        return MetamorphicRelationKind.MONOTONIC

    def penalty(self, base_prediction: torch.Tensor, transformed_prediction: torch.Tensor) -> torch.Tensor:
        if self.direction == "increasing":
            return torch.relu((base_prediction + self.margin) - transformed_prediction).mean()
        return torch.relu(transformed_prediction - (base_prediction - self.margin)).mean()


class Proportional(MetamorphicRelation):
    def __init__(
            self,
            factor: float,
            weight: float = 1.0,
            eps: float = 1e-8,
            raw_out_min: float | None = None,
            raw_out_max: float | None = None,
    ):
        super().__init__(weight=weight)
        self.factor = factor
        self.eps = eps
        self.raw_out_min = float(raw_out_min) if raw_out_min is not None else None
        self.raw_out_max = float(raw_out_max) if raw_out_max is not None else None

    @property
    def kind(self) -> MetamorphicRelationKind:
        return MetamorphicRelationKind.PROPORTIONAL

    def _to_relation_space(self, prediction: torch.Tensor) -> torch.Tensor:
        if (
                self.raw_out_min is None
                or self.raw_out_max is None
                or not self.raw_out_max > self.raw_out_min
        ):
            return prediction
        span = self.raw_out_max - self.raw_out_min
        return prediction * span + self.raw_out_min

    def penalty(self, base_prediction: torch.Tensor, transformed_prediction: torch.Tensor) -> torch.Tensor:
        base_relation = self._to_relation_space(base_prediction)
        transformed_relation = self._to_relation_space(transformed_prediction)
        expected = base_relation * self.factor
        scale = torch.clamp(torch.abs(expected), min=self.eps)
        return torch.mean(torch.abs(transformed_relation - expected) / scale)


@dataclass(frozen=True)
class MetamorphicTransform:
    """
    Transformation element t in T for the worst-case-over-T objective.

    `target_transform` allows label-preserving (None) or label-mapped MRs.
    """

    transform: BatchTransform
    target_transform: TargetTransform | None = None
    name: str | None = None
    weight: float = 1.0

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("MetamorphicTransform.weight must be >= 0")


class TransformSet:
    """
    Explicit container for the set T of metamorphic transformations (over-T transforms).
    """

    def __init__(self, transforms: Iterable[MetamorphicTransform] | None = None):
        self.transforms = tuple(transforms or ())

    def __iter__(self):
        return iter(self.transforms)

    def __len__(self) -> int:
        return len(self.transforms)

    def __bool__(self) -> bool:
        return len(self.transforms) > 0


@dataclass(frozen=True)
class MetamorphicTest:
    """
    Relation-based constraint (custom extension).

    It can optionally carry `target_transform` for frameworks that support target-mapped
    training objectives, even if the relation penalty itself does not use it directly.
    """

    relation: MetamorphicRelation
    transform: BatchTransform
    name: str | None = None
    target_transform: TargetTransform | None = None
    violation_atol: float | None = None
    violation_rtol: float | None = None

    def __post_init__(self):
        if self.target_transform is not None:
            raise ValueError(
                "Relation constraints cannot define target_transform. "
                "Use a target-mapped over_T MetamorphicTransform instead."
            )


def _normalize_rule_category(category) -> str | None:
    if category is None:
        return None
    value = getattr(category, "value", category)
    if value is None:
        return None
    return str(value).strip().lower()


def partition_rule_specs_exclusive(rule_specs: Iterable[object] | None) -> tuple[
    list[MetamorphicTest], TransformSet, dict]:
    """
    Partition catalog rule specs into relation-constraint vs worst-case-over-T branches.

    Rule specs are expected to be exclusive by construction (exactly one artifact):
    - relation_test
    - over_T_transform
    """
    relation_tests: list[MetamorphicTest] = []
    over_T_transforms: list[MetamorphicTransform] = []
    summary = {
        "num_rule_specs": 0,
        "assigned_relation_constraints": 0,
        "assigned_over_T_transforms": 0,
        "dropped_rule_specs": 0,
        "fallback_to_relation": 0,
        "fallback_to_over_T": 0,
        "unknown_category_defaults": 0,
        "by_category": {},
    }
    for spec in list(rule_specs or ()):
        summary["num_rule_specs"] += 1
        name = getattr(spec, "name", None)
        category_name = _normalize_rule_category(getattr(spec, "category", None)) or "unknown"
        summary["by_category"][category_name] = summary["by_category"].get(category_name, 0) + 1

        relation_test = getattr(spec, "relation_test", None)
        over_T_transform = getattr(spec, "over_T_transform", None)
        has_relation = relation_test is not None
        has_over_t = over_T_transform is not None
        if has_relation == has_over_t:
            summary["dropped_rule_specs"] += 1
            continue

        if has_relation:
            relation_tests.append(relation_test)
            summary["assigned_relation_constraints"] += 1
        else:
            over_T_transforms.append(over_T_transform)
            summary["assigned_over_T_transforms"] += 1

        # Keep linter quiet for debugging expansions where `name` is useful.
        _ = name

    return relation_tests, TransformSet(over_T_transforms), summary


class CompositeMetamorphicLoss(nn.Module):
    """
    Unified metamorphic loss.

    It can combine:
    - supervised source loss               : l(f(x), y)
    - worst-case-over-T transformed loss   : max_t l(f(t(x)), y_t)
    - relation constraint penalties        : Agg_i penalty_i(f(x), f(t_i(x)))

    Use weights to enable/disable each term.
    """

    def __init__(
            self,
            supervised_loss: nn.Module | None = None,
            metamorphic_tests: list[MetamorphicTest] | None = None,
            transform_set: TransformSet | Iterable[MetamorphicTransform] | None = None,
            rule_specs: Iterable[object] | None = None,
            supervised_weight: float = 1.0,
            relation_constraint_weight: float = 0.0,
            worst_case_over_T_weight: float = 0.0,
            target_mapped_weight: float = 0.0,
            relation_aggregation: str = "mean",
            target_mapped_aggregation: str = "mean",
    ):
        super().__init__()
        if min(supervised_weight, relation_constraint_weight, worst_case_over_T_weight, target_mapped_weight) < 0:
            raise ValueError("weights must be >= 0")
        if relation_aggregation not in ("mean", "max"):
            raise ValueError("relation_aggregation must be 'mean' or 'max'")
        if target_mapped_aggregation not in ("mean", "max"):
            raise ValueError("target_mapped_aggregation must be 'mean' or 'max'")
        if target_mapped_weight != 0:
            raise ValueError(
                "target_mapped_weight is no longer supported for relation constraints. "
                "Encode target-mapped rules as over_T transforms and use worst_case_over_T_weight."
            )

        if rule_specs is not None and (metamorphic_tests is not None or transform_set is not None):
            raise ValueError("Use either rule_specs or (metamorphic_tests/transform_set), not both")

        self.supervised_loss = supervised_loss or nn.MSELoss()
        if rule_specs is not None:
            assigned_relation_tests, assigned_transform_set, assignment_summary = partition_rule_specs_exclusive(
                rule_specs)
            self.relation_constraints = assigned_relation_tests
            self.over_T_transform_set = assigned_transform_set
            self.rule_assignment_summary = assignment_summary
        else:
            self.relation_constraints = list(metamorphic_tests or [])
            self.over_T_transform_set = transform_set if isinstance(transform_set, TransformSet) else TransformSet(
                transform_set)
            self.rule_assignment_summary = {
                "num_rule_specs": 0,
                "assigned_relation_constraints": len(self.relation_constraints),
                "assigned_over_T_transforms": len(self.over_T_transform_set),
                "dropped_rule_specs": 0,
                "fallback_to_relation": 0,
                "fallback_to_over_T": 0,
                "unknown_category_defaults": 0,
                "by_category": {},
            }

        # Explicit aliases to make external use (evaluation/reporting) transparent.
        self.assigned_relation_constraints = self.relation_constraints
        self.assigned_over_T_transform_set = self.over_T_transform_set
        self.supervised_weight = supervised_weight
        self.relation_constraint_weight = relation_constraint_weight
        self.worst_case_over_T_weight = worst_case_over_T_weight
        self.target_mapped_weight = 0.0
        self.relation_aggregation = relation_aggregation
        self.target_mapped_aggregation = target_mapped_aggregation
        self.last_metrics: dict[str, float | str | None] | None = None

    @classmethod
    def from_rule_specs(
            cls,
            rule_specs: Iterable[object],
            **kwargs,
    ) -> "CompositeMetamorphicLoss":
        return cls(rule_specs=rule_specs, **kwargs)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Backward-compatible behavior when used as a plain loss(pred, target).
        return self.supervised_loss(prediction, target)

    def compute_training_loss(
            self,
            model: nn.Module,
            batch: Batch,
            target: torch.Tensor,
            prediction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prediction is None:
            prediction = model(batch).squeeze()

        supervised = self.supervised_loss(prediction, target)

        relation_constraint_penalty, relation_worst_name = self._compute_relation_constraint_penalty(
            model=model,
            batch=batch,
            target=target,
            prediction=prediction,
        )
        worst_case_over_T_loss, worst_over_T_name = self._compute_worst_case_over_T_loss(
            model=model,
            batch=batch,
            target=target,
        )
        supervised_contrib = self.supervised_weight * supervised
        relation_constraint_contrib = self.relation_constraint_weight * relation_constraint_penalty
        worst_case_over_T_contrib = self.worst_case_over_T_weight * worst_case_over_T_loss
        target_mapped_contrib = supervised.new_tensor(0.0)
        total = supervised_contrib + relation_constraint_contrib + worst_case_over_T_contrib

        enabled_relation_constraints = bool(self.relation_constraints and (self.relation_constraint_weight > 0))
        enabled_over_T = bool(self.over_T_transform_set and (self.worst_case_over_T_weight > 0))
        if enabled_relation_constraints and enabled_over_T:
            loss_type = "composite"
        elif enabled_over_T:
            loss_type = "worst_case_over_T"
        elif enabled_relation_constraints:
            loss_type = "relation_constraints"
        else:
            loss_type = "supervised"

        self.last_metrics = {
            "loss_type": loss_type,
            "total_loss": float(total.detach().item()),
            "supervised_loss": float(supervised_contrib.detach().item()),
            "relation_constraint_penalty": float(relation_constraint_contrib.detach().item()),
            "worst_case_over_T_loss": float(worst_case_over_T_contrib.detach().item()),
            "target_mapped_supervised_loss": 0.0,
            "raw_supervised_loss": float(supervised.detach().item()),
            "raw_relation_constraint_penalty": float(relation_constraint_penalty.detach().item()),
            "raw_worst_case_over_T_loss": float(worst_case_over_T_loss.detach().item()),
            "raw_target_mapped_supervised_loss": 0.0,
            "worst_transform_name": worst_over_T_name or relation_worst_name,
            "worst_relation_constraint_name": relation_worst_name,
            "worst_over_T_transform_name": worst_over_T_name,
            "num_relation_constraints": float(len(self.relation_constraints)),
            "num_over_T_transforms": float(len(self.over_T_transform_set)),
            "num_target_mapped_terms": 0.0,
            "num_rule_specs": float(self.rule_assignment_summary.get("num_rule_specs", 0)),
            "dropped_rule_specs": float(self.rule_assignment_summary.get("dropped_rule_specs", 0)),
        }
        return total

    def _compute_relation_constraint_penalty(
            self,
            model: nn.Module,
            batch: Batch,
            target: torch.Tensor,
            prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, str | None]:
        _ = target  # kept for symmetry/future extensions
        if not self.relation_constraints:
            zero = prediction.new_tensor(0.0)
            return zero, None
        if self.relation_constraint_weight == 0:
            zero = prediction.new_tensor(0.0)
            return zero, None

        penalties = []
        names: list[str | None] = []
        for test in self.relation_constraints:
            transformed_batch = test.transform(_clone_batch(batch))
            transformed_prediction = model(transformed_batch).squeeze()
            penalties.append(test.relation.weight * test.relation.penalty(prediction, transformed_prediction))
            names.append(test.name)

        penalties_tensor = torch.stack(penalties) if penalties else prediction.new_zeros(0)
        if not penalties:
            return prediction.new_tensor(0.0), None
        if self.relation_aggregation == "max":
            relation_constraint_penalty, worst_idx_tensor = torch.max(penalties_tensor, dim=0)
            worst_idx = int(worst_idx_tensor.item()) if worst_idx_tensor.dim() == 0 else int(
                worst_idx_tensor.reshape(-1)[0])
        else:
            relation_constraint_penalty = torch.mean(penalties_tensor)
            worst_idx = int(torch.argmax(penalties_tensor).item())
        return relation_constraint_penalty, names[worst_idx]

    def _compute_worst_case_over_T_loss(
            self,
            model: nn.Module,
            batch: Batch,
            target: torch.Tensor,
    ) -> tuple[torch.Tensor, str | None]:
        if not self.over_T_transform_set or self.worst_case_over_T_weight == 0:
            if torch.is_tensor(target):
                return target.new_tensor(0.0), None
            return torch.tensor(0.0), None

        transformed_losses = []
        transform_names: list[str | None] = []
        for transform_spec in self.over_T_transform_set:
            transformed_batch = transform_spec.transform(_clone_batch(batch))
            transformed_prediction = model(transformed_batch).squeeze()
            transformed_target = _apply_target_transform(
                transform_spec.target_transform,
                target=target,
                source_batch=batch,
                transformed_batch=transformed_batch,
            )
            transformed_loss = self.supervised_loss(transformed_prediction, transformed_target)
            transform_weight = float(getattr(transform_spec, "weight", 1.0))
            transformed_losses.append(transform_weight * transformed_loss)
            transform_names.append(transform_spec.name)

        if not transformed_losses:
            return target.new_tensor(0.0), None
        transformed_losses_tensor = torch.stack(transformed_losses)
        worst_loss, worst_idx_tensor = torch.max(transformed_losses_tensor, dim=0)
        worst_idx = int(worst_idx_tensor.item()) if worst_idx_tensor.dim() == 0 else int(
            worst_idx_tensor.reshape(-1)[0])
        return worst_loss, transform_names[worst_idx]


def _clone_batch(batch: Batch) -> Batch:
    cloned: Batch = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = deepcopy(value)
    return cloned


def _clone_target(target: torch.Tensor) -> torch.Tensor:
    return target.clone() if torch.is_tensor(target) else deepcopy(target)


def _apply_target_transform(
        target_transform: TargetTransform | None,
        target: torch.Tensor,
        source_batch: Batch,
        transformed_batch: Batch,
) -> torch.Tensor:
    if target_transform is None:
        return target
    cloned_target = _clone_target(target)
    try:
        return target_transform(cloned_target, source_batch, transformed_batch)
    except TypeError:
        return target_transform(cloned_target)
