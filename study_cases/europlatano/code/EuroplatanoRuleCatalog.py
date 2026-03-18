from __future__ import annotations

from typing import Iterable, Mapping

import torch

from kan.MetamorphicCatalog import (
    CatalogRuleSpec,
    RuleCategory,
    make_equal_test,
    make_greater_or_equal_test,
    make_proportional_test,
    scale_numerical_t_feature,
)


DEFAULT_EUROPLATANO_RULE_WEIGHT_MAP: dict[str, float] = {
    "production_proportional_to_area": 1.0,
    "production_non_decreasing_with_rainfall": 1.0,
    "production_robust_to_humidity_noise": 1.0,
}


def add_gaussian_noise_to_numerical_t_feature(
    index: int,
    stddev: float,
    clamp_min: float | None = 0.0,
    clamp_max: float | None = 1.0,
):
    def _transform(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        numerical = batch["numerical_t_features"].clone()
        noise = torch.randn_like(numerical[..., index]) * float(stddev)
        numerical[..., index] = numerical[..., index] + noise
        if clamp_min is not None or clamp_max is not None:
            numerical[..., index] = torch.clamp(
                numerical[..., index],
                min=clamp_min,
                max=clamp_max,
            )
        batch["numerical_t_features"] = numerical
        return batch

    return _transform


def build_europlatano_production_rule_specs(
    numerical_t_feature_names: Iterable[str],
    rule_weight_map: Mapping[str, float] | None = None,
    raw_output_min: float | None = None,
    raw_output_max: float | None = None,
) -> tuple[list[CatalogRuleSpec], dict[str, float], list[str]]:
    names = list(numerical_t_feature_names)
    weights = dict(DEFAULT_EUROPLATANO_RULE_WEIGHT_MAP)
    if rule_weight_map is not None:
        for key, value in rule_weight_map.items():
            weights[key] = float(value)
    for key, value in weights.items():
        if value < 0:
            raise ValueError(f"Rule weight for '{key}' must be >= 0")

    specs: list[CatalogRuleSpec] = []
    active_rule_names: list[str] = []
    effective_weights: dict[str, float] = {}

    rule_name = "production_proportional_to_area"
    if "Area" in names:
        idx = names.index("Area")
        weight = float(weights.get(rule_name, 1.0))
        active_rule_names.append(rule_name)
        effective_weights[rule_name] = weight
        test = make_proportional_test(
            transform=scale_numerical_t_feature(index=idx, factor=1.20),
            factor=1.20,
            name=rule_name,
            weight=weight,
            violation_atol=0.0,
            violation_rtol=0.17,
        )
        relation = getattr(test, "relation", None)
        if relation is not None and raw_output_min is not None and raw_output_max is not None:
            relation.raw_out_min = float(raw_output_min)
            relation.raw_out_max = float(raw_output_max)
        specs.append(
            CatalogRuleSpec(
                name=rule_name,
                category=RuleCategory.TARGET_MAPPED,
                relation_test=test,
                description="Production should be proportional to farm area",
                consistency_profile="current_numerical_only",
            )
        )

    rule_name = "production_non_decreasing_with_rainfall"
    if "Territory.Precipitation" in names:
        idx = names.index("Territory.Precipitation")
        weight = float(weights.get(rule_name, 1.0))
        active_rule_names.append(rule_name)
        effective_weights[rule_name] = weight
        specs.append(
            CatalogRuleSpec(
                name=rule_name,
                category=RuleCategory.DIRECTIONAL_ORDINAL,
                relation_test=make_greater_or_equal_test(
                    transform=scale_numerical_t_feature(index=idx, factor=1.10),
                    margin=0.0,
                    name=rule_name,
                    weight=weight,
                    violation_atol=0.0,
                    violation_rtol=0.0,
                ),
                description="Production should not decrease as rainfall increases",
                consistency_profile="current_numerical_only",
            )
        )

    rule_name = "production_robust_to_humidity_noise"
    if "Territory.Humidity" in names:
        idx = names.index("Territory.Humidity")
        weight = float(weights.get(rule_name, 1.0))
        active_rule_names.append(rule_name)
        effective_weights[rule_name] = weight
        specs.append(
            CatalogRuleSpec(
                name=rule_name,
                category=RuleCategory.INVARIANCE,
                relation_test=make_equal_test(
                    transform=add_gaussian_noise_to_numerical_t_feature(index=idx, stddev=0.02),
                    name=rule_name,
                    weight=weight,
                    violation_atol=0.0,
                    violation_rtol=0.05,
                ),
                description="Production should be robust under small humidity noise",
                consistency_profile="current_numerical_only",
            )
        )

    inactive = sorted(set(weights.keys()) - set(active_rule_names))
    return specs, effective_weights, inactive
