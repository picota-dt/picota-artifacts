from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from kan.KAN import KAN
from kan.MetamorphicLoss import CompositeMetamorphicLoss
from kan.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader

from metamorphic_evaluation import compute_violation_report


@dataclass
class EvalMetrics:
    n_samples: int
    output_scale: str
    mae_model: float
    rmse_model: float
    r2: float
    mae_raw: float
    rmse_raw: float
    max_abs_err_raw: float
    p95_abs_err_raw: float
    p99_abs_err_raw: float
    tail5_mean_abs_err_raw: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "n_samples": self.n_samples,
            "output_scale": self.output_scale,
            "mae_model": self.mae_model,
            "rmse_model": self.rmse_model,
            "r2": self.r2,
            "mae_raw": self.mae_raw,
            "rmse_raw": self.rmse_raw,
            "max_abs_err_raw": self.max_abs_err_raw,
            "p95_abs_err_raw": self.p95_abs_err_raw,
            "p99_abs_err_raw": self.p99_abs_err_raw,
            "tail5_mean_abs_err_raw": self.tail5_mean_abs_err_raw,
        }


class MetamorphicAlternativeKanTrainer:
    """
    Generic KAN trainer with optional metamorphic composite loss.

    This trainer is domain-agnostic; provide rule_specs from each use-case module.
    """

    def __init__(
            self,
            name: str,
            input_variables: list[str],
            output_variable: str,
            lookback: int,
            means: list[float],
            stds: list[float],
            out_min: float,
            out_max: float,
            batch_size: int,
            epochs: int,
            device,
            lr: float,
            seed: int,
            rule_specs: Iterable[object] | None = None,
            supervised_weight: float = 1.0,
            relation_constraint_weight: float = 0.25,
            worst_case_over_T_weight: float = 0.0,
    ):
        self.name = name
        self.input_variables = list(input_variables)
        self.output_variable = output_variable
        self.lookback = int(lookback)
        self.means = list(means)
        self.stds = list(stds)
        self.out_min = float(out_min)
        self.out_max = float(out_max)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.device = device
        self.lr = float(lr)
        self.seed = int(seed)
        self.last_validation_metrics: dict[str, float | int | str] | None = None

        if rule_specs is None:
            self.loss_fn = torch.nn.MSELoss()
            self.loss_mode = "baseline_mse"
        else:
            self.loss_fn = CompositeMetamorphicLoss.from_rule_specs(
                rule_specs=rule_specs,
                supervised_loss=torch.nn.MSELoss(),
                supervised_weight=float(supervised_weight),
                relation_constraint_weight=float(relation_constraint_weight),
                worst_case_over_T_weight=float(worst_case_over_T_weight),
            )
            self.loss_mode = "composite_metamorphic"

    def _set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _raw_span_or_one(self) -> float:
        span = self.out_max - self.out_min
        return span if span > 0 else 1.0

    def _make_loader(self, items: list[dict], shuffle: bool) -> DataLoader:
        return DataLoader(TimeSeriesDataset(items), batch_size=self.batch_size, shuffle=shuffle)

    def _compute_loss(self, model: KAN, batch: dict, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        if hasattr(self.loss_fn, "compute_training_loss"):
            return self.loss_fn.compute_training_loss(
                model=model,
                batch=batch,
                target=target,
                prediction=prediction,
            )
        return self.loss_fn(prediction, target)

    def _evaluate_loader(self, model: KAN, data_loader: DataLoader) -> EvalMetrics:
        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for batch in data_loader:
                pred = model(batch).squeeze().reshape(-1)
                out = batch["out"].reshape(-1)
                preds.append(pred.detach().to(self.device))
                targets.append(out.detach().to(self.device))

        if not preds:
            return EvalMetrics(
                n_samples=0,
                output_scale="normalized",
                mae_model=float("inf"),
                rmse_model=float("inf"),
                r2=float("nan"),
                mae_raw=float("inf"),
                rmse_raw=float("inf"),
                max_abs_err_raw=float("inf"),
                p95_abs_err_raw=float("inf"),
                p99_abs_err_raw=float("inf"),
                tail5_mean_abs_err_raw=float("inf"),
            )

        y_pred = torch.cat(preds)
        y_true = torch.cat(targets)
        err = y_pred - y_true
        mae_model = float(torch.mean(torch.abs(err)).item())
        rmse_model = float(torch.sqrt(torch.mean(err ** 2)).item())

        ss_res = float(torch.sum(err ** 2).item())
        ss_tot = float(torch.sum((y_true - torch.mean(y_true)) ** 2).item())
        r2 = float("nan") if ss_tot == 0.0 else float(1.0 - (ss_res / ss_tot))

        raw_span = self._raw_span_or_one()
        mae_raw = float(mae_model * raw_span)
        rmse_raw = float(rmse_model * raw_span)
        abs_err_raw = torch.abs(err) * raw_span
        max_abs_err_raw = float(torch.max(abs_err_raw).item())
        p95_abs_err_raw = float(torch.quantile(abs_err_raw, 0.95).item())
        p99_abs_err_raw = float(torch.quantile(abs_err_raw, 0.99).item())
        sample_count = int(abs_err_raw.numel())
        tail_k = max(1, (sample_count + 19) // 20)  # ceil(5%)
        tail5_mean_abs_err_raw = float(torch.topk(abs_err_raw, k=tail_k).values.mean().item())

        return EvalMetrics(
            n_samples=int(y_true.numel()),
            output_scale="normalized",
            mae_model=mae_model,
            rmse_model=rmse_model,
            r2=r2,
            mae_raw=mae_raw,
            rmse_raw=rmse_raw,
            max_abs_err_raw=max_abs_err_raw,
            p95_abs_err_raw=p95_abs_err_raw,
            p99_abs_err_raw=p99_abs_err_raw,
            tail5_mean_abs_err_raw=tail5_mean_abs_err_raw,
        )

    def train(self, train_items: list[dict], val_items: list[dict]) -> tuple[KAN, dict[str, float | int | str]]:
        if len(train_items) == 0 or len(val_items) == 0:
            raise ValueError("train_items and val_items must be non-empty")

        self._set_seed()
        model = KAN(
            input_features=len(self.input_variables),
            lookback_size=self.lookback,
            means=self.means,
            stds=self.stds,
            output_features=1,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        train_loader = self._make_loader(train_items, shuffle=True)
        val_loader = self._make_loader(val_items, shuffle=False)

        best_state = copy.deepcopy(model.state_dict())
        best_val_mae = float("inf")
        best_val_metrics: dict[str, float | int | str] | None = None

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            total_count = 0
            supervised_sum = 0.0
            relation_sum = 0.0
            worst_case_sum = 0.0
            target_mapped_sum = 0.0
            metric_count = 0

            for batch in train_loader:
                out = batch["out"]
                optimizer.zero_grad()
                pred = model(batch).squeeze()
                loss = self._compute_loss(model=model, batch=batch, target=out, prediction=pred)
                loss.backward()
                optimizer.step()

                batch_size = int(out.shape[0] if out.dim() > 0 else 1)
                total_loss += float(loss.detach().item()) * batch_size
                total_count += batch_size

                if hasattr(self.loss_fn, "last_metrics") and self.loss_fn.last_metrics:
                    metrics = self.loss_fn.last_metrics
                    supervised_sum += float(metrics.get("supervised_loss", 0.0)) * batch_size
                    relation_sum += float(metrics.get("relation_constraint_penalty", 0.0)) * batch_size
                    worst_case_sum += float(metrics.get("worst_case_over_T_loss", 0.0)) * batch_size
                    target_mapped_sum += float(metrics.get("target_mapped_supervised_loss", 0.0)) * batch_size
                    metric_count += batch_size

            val_metrics = self._evaluate_loader(model, val_loader)
            train_avg = total_loss / max(1, total_count)
            if metric_count > 0:
                print(
                    f"{self.name}\tepoch={epoch + 1}/{self.epochs}\tmode={self.loss_mode}\t"
                    f"train_total={train_avg:.6f}\t"
                    f"train_supervised={supervised_sum / metric_count:.6f}\t"
                    f"train_relation_constraint_penalty={relation_sum / metric_count:.6f}\t"
                    f"train_worst_case_over_T_loss={worst_case_sum / metric_count:.6f}\t"
                    f"train_target_mapped={target_mapped_sum / metric_count:.6f}\t"
                    f"val_mae={val_metrics.mae_model:.6f}\t"
                    f"val_rmse={val_metrics.rmse_model:.6f}\t"
                    f"val_r2={val_metrics.r2:.6f}",
                    flush=True,
                )
            else:
                print(
                    f"{self.name}\tepoch={epoch + 1}/{self.epochs}\tmode={self.loss_mode}\t"
                    f"train_loss={train_avg:.6f}\t"
                    f"val_mae={val_metrics.mae_model:.6f}\t"
                    f"val_rmse={val_metrics.rmse_model:.6f}\t"
                    f"val_r2={val_metrics.r2:.6f}",
                    flush=True,
                )

            if val_metrics.mae_model < best_val_mae:
                best_val_mae = val_metrics.mae_model
                best_state = copy.deepcopy(model.state_dict())
                best_val_metrics = val_metrics.as_dict()

        model.load_state_dict(best_state)
        self.last_validation_metrics = best_val_metrics
        if best_val_metrics is None:
            best_val_metrics = self._evaluate_loader(model, val_loader).as_dict()
            self.last_validation_metrics = best_val_metrics

        print(
            f"{self.name}\tbest_val_n={best_val_metrics['n_samples']}\t"
            f"best_val_mae={float(best_val_metrics['mae_model']):.6f}\t"
            f"best_val_rmse={float(best_val_metrics['rmse_model']):.6f}\t"
            f"best_val_r2={float(best_val_metrics['r2']):.6f}\t"
            f"best_val_mae_raw={float(best_val_metrics['mae_raw']):.6f}\t"
            f"best_val_rmse_raw={float(best_val_metrics['rmse_raw']):.6f}\t"
            f"best_val_max_abs_err_raw={float(best_val_metrics['max_abs_err_raw']):.6f}\t"
            f"best_val_p95_abs_err_raw={float(best_val_metrics['p95_abs_err_raw']):.6f}\t"
            f"best_val_p99_abs_err_raw={float(best_val_metrics['p99_abs_err_raw']):.6f}\t"
            f"best_val_tail5_mean_abs_err_raw={float(best_val_metrics['tail5_mean_abs_err_raw']):.6f}",
            flush=True,
        )
        return model, best_val_metrics

    def evaluate(self, model: KAN, items: list[dict]) -> dict[str, float | int | str]:
        if len(items) == 0:
            raise ValueError("items must be non-empty")
        loader = self._make_loader(items, shuffle=False)
        metrics = self._evaluate_loader(model, loader).as_dict()
        return metrics

    def _relation_tests(self) -> list:
        if not isinstance(self.loss_fn, CompositeMetamorphicLoss):
            return []
        tests = list(getattr(self.loss_fn, "assigned_relation_constraints", []) or [])
        if tests:
            return tests
        return list(getattr(self.loss_fn, "relation_constraints", []) or [])

    def evaluate_with_rule_violations(
            self,
            model: KAN,
            items: list[dict],
            atol: float = 1e-6,
            rtol: float = 1e-4,
    ) -> tuple[dict[str, float | int | str], dict | None]:
        metrics = self.evaluate(model, items)
        relation_tests = self._relation_tests()
        if not relation_tests:
            return metrics, None
        loader = self._make_loader(items, shuffle=False)
        report = compute_violation_report(
            model=model,
            data_loader=loader,
            metamorphic_tests=relation_tests,
            atol=float(atol),
            rtol=float(rtol),
        )
        return metrics, report
