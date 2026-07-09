"""Minimum local-record sensitivity experiment (3m/6m/1y/2y)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from hydro_tl_ews.data.camels import CamelsDataset, STATIC_ATTRIBUTES
from hydro_tl_ews.data.datasets import MultiBasinSequenceDataset
from hydro_tl_ews.data.preprocessing import Normalizer, StaticNormalizer
from hydro_tl_ews.evaluation.metrics import kge, nse, pbias
from hydro_tl_ews.training.trainer import Trainer
from hydro_tl_ews.training.transfer import FineTuneConfig, fine_tune_conservative
from hydro_tl_ews.utils.config import ExperimentConfig
from hydro_tl_ews.utils.logging import get_logger

log = get_logger(__name__)


def _period_end(start: str, months: int) -> str:
    st = pd.Timestamp(start)
    return str((st + pd.DateOffset(months=months) - pd.Timedelta(days=1)).date())


def run_min_data_sensitivity(cfg: ExperimentConfig) -> None:
    ds = CamelsDataset(cfg.data["camels_root"])
    target_id = cfg.data["target_basin"]
    target = ds.load_basin(target_id)
    attrs = ds.load_attributes()

    warmup_start = cfg.data["warmup_start"]
    val_period = tuple(cfg.data["validation_period"])
    eval_period = tuple(cfg.data["evaluation_period"])
    seq_len = int(cfg.data.get("sequence_length", 365))
    month_windows = cfg.data.get("warmup_months", [3, 6, 12, 24])

    static_norm = StaticNormalizer.fit(attrs.loc[:, STATIC_ATTRIBUTES])

    ckpt = cfg.model.get("pretrained_checkpoint")
    if not ckpt:
        raise ValueError("min_data_sensitivity stage requires model.pretrained_checkpoint")
    base_model = Trainer.load_model(ckpt)
    base_state = base_model.state_dict()

    ft_cfg = FineTuneConfig(
        head_lr=float(cfg.training.get("head_lr", 1e-3)),
        lstm_lr=float(cfg.training.get("lstm_lr", 1e-5)),
        epochs_head_only=int(cfg.training.get("epochs", 5)),
        epochs_progressive=0,
        patience=int(cfg.training.get("patience", 3)),
        unfreeze_fraction=0.0,
        weight_decay=float(cfg.training.get("weight_decay", 0.0)),
    )

    rows = []
    for months in month_windows:
        warmup_end = _period_end(warmup_start, int(months))
        # Fit normalizer on warmup period only — eval loader uses the SAME
        # statistics to avoid train/eval normalization mismatch.
        dyn_norm = Normalizer.fit(target.forcings.loc[warmup_start:warmup_end])
        warmup_ds = MultiBasinSequenceDataset(
            {target_id: target},
            (warmup_start, warmup_end),
            dyn_norm,
            static_norm,
            sequence_length=seq_len,
        )
        val_ds = MultiBasinSequenceDataset(
            {target_id: target},
            val_period,
            dyn_norm,
            static_norm,
            sequence_length=seq_len,
        )
        eval_ds = MultiBasinSequenceDataset(
            {target_id: target},
            eval_period,
            dyn_norm,
            static_norm,
            sequence_length=seq_len,
        )
        # A warmup shorter than sequence_length yields no training windows
        # (e.g. 3/6-month records with seq_len 365). Record and skip rather
        # than crash the DataLoader on an empty dataset.
        if len(warmup_ds) == 0:
            log.warning("Warmup=%s months yields 0 training windows (< seq_len %d days); skipping.",
                        months, seq_len)
            rows.append({
                "warmup_months": int(months),
                "warmup_period_start": warmup_start,
                "warmup_period_end": warmup_end,
                "NSE": float("nan"), "KGE": float("nan"), "PBIAS": float("nan"),
                "n_train_samples": 0, "n_eval_samples": int(len(eval_ds)),
            })
            continue
        train_loader = DataLoader(warmup_ds, batch_size=int(cfg.training.get("batch_size", 64)),
                                  shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=int(cfg.training.get("batch_size", 64)),
                                shuffle=False) if len(val_ds) else None
        eval_loader = DataLoader(eval_ds, batch_size=int(cfg.training.get("batch_size", 128)),
                                 shuffle=False)

        model = Trainer.load_model(ckpt)
        model.load_state_dict(base_state)
        fine_tune_conservative(model, train_loader, val_loader, ft_cfg)
        trainer = Trainer(model=model, mode="conservative", head_lr=ft_cfg.head_lr)
        pred, obs = trainer.predict(eval_loader)
        rows.append({
            "warmup_months": int(months),
            "warmup_period_start": warmup_start,
            "warmup_period_end": warmup_end,
            "NSE": nse(obs, pred),
            "KGE": kge(obs, pred),
            "PBIAS": pbias(obs, pred),
            "n_train_samples": int(len(warmup_ds)),
            "n_eval_samples": int(len(eval_ds)),
        })
        log.info("Warmup=%s months | NSE=%.4f", months, rows[-1]["NSE"])

    out_table = Path(cfg.output.get("table_path", "results/min_data_sensitivity.csv"))
    out_table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_table, index=False)

    out_json = Path(cfg.output.get("summary_path", "results/min_data_sensitivity.json"))
    out_json.write_text(json.dumps({"target_basin": target_id, "results": rows}, indent=2,
                                   default=float))
    log.info("Minimum data sensitivity complete | wrote %s", out_table)
