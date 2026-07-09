"""Regression tests for the walk-forward validation fixes.

Covers:
1. Trainer.fit restores best-epoch weights on early stopping (validation
   selects the model instead of only stopping training).
2. walk_forward honors refit_train_start (refits must not train on data
   before the data-scarce warmup) and passes a real val_loader to refits
   when val_tail_days > 0.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydro_tl_ews.data.camels import BasinData, DYNAMIC_FEATURES, STATIC_ATTRIBUTES
from hydro_tl_ews.data.preprocessing import Normalizer, StaticNormalizer
from hydro_tl_ews.models.ealstm import EALSTM, EALSTMConfig
from hydro_tl_ews.training.trainer import Trainer
from hydro_tl_ews.training.walk_forward import WalkForwardConfig, walk_forward


def _tiny_model(hidden: int = 8) -> EALSTM:
    return EALSTM(EALSTMConfig(
        dynamic_input_size=len(DYNAMIC_FEATURES),
        static_input_size=len(STATIC_ATTRIBUTES),
        hidden_size=hidden, dropout=0.0,
    ))


def _scripted_trainer(val_losses):
    """Trainer whose run_epoch is scripted: each train epoch stamps the
    epoch number into the head bias, and val epochs return scripted losses."""
    trainer = Trainer(model=_tiny_model(), device="cpu", mode="local")
    calls = {"n": 0}

    def fake_run_epoch(loader, train, basin_std=1.0):
        if train:
            calls["n"] += 1
            with torch.no_grad():
                trainer.model.head.bias.fill_(float(calls["n"]))
            return 1.0
        return val_losses[calls["n"] - 1]

    trainer.run_epoch = fake_run_epoch
    return trainer


def test_fit_restores_best_weights_on_early_stop():
    # Best val at epoch 2; patience 2 exhausted at epoch 4.
    trainer = _scripted_trainer(val_losses=[1.0, 0.5, 0.9, 0.95, 0.4])
    state = trainer.fit(train_loader=None, val_loader=object(),
                        epochs=5, patience=2)
    assert state.best_epoch == 2
    assert state.epoch == 4, "early stopping should trigger at epoch 4"
    assert float(trainer.model.head.bias) == 2.0, \
        "model must carry the best-epoch weights, not the last-epoch weights"


def test_fit_keeps_last_weights_when_disabled():
    trainer = _scripted_trainer(val_losses=[1.0, 0.5, 0.9, 0.95, 0.4])
    state = trainer.fit(train_loader=None, val_loader=object(),
                        epochs=5, patience=2, restore_best_weights=False)
    assert state.best_epoch == 2
    assert float(trainer.model.head.bias) == 4.0


def _synthetic_basin(start="2000-01-01", end="2003-12-31") -> BasinData:
    idx = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(0)
    forcings = pd.DataFrame(
        rng.uniform(0.1, 1.0, size=(len(idx), len(DYNAMIC_FEATURES))),
        index=idx, columns=DYNAMIC_FEATURES)
    streamflow = pd.Series(rng.gamma(2.0, 1.0, size=len(idx)), index=idx)
    attributes = pd.Series(1.0, index=STATIC_ATTRIBUTES)
    return BasinData(basin_id="99999999", forcings=forcings,
                     streamflow=streamflow, attributes=attributes)


def _run_walk_forward(basin, seen, **cfg_kwargs):
    """One-refit walk-forward with a recording no-op refit_fn."""
    cfg = WalkForwardConfig(
        initial_train_end="2002-12-31", eval_end="2003-03-31",
        refit_every_days=90, online_bias_correction=False,
        sequence_length=30, batch_size=64, **cfg_kwargs)
    dyn_norm = Normalizer.fit(basin.forcings.loc[:"2002-12-31"])
    static_norm = StaticNormalizer.fit(basin.attributes.to_frame().T)

    def recording_refit(model, train_loader, val_loader, ft_cfg, device=None):
        seen.append({
            "n_train": len(train_loader.dataset),
            "has_val": val_loader is not None,
            "n_val": len(val_loader.dataset) if val_loader is not None else 0,
        })

    return walk_forward(_tiny_model(), basin, dyn_norm, static_norm, cfg,
                        device="cpu", refit_fn=recording_refit)


def test_refit_train_start_limits_training_window():
    basin = _synthetic_basin()
    full, scarce = [], []
    _run_walk_forward(basin, full, val_tail_days=0)
    _run_walk_forward(basin, scarce, val_tail_days=0,
                      refit_train_start="2002-01-01")
    # Full history: targets 2000-01-30..2002-12-31; scarce: 2002-01-30..12-31.
    assert scarce[0]["n_train"] == 365 - 30 + 1
    assert full[0]["n_train"] > scarce[0]["n_train"]


def test_dataset_exposes_target_dates():
    import numpy as np
    from hydro_tl_ews.data.datasets import MultiBasinSequenceDataset
    basin = _synthetic_basin()
    dyn_norm = Normalizer.fit(basin.forcings)
    static_norm = StaticNormalizer.fit(basin.attributes.to_frame().T)
    ds = MultiBasinSequenceDataset({"99999999": basin},
                                   ("2000-01-01", "2000-12-31"),
                                   dyn_norm, static_norm, sequence_length=30)
    assert len(ds.dates) == len(ds)
    # Right-aligned windows: first target is day seq_len-1 of the period.
    assert np.datetime64("2000-01-30") == ds.dates[0].astype("datetime64[D]")
    assert np.datetime64("2000-12-31") == ds.dates[-1].astype("datetime64[D]")


def test_val_tail_supplies_validation_loader():
    basin = _synthetic_basin()
    seen = []
    result = _run_walk_forward(basin, seen, val_tail_days=20,
                               refit_train_start="2002-01-01")
    assert seen[0]["has_val"], "refit must receive a real validation loader"
    assert seen[0]["n_val"] == 20
    # Training targets end 20 days earlier (tail held out for validation).
    assert seen[0]["n_train"] == (365 - 20) - 30 + 1
    assert len(result.predicted) == len(result.dates) > 0
