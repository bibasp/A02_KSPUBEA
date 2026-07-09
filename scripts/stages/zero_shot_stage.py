"""Zero-shot transfer baseline on the target basin."""
from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import DataLoader

from hydro_tl_ews.data.camels import CamelsDataset, STATIC_ATTRIBUTES
from hydro_tl_ews.data.datasets import MultiBasinSequenceDataset
from hydro_tl_ews.data.preprocessing import Normalizer, StaticNormalizer
from hydro_tl_ews.evaluation.metrics import kge, nse, pbias
from hydro_tl_ews.training.trainer import Trainer
from hydro_tl_ews.utils.config import ExperimentConfig
from hydro_tl_ews.utils.logging import get_logger

log = get_logger(__name__)


def run_zero_shot(cfg: ExperimentConfig) -> None:
    ds = CamelsDataset(cfg.data["camels_root"])
    target_id = cfg.data["target_basin"]
    target = ds.load_basin(target_id)
    attrs = ds.load_attributes()

    eval_period = tuple(cfg.data.get("full_period") or cfg.data["evaluation_period"])
    seq_len = int(cfg.data.get("sequence_length", 365))

    norm_period = tuple(cfg.data.get("normalization_period", eval_period))
    dyn_norm = Normalizer.fit(target.forcings.loc[norm_period[0]:norm_period[1]])
    static_norm = StaticNormalizer.fit(attrs.loc[:, STATIC_ATTRIBUTES])

    eval_ds = MultiBasinSequenceDataset(
        {target_id: target},
        eval_period,
        dyn_norm,
        static_norm,
        sequence_length=seq_len,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(cfg.training.get("batch_size", 256)),
        shuffle=False,
    )

    ckpt = cfg.model.get("pretrained_checkpoint")
    if not ckpt:
        raise ValueError("zero_shot stage requires model.pretrained_checkpoint")
    model = Trainer.load_model(ckpt)
    trainer = Trainer(model=model, mode="zero_shot", head_lr=1e-3)
    preds, obs = trainer.predict(eval_loader)

    metrics = {
        "NSE": nse(obs, preds),
        "KGE": kge(obs, preds),
        "PBIAS": pbias(obs, preds),
        "n_samples": int(len(preds)),
        "target_basin": target_id,
        "evaluation_period": [eval_period[0], eval_period[1]],
    }

    out_metrics = Path(cfg.output.get("metrics_path", "results/zero_shot_metrics.json"))
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2, default=float))

    # Save the dated daily series too — peak/extreme analyses (e.g. the
    # pseudo-ungauged peak-skill study) need the predictions, not just the
    # summary metrics, and re-running inference to get them back is waste.
    import pandas as pd
    preds_path = Path(cfg.output.get(
        "predictions_path",
        str(out_metrics).replace("_metrics.json", "_predictions.csv")))
    pd.DataFrame({"observed": obs, "predicted": preds},
                 index=pd.DatetimeIndex(eval_ds.dates, name="date")).to_csv(preds_path)
    log.info("Zero-shot evaluation complete | %s | predictions -> %s",
             metrics, preds_path)
