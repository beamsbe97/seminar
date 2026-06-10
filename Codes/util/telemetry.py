"""Training telemetry logger shared by all train_vp_*.py scripts.

Persists, at the run root (`eg_save_path`):
  - config.json            : full hyperparameter snapshot (for reproducibility)
  - telemetry_steps.csv    : one row per optimizer step (loss components + weighted contributions)
  - telemetry_epochs.csv   : one row per epoch (mean loss components + val metrics + best)

All CSV writes are append-mode and keyed by epoch / global_step, so they survive
checkpoint resume (the loop seeds global_step from begin_epoch). The optimized loss
is never touched here — this module only *reads* the components the model exposes
on `VP.loss_terms` after each forward pass.

Usage
-----
    from <pkg>.util.telemetry import TelemetryLogger
    tlog = TelemetryLogger(eg_save_path, args,
                           metric_keys=['iou', 'color_blind_iou', 'accuracy'],
                           steps_per_epoch=len(dataloaders['train']),
                           begin_epoch=begin_epoch)
    ...
    # inside the training step loop, after scaler.update():
    tlog.log_step(epoch, i, optimizer.param_groups[0]['lr'], VP.loss_terms)
    ...
    # once per epoch, after val metrics + best are known:
    tlog.log_epoch(epoch, lr_list[-1], eval_dict, best_iou)
"""
import csv
import json
import os

try:
    import torch
except Exception:  # torch always present in practice; keep import-safe for tooling
    torch = None

_LOSS_KEYS = ['l_tp', 'l_pa', 'l_div', 'l_conf', 'reg', 'l_total']


def _to_float(v):
    if v is None:
        return float('nan')
    if torch is not None and torch.is_tensor(v):
        return v.item()
    return float(v)


class TelemetryLogger:
    def __init__(self, run_dir, args, metric_keys, steps_per_epoch, begin_epoch=1):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.metric_keys = list(metric_keys)
        self.lamba = float(getattr(args, 'lamba', 1.0) or 0.0)
        self.div = float(getattr(args, 'diversity_lambda', 0.0) or 0.0)
        self.conf = float(getattr(args, 'conf_lambda', 0.0) or 0.0)

        self.step_csv = os.path.join(run_dir, 'telemetry_steps.csv')
        self.epoch_csv = os.path.join(run_dir, 'telemetry_epochs.csv')

        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)

        self._step_fields = (['epoch', 'step', 'global_step', 'lr'] + _LOSS_KEYS
                             + ['lamba', 'diversity_lambda', 'conf_lambda',
                                'w_l_pa', 'w_l_div', 'w_l_conf'])
        self._epoch_fields = (['epoch', 'lr'] + _LOSS_KEYS
                              + ['val_' + m for m in self.metric_keys] + ['best_metric'])
        if not os.path.exists(self.step_csv):
            with open(self.step_csv, 'w', newline='') as f:
                csv.writer(f).writerow(self._step_fields)
        if not os.path.exists(self.epoch_csv):
            with open(self.epoch_csv, 'w', newline='') as f:
                csv.writer(f).writerow(self._epoch_fields)

        self.global_step = (begin_epoch - 1) * max(steps_per_epoch, 0)
        self._reset_epoch()

    def _reset_epoch(self):
        self._sum = {k: 0.0 for k in _LOSS_KEYS}
        self._n = 0

    def log_step(self, epoch, step, lr, loss_terms):
        lt = {k: _to_float(loss_terms.get(k)) for k in _LOSS_KEYS}
        with open(self.step_csv, 'a', newline='') as f:
            csv.writer(f).writerow(
                [epoch, step, self.global_step, lr]
                + [lt[k] for k in _LOSS_KEYS]
                + [self.lamba, self.div, self.conf,
                   self.lamba * lt['l_pa'], self.div * lt['l_div'], self.conf * lt['l_conf']])
        for k in _LOSS_KEYS:
            v = lt[k]
            if v == v:  # skip NaN
                self._sum[k] += v
        self._n += 1
        self.global_step += 1

    def log_epoch(self, epoch, lr, eval_dict, best_metric):
        mean = {k: (self._sum[k] / self._n if self._n else float('nan')) for k in _LOSS_KEYS}
        row = ([epoch, lr] + [mean[k] for k in _LOSS_KEYS]
               + [eval_dict.get(m) for m in self.metric_keys] + [best_metric])
        with open(self.epoch_csv, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        self._reset_epoch()
