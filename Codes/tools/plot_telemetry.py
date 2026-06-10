#!/usr/bin/env python3
"""Plot training telemetry written by train_vp_segmentation.py.

Each run directory (the `eg_save_path` of a run, i.e. the `{setting}` folder under
output_dir/.../) is expected to contain `telemetry_steps.csv`, `telemetry_epochs.csv`
and `config.json`. Pass one or more run dirs to overlay them (e.g. baseline vs L_div).

Examples
--------
  # single run
  python -m Codes.tools.plot_telemetry path/to/run_dir

  # compare baseline vs diversity run, with explicit labels
  python -m Codes.tools.plot_telemetry RUN_BASELINE RUN_DIV \
      --labels baseline div0.01 --out Data/output/plots/cmp
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def smooth(s, k):
    if k <= 1:
        return s
    return s.rolling(window=k, min_periods=1, center=False).mean()


def load_run(run_dir):
    steps_p = os.path.join(run_dir, 'telemetry_steps.csv')
    epochs_p = os.path.join(run_dir, 'telemetry_epochs.csv')
    cfg_p = os.path.join(run_dir, 'config.json')
    steps = pd.read_csv(steps_p) if os.path.exists(steps_p) else None
    epochs = pd.read_csv(epochs_p) if os.path.exists(epochs_p) else None
    cfg = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
    return steps, epochs, cfg


def main():
    ap = argparse.ArgumentParser(description='Plot CONDENSER training telemetry.')
    ap.add_argument('run_dirs', nargs='+', help='one or more run directories (eg_save_path)')
    ap.add_argument('--labels', nargs='*', default=None, help='legend label per run dir')
    ap.add_argument('--out', default=None, help='output prefix for PNGs (default: first run dir)')
    ap.add_argument('--smooth', type=int, default=50, help='step-curve smoothing window')
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(d)) or d for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise SystemExit('--labels must match the number of run_dirs')
    out_prefix = args.out or os.path.join(args.run_dirs[0], 'telemetry')
    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)

    runs = [load_run(d) for d in args.run_dirs]

    # ----- 1) per-epoch loss components + raw scales (the normalization check) -----
    comp = ['l_tp', 'l_pa', 'l_div', 'l_conf', 'l_total']
    titles = {
        'l_tp': 'Task / reconstruction (L_TP)',
        'l_pa': 'Pre-alignment raw (L_PA)',
        'l_div': 'Mean pairwise candidate cosine (L_div)',
        'l_conf': 'Confidence penalty raw (L_conf)',
        'l_total': 'Total optimized loss',
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    for ax, c in zip(axes, comp):
        for (steps, epochs, cfg), lab in zip(runs, labels):
            if epochs is not None and c in epochs:
                ax.plot(epochs['epoch'], epochs[c], marker='.', label=lab)
        ax.set_title(titles[c]); ax.set_xlabel('epoch'); ax.set_ylabel(c); ax.grid(alpha=0.3); ax.legend()
    # val mIoU
    ax = axes[5]
    for (steps, epochs, cfg), lab in zip(runs, labels):
        if epochs is not None and 'val_iou' in epochs:
            ax.plot(epochs['epoch'], epochs['val_iou'], marker='.', label=lab)
    ax.set_title('Validation mIoU'); ax.set_xlabel('epoch'); ax.set_ylabel('mIoU'); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_prefix + '_epochs.png', dpi=130); plt.close(fig)

    # ----- 2) weighted contributions on a shared axis (does L_div actually matter?) -----
    fig, ax = plt.subplots(figsize=(10, 6))
    for (steps, epochs, cfg), lab in zip(runs, labels):
        if epochs is None:
            continue
        lamba = float(cfg.get('lamba', 1.0))
        dl = float(cfg.get('diversity_lambda', 0.0))
        cl = float(cfg.get('conf_lambda', 0.0))
        ax.plot(epochs['epoch'], epochs['l_tp'], label=f'{lab}: L_TP')
        ax.plot(epochs['epoch'], lamba * epochs['l_pa'], '--', label=f'{lab}: lamba*L_PA')
        if dl > 0:
            ax.plot(epochs['epoch'], dl * epochs['l_div'], ':', label=f'{lab}: div_lambda*L_div')
        if cl > 0:
            ax.plot(epochs['epoch'], cl * epochs['l_conf'], '-.', label=f'{lab}: conf_lambda*L_conf')
    ax.set_yscale('log')
    ax.set_title('Weighted loss contributions (log scale) — relative magnitudes')
    ax.set_xlabel('epoch'); ax.set_ylabel('contribution to total loss'); ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_prefix + '_weighted_contributions.png', dpi=130); plt.close(fig)

    # ----- 3) per-step total loss (smoothed) -----
    fig, ax = plt.subplots(figsize=(10, 6))
    for (steps, epochs, cfg), lab in zip(runs, labels):
        if steps is not None and 'l_total' in steps:
            ax.plot(steps['global_step'], smooth(steps['l_total'], args.smooth), label=lab)
    ax.set_title(f'Per-step total loss (smoothed w={args.smooth})')
    ax.set_xlabel('global step'); ax.set_ylabel('l_total'); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_prefix + '_steps.png', dpi=130); plt.close(fig)

    # ----- text summary -----
    print(f'Wrote: {out_prefix}_epochs.png, _weighted_contributions.png, _steps.png\n')
    for (steps, epochs, cfg), lab in zip(runs, labels):
        if epochs is None or not len(epochs):
            continue
        last = epochs.iloc[-1]
        best = epochs['val_iou'].max() if 'val_iou' in epochs else float('nan')
        print(f'[{lab}] epochs={len(epochs)} best_val_iou={best:.4f} '
              f'last: L_TP={last.l_tp:.4f} L_PA={last.l_pa:.4f} L_div={last.l_div:.4f} '
              f'L_conf={last.l_conf:.4f} total={last.l_total:.4f}')


if __name__ == '__main__':
    main()
