#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs results

EPOCHS=${EPOCHS:-1}
BATCH=${BATCH:-1}
WORKERS=${WORKERS:-0}
BASE_OUT=${BASE_OUT:-nlrt_ablation}
COMMON_ARGS=(
  --data_path data/bunny --vid bunny --conv_type convnext pshuffel --act gelu --norm none
  --crop_list 640_1280 --resize_list -1 --loss L2 --enc_strds 5 4 4 2 2 --enc_dim 64_16
  --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2 --modelsize 0.2 --lower_width 12
  -e "$EPOCHS" --eval_freq 1 -b "$BATCH" --lr 0.001 --workers "$WORKERS" --debug
  --nlrt_h0 2 --nlrt_w0 4 --nlrt_c 16 --nlrt_t_pe 1.25_16 --nlrt_lambda_reg 1e-6
)

run_exp() {
  local name="$1"
  shift
  local log_file="logs/${name}.log"
  echo "[RUN] ${name}"
  python train_nerv_all.py "${COMMON_ARGS[@]}" --outf "$BASE_OUT/$name" --suffix "_${name}" "$@" | tee "$log_file"
}

run_exp baseline --repr_type baseline
run_exp nlrt_r8 --repr_type nlrt --nlrt_rank 8 --nlrt_lambda_t 1e-4
run_exp nlrt_r16 --repr_type nlrt --nlrt_rank 16 --nlrt_lambda_t 1e-4
run_exp nlrt_r32 --repr_type nlrt --nlrt_rank 32 --nlrt_lambda_t 1e-4
run_exp nlrt_r16_t0 --repr_type nlrt --nlrt_rank 16 --nlrt_lambda_t 0
run_exp nlrt_r16_t1e4 --repr_type nlrt --nlrt_rank 16 --nlrt_lambda_t 1e-4
run_exp nlrt_r16_t1e3 --repr_type nlrt --nlrt_rank 16 --nlrt_lambda_t 1e-3

python - <<'PY'
import os
import glob
import pandas as pd

rows = []
for csv_path in sorted(glob.glob('output/debug/bunny/**/epoch*.csv', recursive=True)):
    run_name = os.path.basename(os.path.dirname(csv_path))
    if not any(tag in run_name for tag in ['_baseline', '_nlrt_r8', '_nlrt_r16', '_nlrt_r32', '_nlrt_r16_t0', '_nlrt_r16_t1e4', '_nlrt_r16_t1e3']):
        continue
    csv_candidates = [csv_path]
    if not csv_candidates:
        continue
    df = pd.read_csv(csv_path)
    row = df.iloc[0].to_dict()
    ckpt_path = os.path.join(os.path.dirname(csv_path), f"epoch{int(row.get('CurEpoch', 1))}.pth")
    if os.path.isfile(ckpt_path):
        row['checkpoint_size_mb'] = os.path.getsize(ckpt_path) / (1024 * 1024)
    if run_name.endswith('_baseline'):
        row['run_name'] = 'baseline'
    elif run_name.endswith('_nlrt_r8'):
        row['run_name'] = 'nlrt_r8'
    elif run_name.endswith('_nlrt_r16'):
        row['run_name'] = 'nlrt_r16'
    elif run_name.endswith('_nlrt_r32'):
        row['run_name'] = 'nlrt_r32'
    elif run_name.endswith('_nlrt_r16_t0'):
        row['run_name'] = 'nlrt_r16_t0'
    elif run_name.endswith('_nlrt_r16_t1e4'):
        row['run_name'] = 'nlrt_r16_t1e4'
    elif run_name.endswith('_nlrt_r16_t1e3'):
        row['run_name'] = 'nlrt_r16_t1e3'
    else:
        continue
    rows.append(row)

if not rows:
    raise RuntimeError('No run results found to summarize.')

out = pd.DataFrame(rows)
out = out.sort_values('run_name')
keep_cols = [
    'run_name', 'repr_type', 'nlrt_rank', 'nlrt_lambda_t', 'pred_seen_psnr', 'pred_seen_ssim', 'lpips',
    'param_count_m', 'checkpoint_size_mb', 'decode_ms_per_frame', 'train_it_per_s',
    'loss_rec', 'loss_temp', 'loss_reg', 'loss_total', 'git_commit', 'python', 'pytorch', 'cuda'
]
for c in keep_cols:
    if c not in out.columns:
        out[c] = 'N/A'
out['lpips'] = out['lpips'].fillna('N/A')
out = out[keep_cols]
out.to_csv('results/nlrt_ablation.csv', index=False)
print('Saved results/nlrt_ablation.csv')
PY
