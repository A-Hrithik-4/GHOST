import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

df = pd.read_csv('/Users/hrithika/Desktop/GHOST/data/processed/ghosttrack_phase6_ekf.csv')
t = df['timestamp'].values
cnn = df['cnn_predicted_speed'].values
gt = df['ground_truth_speed'].values

num_samples = len(df) - 30 + 1
n_train = int(num_samples * 0.70)
n_val = int(num_samples * 0.15)

train_rows = np.arange(29, n_train + 29)
val_rows = np.arange(n_train + 29, n_train + n_val + 29)
test_rows = np.arange(n_train + n_val + 29, len(df))
outage_rows = df[
    (df['timestamp'] >= 30.0) &
    (df['timestamp'] < 90.0)
].index.values

# Clean training data only
t_train = t[train_rows]
train_clean_mask = (t_train < 30.0) | (t_train >= 90.0)
train_rows_clean = train_rows[train_clean_mask]

cnn_tr = cnn[train_rows_clean]
gt_tr = gt[train_rows_clean]

# ============================================================
# FIT ISOTONIC — TRAINING DATA ONLY
# ============================================================
model_e = IsotonicRegression(
    out_of_bounds='clip'
).fit(cnn_tr, gt_tr)

# Predictions
pred_tr = np.maximum(model_e.predict(cnn[train_rows_clean]), 0.0)
pred_val = np.maximum(model_e.predict(cnn[val_rows]), 0.0)
pred_test = np.maximum(model_e.predict(cnn[test_rows]), 0.0)
pred_out = np.maximum(model_e.predict(cnn[outage_rows]), 0.0)

def stats(name, rows, pred):
    print(f'\n{name}')
    print('-' * 80)
    print(f'N              : {len(rows)}')
    print(f'GT mean        : {np.mean(gt[rows]):.2f}')
    print(f'GT median      : {np.median(gt[rows]):.2f}')
    print(f'GT std         : {np.std(gt[rows]):.2f}')
    print(f'GT min         : {np.min(gt[rows]):.2f}')
    print(f'GT max         : {np.max(gt[rows]):.2f}')
    print(f'GT p5          : {np.percentile(gt[rows],5):.2f}')
    print(f'GT p25         : {np.percentile(gt[rows],25):.2f}')
    print(f'GT p75         : {np.percentile(gt[rows],75):.2f}')
    print(f'GT p95         : {np.percentile(gt[rows],95):.2f}')
    print(f'CNN mean       : {np.mean(cnn[rows]):.2f}')
    print(f'CNN median     : {np.median(cnn[rows]):.2f}')
    print(f'CNN std        : {np.std(cnn[rows]):.2f}')
    print(f'CNN min        : {np.min(cnn[rows]):.2f}')
    print(f'CNN max        : {np.max(cnn[rows]):.2f}')
    print(f'CNN MAE        : {mean_absolute_error(gt[rows], cnn[rows]):.2f}')
    print(f'CNN RMSE       : {root_mean_squared_error(gt[rows], cnn[rows]):.2f}')
    print(f'CNN bias       : {np.mean(cnn[rows] - gt[rows]):+.2f}')
    print(f'ISO mean       : {np.mean(pred):.2f}')
    print(f'ISO MAE        : {mean_absolute_error(gt[rows], pred):.2f}')
    print(f'ISO RMSE       : {root_mean_squared_error(gt[rows], pred):.2f}')
    print(f'ISO bias       : {np.mean(pred - gt[rows]):+.2f}')

print('\n' + '=' * 80)
print('ISOTONIC PREDICTION PERFORMANCE SUMMARY')
print('=' * 80)
stats('TRAIN — CLEAN', train_rows_clean, pred_tr)
stats('VALIDATION', val_rows, pred_val)
stats('TEST', test_rows, pred_test)
stats('GNSS OUTAGE', outage_rows, pred_out)

# ============================================================
# COHEN\'S D
# ============================================================
def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    s1 = np.std(x1, ddof=1)
    s2 = np.std(x2, ddof=1)
    pooled = np.sqrt(
        ((n1-1)*s1**2 + (n2-1)*s2**2) /
        (n1+n2-2)
    )
    return (np.mean(x1) - np.mean(x2)) / pooled

print('\n' + '=' * 80)
print("STANDARDIZED MEAN DIFFERENCES — COHEN'S D")
print('=' * 80)
print(f'Train vs Validation : {cohens_d(gt[train_rows_clean], gt[val_rows]):+.3f}')
print(f'Train vs Test       : {cohens_d(gt[train_rows_clean], gt[test_rows]):+.3f}')
print(f'Train vs Outage     : {cohens_d(gt[train_rows_clean], gt[outage_rows]):+.3f}')
print(f'Validation vs Outage : {cohens_d(gt[val_rows], gt[outage_rows]):+.3f}')
print(f'Test vs Outage      : {cohens_d(gt[test_rows], gt[outage_rows]):+.3f}')

# ============================================================
# SUPPORT / RANGE AUDIT
# ============================================================
print('\n' + '=' * 80)
print('PREDICTION SUPPORT AUDIT')
print('=' * 80)
print(f'Train CNN range   : {np.min(cnn_tr):.4f} -> {np.max(cnn_tr):.4f}')
print(f'Val CNN range     : {np.min(cnn[val_rows]):.4f} -> {np.max(cnn[val_rows]):.4f}')
print(f'Test CNN range    : {np.min(cnn[test_rows]):.4f} -> {np.max(cnn[test_rows]):.4f}')
print(f'Outage CNN range  : {np.min(cnn[outage_rows]):.4f} -> {np.max(cnn[outage_rows]):.4f}')

train_min = np.min(cnn_tr)
train_max = np.max(cnn_tr)

for name, rows in [
    ('Validation', val_rows),
    ('Test', test_rows),
    ('Outage', outage_rows)
]:
    outside = np.sum(
        (cnn[rows] < train_min) |
        (cnn[rows] > train_max)
    )
    print(
        f'{name:12s}: '
        f'{outside}/{len(rows)} outside training support '
        f'({100*outside/len(rows):.2f}%)'
    )

# ============================================================
# ISOTONIC MAPPING
# ============================================================
print('\n' + '=' * 80)
print('ISOTONIC MAPPING')
print('=' * 80)
x_thresholds = model_e.X_thresholds_
y_thresholds = model_e.y_thresholds_
print(f'Number of isotonic breakpoints: {len(x_thresholds)}')
print(f'First 20 breakpoints:')
for x, y in list(zip(x_thresholds, y_thresholds))[:20]:
    print(f'CNN {x:.4f} -> Calibrated {y:.4f}')

if len(x_thresholds) > 20:
    print('...')
    print('Last 20 breakpoints:')
    for x, y in list(zip(x_thresholds, y_thresholds))[-20:]:
        print(f'CNN {x:.4f} -> Calibrated {y:.4f}')

# ============================================================
# SPEED REGIME ANALYSIS
# ============================================================
bins = [0, 40, 50, 60, 70, 80, np.inf]
labels = ['0-40', '40-50', '50-60', '60-70', '70-80', '80+']
print('\n' + '=' * 80)
print('HIGH-SPEED REGIME ANALYSIS')
print('=' * 80)
for split_name, rows, pred in [
    ('TRAIN', train_rows_clean, pred_tr),
    ('VALIDATION', val_rows, pred_val),
    ('TEST', test_rows, pred_test),
    ('OUTAGE', outage_rows, pred_out)
]:
    print(f'\n{split_name}')
    gt_split = gt[rows]
    cnn_split = cnn[rows]
    for i in range(len(bins)-1):
        mask = (
            (gt_split >= bins[i]) &
            (gt_split < bins[i+1])
        )
        if np.sum(mask) == 0:
            continue
        print(
            f'{labels[i]:>6s} | '
            f'N={np.sum(mask):5d} | '
            f'CNN MAE={mean_absolute_error(gt_split[mask], cnn_split[mask]):6.2f} | '
            f'ISO MAE={mean_absolute_error(gt_split[mask], pred[mask]):6.2f} | '
            f'CNN bias={np.mean(cnn_split[mask]-gt_split[mask]):+7.2f} | '
            f'ISO bias={np.mean(pred[mask]-gt_split[mask]):+7.2f}'
        )
