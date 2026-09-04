import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('/Users/hrithika/Desktop/GHOST/data/processed/ghosttrack_phase6_ekf.csv')
t = df['timestamp'].values
cnn = df['cnn_predicted_speed'].values
gt = df['ground_truth_speed'].values

num_samples = len(df) - 30 + 1
n_train = int(num_samples * 0.70)
train_rows = np.arange(29, n_train + 29)

# Remove outage from calibration data
t_train = t[train_rows]
clean_mask = (t_train < 30.0) | (t_train >= 90.0)
clean_rows = train_rows[clean_mask]

# ============================================================
# TEMPORAL BLOCKS
# ============================================================
n = len(clean_rows)
early_end = int(n * 0.33)
middle_end = int(n * 0.66)

early = clean_rows[:early_end]
middle = clean_rows[early_end:middle_end]
late = clean_rows[middle_end:]

print('=' * 80)
print('TEMPORAL STABILITY AUDIT')
print('=' * 80)
print(f'Clean rows total : {n}')
print(f'Early rows       : {len(early)}')
print(f'Middle rows      : {len(middle)}')
print(f'Late rows        : {len(late)}')

def evaluate(name, fit_rows, eval_rows):
    print(f'\n{name}')
    print('-' * 80)
    X_fit = cnn[fit_rows].reshape(-1,1)
    y_fit = gt[fit_rows]
    X_eval = cnn[eval_rows].reshape(-1,1)
    y_eval = gt[eval_rows]
    # Raw
    raw_mae = mean_absolute_error(y_eval, cnn[eval_rows])
    # Linear
    lin = LinearRegression().fit(X_fit, y_fit)
    p_lin = np.maximum(lin.predict(X_eval), 0)
    # Poly2
    poly2 = make_pipeline(
        PolynomialFeatures(2),
        LinearRegression()
    ).fit(X_fit, y_fit)
    p_poly2 = np.maximum(poly2.predict(X_eval), 0)
    # Isotonic
    iso = IsotonicRegression(
        out_of_bounds='clip'
    ).fit(cnn[fit_rows], gt[fit_rows])
    p_iso = np.maximum(
        iso.predict(cnn[eval_rows]),
        0
    )
    print(f'Raw CNN MAE : {raw_mae:.2f} | Bias: {np.mean(cnn[eval_rows]-y_eval):+.2f}')
    print(f'Linear MAE  : {mean_absolute_error(y_eval,p_lin):.2f} | Bias: {np.mean(p_lin-y_eval):+.2f}')
    print(f'Poly2 MAE   : {mean_absolute_error(y_eval,p_poly2):.2f} | Bias: {np.mean(p_poly2-y_eval):+.2f}')
    print(f'Isotonic MAE: {mean_absolute_error(y_eval,p_iso):.2f} | Bias: {np.mean(p_iso-y_eval):+.2f}')

# Early -> Middle
evaluate(
    'EARLY -> MIDDLE',
    early,
    middle
)

# Early -> Late
evaluate(
    'EARLY -> LATE',
    early,
    late
)

# Early + Middle -> Late
evaluate(
    'EARLY + MIDDLE -> LATE',
    np.concatenate([early,middle]),
    late
)

# Early -> full later period
evaluate(
    'EARLY -> MIDDLE + LATE',
    early,
    np.concatenate([middle,late])
)

print('\n' + '=' * 80)
print('TEMPORAL DISTRIBUTION SUMMARY')
print('=' * 80)
for name, rows in [
    ('EARLY', early),
    ('MIDDLE', middle),
    ('LATE', late)
]:
    print(
        f'{name:8s} | '
        f'N={len(rows):5d} | '
        f'GT mean={np.mean(gt[rows]):6.2f} | '
        f'GT std={np.std(gt[rows]):6.2f} | '
        f'CNN mean={np.mean(cnn[rows]):6.2f} | '
        f'CNN std={np.std(cnn[rows]):6.2f}'
    )
