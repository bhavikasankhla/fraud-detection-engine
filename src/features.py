import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time features
    df['hour']        = (df['TransactionDT'] // 3600) % 24
    df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7
    df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night']    = df['hour'].isin(range(0, 6)).astype(int)

    # Amount features
    df['amt_log']      = np.log1p(df['TransactionAmt'])
    df['amt_cents']    = df['TransactionAmt'] % 1
    df['amt_is_round'] = (df['TransactionAmt'] % 10 == 0).astype(int)
    df['amt_bin']      = pd.cut(
        df['TransactionAmt'],
        bins=[0, 10, 50, 200, 500, 10000],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    # Card aggregates
    for col in ['card1', 'card2']:
        if col in df.columns:
            grp = df.groupby(col)['TransactionAmt']
            df[f'{col}_amt_mean']   = df[col].map(grp.mean())
            df[f'{col}_amt_std']    = df[col].map(grp.std()).fillna(0)
            df[f'{col}_txn_count']  = df[col].map(grp.count())
            df[f'{col}_amt_zscore'] = (
                (df['TransactionAmt'] - df[f'{col}_amt_mean']) /
                (df[f'{col}_amt_std'] + 1e-9)
            )

    # Email domain features
    for col in ['P_emaildomain', 'R_emaildomain']:
        if col in df.columns:
            risky = ['gmail', 'yahoo', 'hotmail', 'outlook']
            df[f'{col}_is_free'] = df[col].fillna('').apply(
                lambda x: 1 if any(r in str(x) for r in risky) else 0
            )

    # Address match
    if 'addr1' in df.columns and 'addr2' in df.columns:
        df['addr_match'] = (df['addr1'] == df['addr2']).astype(int)
        df['addr1_null'] = df['addr1'].isnull().astype(int)

    # V-feature summaries
    v_cols = [c for c in df.columns if c.startswith('V')]
    if v_cols:
        v_sub = df[v_cols].fillna(-999)
        df['v_mean']       = v_sub.mean(axis=1)
        df['v_std']        = v_sub.std(axis=1)
        df['v_sum']        = v_sub.sum(axis=1)
        df['v_null_count'] = df[v_cols].isnull().sum(axis=1)

    # C-feature summaries
    c_cols = [c for c in df.columns if c.startswith('C')]
    if c_cols:
        df['c_mean'] = df[c_cols].fillna(0).mean(axis=1)
        df['c_sum']  = df[c_cols].fillna(0).sum(axis=1)

    return df


def align_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Ensures input dataframe has exactly the columns the model
    was trained on — adds missing ones as 0, drops extras.
    """
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names].fillna(0)


from sklearn.preprocessing import LabelEncoder

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode any remaining object columns — same as training pipeline."""
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].fillna('missing')
        df[col] = le.fit_transform(df[col].astype(str))
    return df