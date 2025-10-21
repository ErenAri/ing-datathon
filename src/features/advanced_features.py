"""
Advanced Feature Engineering for ING Hubs Datathon

This module provides advanced features beyond basic aggregations:
- RFM (Recency, Frequency, Monetary) Analysis
- Behavioral Change Detection
- Customer Lifecycle Features
- Time-based Activity Patterns

These features help identify at-risk customers and predict churn more accurately.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def validate_no_future_leakage(df: pd.DataFrame, ref_date) -> None:
    """
    Validate that no windowed/time-based data in `df` leaks information
    from at-or-after `ref_date`.

    The validator checks common date-like markers typically produced when
    creating rolling features:
    - "window_end", "end_date", "period_end"
    - generic transaction/event columns like "date"
    - monthly buckets like "month" (Period) or string/object YYYY-MM

    Behavior:
    - If any such column exists, ensure its values (interpreted as the END
      of the period) are strictly less than `ref_date`.
    - If violations are found, raise ValueError detailing offending columns
      and counts.
    - If no date-like columns are found, emit a warning and return.

    Parameters
    ----------
    df : pd.DataFrame
        Frame containing time/rolling derived features or source events.
    ref_date : datetime-like or str
        Reference cutoff date; all window ends must be < this value.
    """
    if df is None or len(df) == 0:
        return

    ref_dt = pd.to_datetime(ref_date)

    candidate_cols = [
        'window_end', 'end_date', 'period_end', 'date', 'month', 'month_end', 'window_end_date'
    ]

    present = [c for c in candidate_cols if c in df.columns]
    if not present:
        # Try to auto-detect Period or datetime-like columns
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_datetime64_any_dtype(s):
                present.append(c)
                continue
            # Lightweight period dtype detection without relying on pandas api typing helpers
            dtype_name = getattr(s.dtype, 'name', '')
            if isinstance(dtype_name, str) and dtype_name.startswith('period'):
                present.append(c)
        if not present:
            warnings.warn("validate_no_future_leakage: no date-like columns found; skipped.")
            return

    violations = []
    for col in present:
        s = df[col]

        # Normalize to datetime end-instants for comparison against ref_dt
        dtype_name = getattr(s.dtype, 'name', '')
        is_period = isinstance(dtype_name, str) and dtype_name.startswith('period')
        if is_period:
            # Series[Period] -> month/period end timestamp
            try:
                ends = s.dt.to_timestamp(how='end')
            except Exception:
                # Fallback to generic to_datetime if .dt path is unavailable
                ends = pd.to_datetime(s.astype(str), errors='coerce')
        else:
            if col == 'month' and s.dtype == 'O':
                # Try parsing YYYY-MM strings to month end timestamps
                try:
                    ends = pd.PeriodIndex(s.astype(str), freq='M').to_timestamp(how='end')
                except Exception:
                    # Fallback to generic to_datetime
                    ends = pd.to_datetime(s, errors='coerce')
            else:
                ends = pd.to_datetime(s, errors='coerce')

        if ends.notna().any():
            bad_mask = ends >= ref_dt
            bad_cnt = int(bad_mask.sum())
            if bad_cnt > 0:
                violations.append({
                    'column': col,
                    'violations': bad_cnt,
                    'max_seen': str(ends.max())
                })

    if violations:
        details = ", ".join([f"{v['column']} (n={v['violations']}, max={v['max_seen']})" for v in violations])
        raise ValueError(
            f"Future leakage detected: window/date values not strictly before ref_date {ref_dt.date()}: {details}"
        )


def _winsorize_trend_ratio_columns(df: pd.DataFrame, lower_pct: float = 1.0, upper_pct: float = 99.0):
    """
    Winsorize trend/ratio-like feature columns in-place and log modifications.

    Columns affected (case-insensitive substring match):
    - contains 'trend'
    - contains 'ratio'
    - contains '_DIV_' (common ratio interaction naming)

    Non-numeric columns are ignored. Columns with insufficient non-NaN values
    or undefined quantiles are skipped. Returns a list of modification logs.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe to modify in-place.
    lower_pct : float
        Lower percentile to clamp at (default 1.0).
    upper_pct : float
        Upper percentile to clamp at (default 99.0).

    Returns
    -------
    List[str]
        Human-readable logs for each column that was modified.
    """
    logs = []
    if df is None or df.empty:
        return logs

    col_names = list(df.columns)
    targets = [
        c for c in col_names
        if isinstance(c, str) and (
            ('trend' in c.lower()) or ('ratio' in c.lower()) or ('_DIV_' in c)
        )
    ]

    for col in targets:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue

        arr = s.to_numpy(dtype=float, copy=False)
        finite_mask = np.isfinite(arr)
        valid = arr[finite_mask]
        if valid.size < 5:
            # Too few values to reliably compute percentiles
            continue

        q_low = np.nanpercentile(valid, lower_pct)
        q_high = np.nanpercentile(valid, upper_pct)
        if not np.isfinite(q_low) or not np.isfinite(q_high):
            continue
        if q_low > q_high:
            # Swap if pathological
            q_low, q_high = q_high, q_low

        clipped = np.clip(arr, q_low, q_high)
        changed = int(np.sum(np.not_equal(arr, clipped)))
        if changed > 0:
            df[col] = clipped
            logs.append(f"winsorized {col}: [{lower_pct}%, {upper_pct}%] -> [{q_low:.4g}, {q_high:.4g}] (changed {changed})")

    return logs


class AdvancedFeatureEngineering:
    """
    Advanced feature engineering for customer churn prediction

    Creates sophisticated features that capture:
    - Customer value and engagement (RFM)
    - Behavioral changes and trends
    - Customer lifecycle stage
    - Temporal patterns
    """

    def __init__(self):
        """Initialize the advanced feature engineering class"""
        self.rfm_segments = {
            'Champions': (4, 5, 4, 5, 4, 5),
            'Loyal Customers': (2, 5, 3, 5, 3, 5),
            'Potential Loyalist': (3, 5, 1, 3, 1, 3),
            'Recent Customers': (4, 5, 0, 1, 0, 1),
            'Promising': (3, 4, 0, 1, 0, 1),
            'Needs Attention': (2, 3, 2, 3, 2, 3),
            'About to Sleep': (2, 3, 0, 2, 0, 2),
            'At Risk': (0, 2, 2, 5, 2, 5),
            'Cannot Lose Them': (0, 1, 4, 5, 4, 5),
            'Hibernating': (1, 2, 1, 2, 1, 2),
            'Lost': (0, 2, 0, 2, 0, 2)
        }

    def create_rfm_features(self, history_df, ref_date):
        """
        Create RFM (Recency, Frequency, Monetary) features

        RFM is a proven method for customer segmentation:
        - Recency: How recently did the customer transact?
        - Frequency: How often do they transact?
        - Monetary: How much do they spend?

        Parameters:
        -----------
        history_df : pd.DataFrame
            Customer transaction history
        ref_date : str
            Reference date for feature calculation

        Returns:
        --------
        pd.DataFrame : RFM features for each customer
        """
        print("  Creating RFM features...")

        ref_date = pd.to_datetime(ref_date)
        history_df = history_df.copy()
        history_df['date'] = pd.to_datetime(history_df['date'])

        # Filter to last 12 months for RFM calculation
        start_date = ref_date - pd.DateOffset(months=12)
        rfm_data = history_df[
            (history_df['date'] > start_date) &
            (history_df['date'] <= ref_date)
        ].copy()

        rfm_features = {}

        # RECENCY: Days since last transaction
        last_transaction = rfm_data.groupby('cust_id')['date'].max()
        rfm_features['rfm_recency_days'] = (ref_date - last_transaction).dt.days

        # FREQUENCY: Total number of transactions in last 12 months
        rfm_features['rfm_frequency_count'] = (
            rfm_data.groupby('cust_id').size()
        )

        # MONETARY: Total transaction value in last 12 months
        # Use both EFT and CC transactions
        rfm_data['total_amount'] = (
            rfm_data['mobile_eft_all_amt'].fillna(0) +
            rfm_data['cc_transaction_all_amt'].fillna(0)
        )
        rfm_features['rfm_monetary_value'] = (
            rfm_data.groupby('cust_id')['total_amount'].sum()
        )

        # Create RFM scores (1-5 scale using quintiles) with robust handling of duplicate bin edges
        rfm_df = pd.DataFrame(rfm_features)

        # Helper to convert qcut categories to dynamic labels safely
        def _qcut_to_labels(series, q=5, reverse=False):
            try:
                cats = pd.qcut(series, q=q, duplicates='drop')
            except ValueError:
                # Fallback: if qcut still fails, use percent rank
                pct = series.rank(pct=True, method='average')
                bins = np.ceil(pct * q).astype(float)
                bins = bins.clip(1, q)
                if reverse:
                    bins = (q - bins + 1)
                return bins.astype(float)

            n_bins = len(cats.cat.categories)
            codes = cats.cat.codes  # -1 indicates NaN
            labels = list(range(1, n_bins + 1))
            if reverse:
                labels = labels[::-1]
            # Map codes -> labels, keep NaN for -1
            codes_arr = codes.to_numpy()
            valid_mask = codes_arr >= 0
            mapping = np.array(labels, dtype=float)
            out = np.full(series.shape[0], np.nan, dtype=float)
            out[valid_mask] = mapping[codes_arr[valid_mask]]
            return pd.Series(out, index=series.index, dtype=float)

        # Recency score (lower is better, so we reverse it)
        rfm_df['rfm_r_score'] = _qcut_to_labels(rfm_df['rfm_recency_days'], q=5, reverse=True)

        # Frequency score (higher is better)
        rfm_df['rfm_f_score'] = _qcut_to_labels(rfm_df['rfm_frequency_count'], q=5, reverse=False)

        # Monetary score (higher is better)
        rfm_df['rfm_m_score'] = _qcut_to_labels(rfm_df['rfm_monetary_value'], q=5, reverse=False)

        # Combined RFM score (weighted average)
        rfm_df['rfm_score'] = (
            0.15 * rfm_df['rfm_r_score'].fillna(1) +
            0.45 * rfm_df['rfm_f_score'].fillna(1) +
            0.40 * rfm_df['rfm_m_score'].fillna(1)
        )

        # RFM segment assignment
        rfm_df['rfm_segment'] = rfm_df.apply(
            lambda x: self._assign_rfm_segment(
                x['rfm_r_score'],
                x['rfm_f_score'],
                x['rfm_m_score']
            ),
            axis=1
        )

        # Encode segment as numeric (higher = better)
        segment_ranking = {
            'Champions': 11,
            'Loyal Customers': 10,
            'Potential Loyalist': 9,
            'Recent Customers': 8,
            'Promising': 7,
            'Needs Attention': 6,
            'About to Sleep': 5,
            'At Risk': 4,
            'Cannot Lose Them': 3,
            'Hibernating': 2,
            'Lost': 1
        }
        rfm_df['rfm_segment_encoded'] = rfm_df['rfm_segment'].map(segment_ranking)

        # Additional RFM-derived features
        rfm_df['rfm_avg_transaction_value'] = (
            rfm_df['rfm_monetary_value'] / (rfm_df['rfm_frequency_count'] + 1)
        )

        rfm_df['rfm_frequency_per_day'] = (
            rfm_df['rfm_frequency_count'] / 365.0
        )

        def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
            denom = denominator.replace(0, np.nan)
            res = numerator / denom
            return res.replace([np.inf, -np.inf], np.nan)

        # Explore additional windows (3M & 6M) for frequency/monetary behaviour
        window_defs = [(3, '3m'), (6, '6m')]
        for months, label in window_defs:
            window_start = ref_date - pd.DateOffset(months=months)
            window_slice = history_df[
                (history_df['date'] > window_start) &
                (history_df['date'] <= ref_date)
            ].copy()
            if window_slice.empty:
                continue

            window_slice['total_amount'] = (
                window_slice['mobile_eft_all_amt'].fillna(0) +
                window_slice['cc_transaction_all_amt'].fillna(0)
            )

            freq_series = window_slice.groupby('cust_id').size()
            amt_series = window_slice.groupby('cust_id')['total_amount'].sum()

            rfm_df[f'rfm_frequency_count_{label}'] = freq_series
            rfm_df[f'rfm_monetary_value_{label}'] = amt_series
            rfm_df[f'rfm_frequency_per_month_{label}'] = freq_series / float(months)
            rfm_df[f'rfm_avg_transaction_value_{label}'] = (
                amt_series / (freq_series + 1)
            )

        # Ratios and deltas vs annual (12M) baseline
        if 'rfm_frequency_count' in rfm_df.columns:
            base_freq = rfm_df['rfm_frequency_count'].astype(float).fillna(0)
            for months, label in window_defs:
                col = f'rfm_frequency_count_{label}'
                if col in rfm_df.columns:
                    rfm_df[f'rfm_frequency_share_{label}'] = _safe_div(
                        rfm_df[col].astype(float),
                        base_freq.replace({0: np.nan})
                    )
                    expected = base_freq * (months / 12.0)
                    rfm_df[f'rfm_frequency_delta_{label}'] = (
                        rfm_df[col].astype(float) - expected
                    )

        if 'rfm_monetary_value' in rfm_df.columns:
            base_amt = rfm_df['rfm_monetary_value'].astype(float).fillna(0)
            for months, label in window_defs:
                col = f'rfm_monetary_value_{label}'
                if col in rfm_df.columns:
                    rfm_df[f'rfm_monetary_share_{label}'] = _safe_div(
                        rfm_df[col].astype(float),
                        base_amt.replace({0: np.nan})
                    )
                    expected_amt = base_amt * (months / 12.0)
                    rfm_df[f'rfm_monetary_delta_{label}'] = (
                        rfm_df[col].astype(float) - expected_amt
                    )

        # Exponentially weighted recency score emphasising very recent activity
        if not rfm_data.empty:
            rfm_temp = rfm_data[['cust_id', 'date']].copy()
            rfm_temp['days_since'] = (ref_date - rfm_temp['date']).dt.days.astype(float)
            decay_constant = 45.0  # ~1.5 month half-life
            rfm_temp['rfm_recency_weight'] = np.exp(-rfm_temp['days_since'] / decay_constant)
            weighted_recency = rfm_temp.groupby('cust_id')['rfm_recency_weight'].sum()
            rfm_df['rfm_recency_exp_weighted'] = weighted_recency

        # Handle missing values
        rfm_df = rfm_df.fillna(-999)

        print(f"    ✓ Created {len(rfm_df.columns)} RFM features")

        return rfm_df

    def _assign_rfm_segment(self, r_score, f_score, m_score):
        """
        Assign RFM segment based on R, F, M scores

        Parameters:
        -----------
        r_score, f_score, m_score : float
            RFM scores (1-5 scale)

        Returns:
        --------
        str : Segment name
        """
        # Handle missing scores
        if pd.isna(r_score) or pd.isna(f_score) or pd.isna(m_score):
            return 'Lost'

        r_score = int(r_score)
        f_score = int(f_score)
        m_score = int(m_score)

        for segment, (r_min, r_max, f_min, f_max, m_min, m_max) in self.rfm_segments.items():
            if (r_min <= r_score <= r_max and
                f_min <= f_score <= f_max and
                m_min <= m_score <= m_max):
                return segment

        return 'Lost'  # Default segment

    def create_behavioral_change_features(self, history_df, ref_date):
        """
        Create behavioral change detection features

        Detects changes in customer behavior by comparing recent vs previous periods.
        Sudden changes often indicate churn risk.

        Parameters:
        -----------
        history_df : pd.DataFrame
            Customer transaction history
        ref_date : str
            Reference date for feature calculation

        Returns:
        --------
        pd.DataFrame : Behavioral change features
        """
        print("  Creating behavioral change features...")

        ref_date = pd.to_datetime(ref_date)
        history_df = history_df.copy()
        history_df['date'] = pd.to_datetime(history_df['date'])

        history_df['total_amt'] = (
            history_df['mobile_eft_all_amt'].fillna(0) +
            history_df['cc_transaction_all_amt'].fillna(0)
        )

        change_features = {}

        window_specs = [('1m', 1), ('3m', 3), ('6m', 6)]

        def _safe_div_series(num: pd.Series, denom: pd.Series) -> pd.Series:
            denom = denom.replace(0, np.nan)
            res = num / denom
            return res.replace([np.inf, -np.inf], np.nan)

        for label, months in window_specs:
            recent_start = ref_date - pd.DateOffset(months=months)
            previous_start = ref_date - pd.DateOffset(months=2 * months)

            # Recent period (last `months`)
            recent_data = history_df[
                (history_df['date'] > recent_start) &
                (history_df['date'] <= ref_date)
            ]

            # Previous period (`months` immediately before the recent window)
            previous_data = history_df[
                (history_df['date'] > previous_start) &
                (history_df['date'] <= recent_start)
            ]

            # Transaction count change
            recent_txn_count = recent_data.groupby('cust_id').size()
            previous_txn_count = previous_data.groupby('cust_id').size()

            change_features[f'behavior_txn_count_recent_{label}'] = recent_txn_count
            change_features[f'behavior_txn_count_previous_{label}'] = previous_txn_count
            change_features[f'behavior_txn_count_change_pct_{label}'] = (
                (recent_txn_count - previous_txn_count) / (previous_txn_count + 1) * 100
            )
            change_features[f'behavior_txn_count_change_ratio_{label}'] = _safe_div_series(
                recent_txn_count.astype(float), previous_txn_count.astype(float) + 1e-9
            )

            # Transaction amount change
            recent_amt = recent_data.groupby('cust_id')['total_amt'].sum()
            previous_amt = previous_data.groupby('cust_id')['total_amt'].sum()

            change_features[f'behavior_amt_recent_{label}'] = recent_amt
            change_features[f'behavior_amt_previous_{label}'] = previous_amt
            change_features[f'behavior_amt_change_pct_{label}'] = (
                (recent_amt - previous_amt) / (previous_amt + 1) * 100
            )
            change_features[f'behavior_amt_change_ratio_{label}'] = _safe_div_series(
                recent_amt.astype(float), previous_amt.astype(float) + 1e-9
            )

            # Product usage change
            recent_products = recent_data.groupby('cust_id')['active_product_category_nbr'].mean()
            previous_products = previous_data.groupby('cust_id')['active_product_category_nbr'].mean()

            change_features[f'behavior_products_recent_{label}'] = recent_products
            change_features[f'behavior_products_previous_{label}'] = previous_products
            change_features[f'behavior_products_change_pct_{label}'] = (
                (recent_products - previous_products) / (previous_products + 1) * 100
            )
            change_features[f'behavior_products_change_ratio_{label}'] = _safe_div_series(
                recent_products.astype(float),
                previous_products.astype(float) + 1e-9
            )

            # Preserve legacy column names for 3-month window (backwards compatibility)
            if label == '3m':
                change_features['behavior_txn_count_recent'] = recent_txn_count
                change_features['behavior_txn_count_previous'] = previous_txn_count
                change_features['behavior_txn_count_change_pct'] = change_features[f'behavior_txn_count_change_pct_{label}']

                change_features['behavior_amt_recent'] = recent_amt
                change_features['behavior_amt_previous'] = previous_amt
                change_features['behavior_amt_change_pct'] = change_features[f'behavior_amt_change_pct_{label}']

                change_features['behavior_products_recent'] = recent_products
                change_features['behavior_products_previous'] = previous_products
                change_features['behavior_products_change_pct'] = change_features[f'behavior_products_change_pct_{label}']

        change_df = pd.DataFrame(change_features)

        # Cross-window comparisons to detect fast deterioration vs gradual decline
        if {'behavior_txn_count_recent_1m', 'behavior_txn_count_recent_6m'}.issubset(change_df.columns):
            monthly_normalizer = change_df['behavior_txn_count_recent_6m'] / 6.0
            change_df['behavior_txn_recent_share_1m_over_6m'] = _safe_div_series(
                change_df['behavior_txn_count_recent_1m'],
                monthly_normalizer
            )

        if {'behavior_amt_recent_1m', 'behavior_amt_recent_6m'}.issubset(change_df.columns):
            amt_normalizer = change_df['behavior_amt_recent_6m'] / 6.0
            change_df['behavior_amt_recent_share_1m_over_6m'] = _safe_div_series(
                change_df['behavior_amt_recent_1m'],
                amt_normalizer
            )

        # Activity trend classification
        def classify_trend(row):
            """Classify if customer activity is increasing, decreasing, or stable"""
            txn_change = row.get('behavior_txn_count_change_pct_3m', row.get('behavior_txn_count_change_pct', 0))
            amt_change = row.get('behavior_amt_change_pct_3m', row.get('behavior_amt_change_pct', 0))
            one_month_change = row.get('behavior_txn_count_change_pct_1m', np.nan)

            if pd.isna(txn_change) or pd.isna(amt_change):
                return 0  # Unknown

            # Decreasing: significant drop in both metrics
            if (
                txn_change < -20 or amt_change < -20 or
                (not pd.isna(one_month_change) and one_month_change < -30)
            ):
                return -1  # Decreasing (warning sign)
            # Increasing: significant increase
            elif txn_change > 20 or amt_change > 20:
                return 1  # Increasing
            else:
                return 0  # Stable

        change_df['behavior_activity_trend'] = change_df.apply(classify_trend, axis=1)

        # Volatility score (how much behavior is changing)
        vol_components = []
        for label, _ in window_specs:
            col_txn = f'behavior_txn_count_change_pct_{label}'
            col_amt = f'behavior_amt_change_pct_{label}'
            if col_txn in change_df.columns:
                vol_components.append(change_df[col_txn].abs().fillna(0))
            if col_amt in change_df.columns:
                vol_components.append(change_df[col_amt].abs().fillna(0))

        if vol_components:
            change_df['behavior_volatility_score'] = pd.concat(vol_components, axis=1).mean(axis=1)
        else:
            change_df['behavior_volatility_score'] = 0.0

        # Handle missing values
        change_df = change_df.fillna(-999)

        print(f"    ✓ Created {len(change_df.columns)} behavioral change features")

        return change_df

    def create_lifecycle_features(self, history_df, customers_df, ref_date):
        """
        Create customer lifecycle features

        Captures where the customer is in their journey with the bank.
        Different lifecycle stages have different churn patterns.

        Parameters:
        -----------
        history_df : pd.DataFrame
            Customer transaction history
        customers_df : pd.DataFrame
            Customer demographic data
        ref_date : str
            Reference date for feature calculation

        Returns:
        --------
        pd.DataFrame : Lifecycle features
        """
        print("  Creating customer lifecycle features...")

        ref_date = pd.to_datetime(ref_date)
        history_df = history_df.copy()
        history_df['date'] = pd.to_datetime(history_df['date'])

        lifecycle_features = {}

        # Account age (already have tenure in months, calculate in days)
        customers_subset = customers_df.copy()
        lifecycle_features['lifecycle_tenure_days'] = customers_subset.set_index('cust_id')['tenure'] * 30

        # Days since first transaction
        first_transaction = history_df.groupby('cust_id')['date'].min()
        lifecycle_features['lifecycle_days_since_first_txn'] = (
            (ref_date - first_transaction).dt.days
        )

        # Engagement intensity (transactions per month of tenure)
        total_txns = history_df.groupby('cust_id').size()
        tenure_months = customers_subset.set_index('cust_id')['tenure']
        lifecycle_features['lifecycle_txn_per_month'] = (
            total_txns / (tenure_months + 1)
        )

        # Product adoption rate (unique products per year)
        # Using product category as proxy
        last_12m = ref_date - pd.DateOffset(months=12)
        recent_history = history_df[history_df['date'] > last_12m]

        avg_products_last_12m = recent_history.groupby('cust_id')['active_product_category_nbr'].mean()
        lifecycle_features['lifecycle_products_per_year'] = avg_products_last_12m

        # Engagement intensity by shorter windows to capture lifecycle shifts
        rolling_windows = [(3, '3m'), (6, '6m')]
        for months, label in rolling_windows:
            window_start = ref_date - pd.DateOffset(months=months)
            window_slice = history_df[history_df['date'] > window_start]

            monthly_txn_count = window_slice.groupby('cust_id').size()
            lifecycle_features[f'lifecycle_txn_per_month_{label}'] = (
                monthly_txn_count / float(months)
            )
            lifecycle_features[f'lifecycle_txn_share_{label}'] = (
                monthly_txn_count / (total_txns + 1)
            )

            avg_products_window = window_slice.groupby('cust_id')['active_product_category_nbr'].mean()
            lifecycle_features[f'lifecycle_products_avg_{label}'] = avg_products_window

        # Activity consistency score (coefficient of variation)
        # Lower CV = more consistent, higher CV = more volatile
        monthly_txns = history_df.groupby([
            'cust_id',
            history_df['date'].dt.to_period('M')
        ]).size().unstack(fill_value=0)

        lifecycle_features['lifecycle_activity_consistency'] = (
            1 / (monthly_txns.std(axis=1) / (monthly_txns.mean(axis=1) + 1) + 1)
        )

        # Peak activity detection
        # When was the customer most active?
        last_6m = ref_date - pd.DateOffset(months=6)
        peak_data = history_df[history_df['date'] > last_6m].copy()

        peak_data['month'] = peak_data['date'].dt.to_period('M')
        monthly_activity = peak_data.groupby(['cust_id', 'month']).size().reset_index(name='count')

        peak_month = monthly_activity.loc[
            monthly_activity.groupby('cust_id')['count'].idxmax()
        ]
        peak_month['month_date'] = peak_month['month'].dt.to_timestamp()

        lifecycle_features['lifecycle_days_since_peak'] = (
            ref_date - peak_month.set_index('cust_id')['month_date']
        ).dt.days

        # Multi-window lifecycle trends (6M vs previous 6M) to capture momentum shifts
        last_6m_start = ref_date - pd.DateOffset(months=6)
        prev_6m_start = ref_date - pd.DateOffset(months=12)

        last_6m_data = history_df[
            (history_df['date'] > last_6m_start) &
            (history_df['date'] <= ref_date)
        ]
        prev_6m_data = history_df[
            (history_df['date'] > prev_6m_start) &
            (history_df['date'] <= last_6m_start)
        ]

        last_6m_txn = last_6m_data.groupby('cust_id').size()
        prev_6m_txn = prev_6m_data.groupby('cust_id').size()

        lifecycle_features['lifecycle_txn_trend_6m'] = (
            (last_6m_txn - prev_6m_txn) / (prev_6m_txn + 1)
        )

        last_6m_amt = last_6m_data.groupby('cust_id')[
            ['mobile_eft_all_amt', 'cc_transaction_all_amt']
        ].sum().sum(axis=1)
        prev_6m_amt = prev_6m_data.groupby('cust_id')[
            ['mobile_eft_all_amt', 'cc_transaction_all_amt']
        ].sum().sum(axis=1)
        lifecycle_features['lifecycle_amt_trend_6m'] = (
            (last_6m_amt - prev_6m_amt) / (prev_6m_amt + 1)
        )

        # Customer lifecycle stage
        def determine_lifecycle_stage(row):
            """Determine lifecycle stage based on tenure and activity"""
            tenure_days = row.get('lifecycle_tenure_days', 0)
            txn_per_month = row.get('lifecycle_txn_per_month', 0)

            if pd.isna(tenure_days) or pd.isna(txn_per_month):
                return 0

            # New customer (< 6 months)
            if tenure_days < 180:
                return 1 if txn_per_month > 2 else 2  # Active New / Inactive New
            # Growing customer (6-24 months)
            elif tenure_days < 730:
                return 3 if txn_per_month > 3 else 4  # Growing / Stagnant
            # Mature customer (> 24 months)
            else:
                return 5 if txn_per_month > 2 else 6  # Mature Active / Mature Inactive

        lifecycle_df = pd.DataFrame(lifecycle_features)
        lifecycle_df['lifecycle_stage'] = lifecycle_df.apply(determine_lifecycle_stage, axis=1)

        # Handle missing values
        lifecycle_df = lifecycle_df.fillna(-999)

        print(f"    ✓ Created {len(lifecycle_df.columns)} lifecycle features")

        return lifecycle_df

    def create_time_based_features(self, history_df, customers_df, ref_date):
        """
        Create time-based activity features

        Captures temporal patterns like inactivity periods and activity decay.

        Parameters:
        -----------
        history_df : pd.DataFrame
            Customer transaction history
        customers_df : pd.DataFrame
            Customer demographic data
        ref_date : str
            Reference date for feature calculation

        Returns:
        --------
        pd.DataFrame : Time-based features
        """
        print("  Creating time-based features...")

        ref_date = pd.to_datetime(ref_date)
        history_df = history_df.copy()
        history_df['date'] = pd.to_datetime(history_df['date'])

        time_features = {}

        # Days since account opening (similar to tenure but in days)
        customers_subset = customers_df.set_index('cust_id')
        time_features['time_days_since_account_open'] = customers_subset['tenure'] * 30

        # Days since first transaction
        first_txn = history_df.groupby('cust_id')['date'].min()
        time_features['time_days_since_first_txn'] = (ref_date - first_txn).dt.days

        # Days since last transaction
        last_txn = history_df.groupby('cust_id')['date'].max()
        time_features['time_days_since_last_txn'] = (ref_date - last_txn).dt.days

        # Months of inactivity in last year
        # Count months with zero transactions in the last 12 months
        last_12m = ref_date - pd.DateOffset(months=12)
        recent_history = history_df[history_df['date'] > last_12m].copy()

        # Active months in recent 12 months
        recent_history['month'] = recent_history['date'].dt.to_period('M')
        active_months = recent_history.groupby('cust_id')['month'].nunique()
        time_features['time_inactive_months_last_12m'] = (12 - active_months).clip(lower=0, upper=12)

        # Consecutive inactive months (current streak) - vectorized
        # For customers with no recent activity, streak = 12
        last_recent_txn = recent_history.groupby('cust_id')['date'].max()
        # Compute months difference between ref_date and last recent txn
        months_diff = (ref_date.year - last_recent_txn.dt.year) * 12 + (ref_date.month - last_recent_txn.dt.month)
        inactive_streaks = months_diff.clip(lower=0).fillna(12).clip(upper=12)
        time_features['time_consecutive_inactive_months'] = inactive_streaks

        # Shorter windows (1M, 3M, 6M) for activity surveillance
        activity_windows = [(1, '1m'), (3, '3m'), (6, '6m')]
        recent_12m_counts = recent_history.groupby('cust_id').size()

        for months, label in activity_windows:
            window_start = ref_date - pd.DateOffset(months=months)
            window_slice = history_df[
                (history_df['date'] > window_start) &
                (history_df['date'] <= ref_date)
            ].copy()

            txn_counts = window_slice.groupby('cust_id').size()
            time_features[f'time_txn_count_last_{label}'] = txn_counts
            time_features[f'time_txn_per_month_last_{label}'] = (
                txn_counts / float(max(months, 1))
            )

            window_slice['month'] = window_slice['date'].dt.to_period('M')
            active_months_window = window_slice.groupby('cust_id')['month'].nunique()
            time_features[f'time_active_months_last_{label}'] = active_months_window
            time_features[f'time_inactive_months_last_{label}'] = (
                months - active_months_window
            ).clip(lower=0)

            if months > 0:
                denominator = recent_12m_counts / 12.0
                time_features[f'time_activity_share_last_{label}'] = (
                    txn_counts / (denominator + 1e-9)
                )

            if not window_slice.empty:
                sorted_slice = window_slice.sort_values(['cust_id', 'date'])
                gaps_window = sorted_slice.groupby('cust_id')['date'].diff().dt.days
                avg_gap_window = gaps_window.groupby(sorted_slice['cust_id']).mean()
                time_features[f'time_avg_days_between_txn_last_{label}'] = avg_gap_window

        # Activity decay rate
        # Compare activity in recent 3 months vs 6-9 months ago
        recent_3m = ref_date - pd.DateOffset(months=3)
        old_6m = ref_date - pd.DateOffset(months=9)
        old_3m = ref_date - pd.DateOffset(months=6)

        recent_count = history_df[
            (history_df['date'] > recent_3m) & (history_df['date'] <= ref_date)
        ].groupby('cust_id').size()

        old_count = history_df[
            (history_df['date'] > old_6m) & (history_df['date'] <= old_3m)
        ].groupby('cust_id').size()

        time_features['time_activity_decay_rate'] = (
            (old_count - recent_count) / (old_count + 1)
        )

        # Capture acceleration by comparing 1M vs 3M momentum
        if 'time_txn_count_last_1m' in time_features and 'time_txn_count_last_3m' in time_features:
            denom = (time_features['time_txn_count_last_3m'] / 3.0)
            time_features['time_activity_acceleration_1m_vs_3m'] = (
                (time_features['time_txn_count_last_1m'] / (denom + 1e-9)) - 1
            )

        # Time since peak activity
        # Find month with highest activity and calculate days since
        all_history = history_df.copy()
        all_history['month'] = all_history['date'].dt.to_period('M')
        monthly_counts = all_history.groupby(['cust_id', 'month']).size().reset_index(name='count')

        peak_months = monthly_counts.loc[
            monthly_counts.groupby('cust_id')['count'].idxmax()
        ]
        peak_months['peak_date'] = peak_months['month'].dt.to_timestamp()

        time_features['time_days_since_peak_activity'] = (
            ref_date - peak_months.set_index('cust_id')['peak_date']
        ).dt.days

        # Average gap between transactions (vectorized)
        if not history_df.empty:
            sorted_hist = history_df.sort_values(['cust_id', 'date'])
            gaps = sorted_hist.groupby('cust_id')['date'].diff().dt.days
            avg_gaps = gaps.groupby(sorted_hist['cust_id']).mean()
        else:
            avg_gaps = pd.Series(dtype=float)
        time_features['time_avg_days_between_txn'] = avg_gaps.fillna(-999)

        # Time between the last two observed transactions
        last_txn_dates = history_df.groupby('cust_id')['date'].max()
        second_last_txn = history_df.sort_values(['cust_id', 'date']).groupby('cust_id')['date'].nth(-2)
        time_features['time_days_between_last_two_txn'] = (
            (last_txn_dates - second_last_txn).dt.days
        )

        time_df = pd.DataFrame(time_features)

        # Handle missing values
        time_df = time_df.fillna(-999)

        print(f"    ✓ Created {len(time_df.columns)} time-based features")

        return time_df

    def create_all_advanced_features(self, history_df, customers_df, ref_date):
        """
        Create all advanced features

        Main method that combines all advanced feature engineering:
        - RFM Analysis
        - Behavioral Change Detection
        - Customer Lifecycle Features
        - Time-based Features

        Parameters:
        -----------
        history_df : pd.DataFrame
            Customer transaction history
        customers_df : pd.DataFrame
            Customer demographic data
        ref_date : str
            Reference date for feature calculation

        Returns:
        --------
        pd.DataFrame : All advanced features combined
        """
        print(f"\nCreating advanced features for ref_date: {ref_date}")

        # 1. RFM Features
        rfm_features = self.create_rfm_features(history_df, ref_date)

        # 2. Behavioral Change Features
        behavioral_features = self.create_behavioral_change_features(history_df, ref_date)

        # 3. Lifecycle Features
        lifecycle_features = self.create_lifecycle_features(history_df, customers_df, ref_date)

        # 4. Time-based Features
        time_features = self.create_time_based_features(history_df, customers_df, ref_date)

        # Combine all features
        all_features = pd.concat([
            rfm_features,
            behavioral_features,
            lifecycle_features,
            time_features
        ], axis=1)

        # Reset index to have cust_id as column
        all_features = all_features.reset_index()
        if 'index' in all_features.columns:
            all_features = all_features.rename(columns={'index': 'cust_id'})

        # Winsorize trend/ratio-like features to reduce outlier impact
        win_logs = _winsorize_trend_ratio_columns(all_features, lower_pct=1.0, upper_pct=99.0)
        if win_logs:
            print(f"    ✓ Winsorized {len(win_logs)} trend/ratio columns:")
            for msg in win_logs[:20]:  # avoid overly verbose logs
                print(f"      - {msg}")
            if len(win_logs) > 20:
                print(f"      ... and {len(win_logs) - 20} more")

        # Final cleanup
        all_features = all_features.fillna(-999)
        all_features = all_features.replace([np.inf, -np.inf], -999)

        print(f"\n✓ Total advanced features created: {len(all_features.columns) - 1}")
        print(f"✓ Customers processed: {len(all_features)}")

        return all_features


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("ADVANCED FEATURE ENGINEERING - EXAMPLE USAGE")
    print("="*60)

    # This is a demonstration of how to use the class
    # In practice, you would load your actual data

    print("\nExample initialization:")
    print("  fe_advanced = AdvancedFeatureEngineering()")
    print("  advanced_features = fe_advanced.create_all_advanced_features(")
    print("      history_df, customers_df, ref_date='2018-03-01'")
    print("  )")
    print("\nFeatures created:")
    print("  - RFM Analysis (recency, frequency, monetary, segments)")
    print("  - Behavioral Change Detection (trend analysis)")
    print("  - Customer Lifecycle (tenure, engagement, stage)")
    print("  - Time-based Features (inactivity, decay rates)")
    print("\n" + "="*60)
