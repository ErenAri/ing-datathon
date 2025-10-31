"""
Performance Features for Top 20 Push

8 carefully selected features targeting Recall@10 and Lift@10 (60% of competition metric).
Uses actual customer_history columns: mobile_eft_all_cnt, mobile_eft_all_amt, cc_transaction_all_amt,
cc_transaction_all_cnt, active_product_category_nbr
"""

import numpy as np
import pandas as pd


class PerformanceFeatureEngineering:
    """Lean set of 8 high-impact features for top-decile churn prediction."""

    def create_all_performance_features(
        self,
        history_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        ref_date: str
    ) -> pd.DataFrame:
        """
        Create 8 performance features optimized for Recall@10 and Lift@10.
        Returns DataFrame with cust_id + 8 features.
        """
        ref_date_dt = pd.to_datetime(ref_date)
        history = history_df[pd.to_datetime(history_df['date']) <= ref_date_dt].copy()

        # Calculate total transaction counts and amounts
        history['total_cnt'] = history[['mobile_eft_all_cnt', 'cc_transaction_all_cnt']].fillna(0).sum(axis=1)
        history['total_amt'] = history[['mobile_eft_all_amt', 'cc_transaction_all_amt']].fillna(0).sum(axis=1)

        result = customers_df[['cust_id']].copy()

        # === FEATURE 1: Recent Activity Drop Ratio ===
        recent_3m = history[history['date'] >= (ref_date_dt - pd.DateOffset(months=3))]
        older_3m = history[(history['date'] >= (ref_date_dt - pd.DateOffset(months=6))) &
                           (history['date'] < (ref_date_dt - pd.DateOffset(months=3)))]

        recent_txn = recent_3m.groupby('cust_id')['total_cnt'].sum()
        older_txn = older_3m.groupby('cust_id')['total_cnt'].sum()

        result['activity_drop_ratio'] = result['cust_id'].map(
            lambda x: recent_txn.get(x, 0) / max(older_txn.get(x, 0), 1)
        ).fillna(1.0)

        # === FEATURE 2: Days Since Last Transaction ===
        last_dates = history.groupby('cust_id')['date'].max()
        result['days_since_last_txn'] = result['cust_id'].map(
            lambda x: (ref_date_dt - pd.to_datetime(last_dates.get(x, ref_date))).days
        ).fillna(365)

        # === FEATURE 3: Balance Trend (Simple) ===
        history['month'] = pd.to_datetime(history['date']).dt.to_period('M')
        monthly_amt = history.groupby(['cust_id', 'month'])['total_amt'].sum().reset_index()

        trends = {}
        for cust_id in monthly_amt['cust_id'].unique():
            cust_data = monthly_amt[monthly_amt['cust_id'] == cust_id].sort_values('month')
            if len(cust_data) >= 3:
                x = np.arange(len(cust_data))
                y = cust_data['total_amt'].values
                try:
                    trend = np.polyfit(x, y, 1)[0]
                    trends[cust_id] = trend
                except:
                    trends[cust_id] = 0.0
            else:
                trends[cust_id] = 0.0

        result['balance_trend'] = result['cust_id'].map(trends).fillna(0.0)

        # === FEATURE 4: Product Abandonment ===
        products_6m = history[history['date'] >= (ref_date_dt - pd.DateOffset(months=6))].groupby('cust_id')['active_product_category_nbr'].nunique()
        products_3m = history[history['date'] >= (ref_date_dt - pd.DateOffset(months=3))].groupby('cust_id')['active_product_category_nbr'].nunique()

        result['product_abandonment'] = result['cust_id'].map(
            lambda x: 1 - (products_3m.get(x, 0) / max(products_6m.get(x, 0), 1))
        ).fillna(0.0)

        # === FEATURE 5: Transaction Irregularity ===
        txn_stats = history.groupby('cust_id')['total_cnt'].agg(['mean', 'std'])
        result['txn_irregularity'] = result['cust_id'].map(
            lambda x: txn_stats.loc[x, 'std'] / (txn_stats.loc[x, 'mean'] + 1e-9) if x in txn_stats.index else 0.0
        ).fillna(0.0)

        # === FEATURE 6: Low Tenure with High Activity ===
        first_dates = history.groupby('cust_id')['date'].min()
        tenure_days = result['cust_id'].map(
            lambda x: (ref_date_dt - pd.to_datetime(first_dates.get(x, ref_date))).days
        ).fillna(0)

        total_txn = history.groupby('cust_id')['total_cnt'].sum()
        median_txn = total_txn.median()

        result['new_joiner_risk'] = ((tenure_days < 180) & (result['cust_id'].map(total_txn).fillna(0) > median_txn)).astype(float)

        # === FEATURE 7: Consecutive Inactive Months ===
        last_6_months = pd.period_range(end=ref_date_dt.to_period('M'), periods=6, freq='M')
        active_months_by_cust = history.groupby('cust_id')['month'].apply(set).to_dict()

        def count_consecutive_inactive(cust_id):
            active_months = active_months_by_cust.get(cust_id, set())
            consecutive = 0
            for month in reversed(last_6_months):
                if month not in active_months:
                    consecutive += 1
                else:
                    break
            return consecutive

        result['consecutive_inactive_months'] = result['cust_id'].map(count_consecutive_inactive).fillna(0)

        # === FEATURE 8: Below-Peer Performance ===
        tenure_brackets = pd.cut(tenure_days, bins=[0, 90, 180, 365, 730, 10000], labels=['0-3m', '3-6m', '6-12m', '1-2y', '2y+'])
        total_amt_by_cust = result['cust_id'].map(history.groupby('cust_id')['total_amt'].sum()).fillna(0)

        below_peer = np.zeros(len(result))
        for bracket in ['0-3m', '3-6m', '6-12m', '1-2y', '2y+']:
            mask = (tenure_brackets == bracket)
            if mask.sum() > 10:
                median_amt = total_amt_by_cust[mask].median()
                below_peer[mask] = (total_amt_by_cust[mask] < median_amt * 0.5).astype(float)

        result['below_peer_flag'] = below_peer

        # Fill remaining NaNs
        for col in result.columns:
            if col != 'cust_id':
                result[col] = result[col].fillna(0.0)

        print(f"  [PERF] Created 8 performance features for {len(result)} customers")

        return result
