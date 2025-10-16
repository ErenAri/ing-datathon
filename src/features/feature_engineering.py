import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ChurnFeatureEngineering:
    """
    Advanced feature engineering for ING Hubs Datathon
    Optimized for composite metric (Gini, Recall@10%, Lift@10%)
    """
    
    def __init__(self):
        self.label_encoders = {}
    
    def create_time_windows(self, history_df, ref_date, windows=[1, 3, 6, 12]):
        """
        Create aggregated features for different time windows before ref_date
        Critical for capturing behavior trends
        """
        history_df['date'] = pd.to_datetime(history_df['date'])
        ref_date = pd.to_datetime(ref_date)
        
        features = {}
        
        for window in windows:
            # Filter data for this window
            start_date = ref_date - pd.DateOffset(months=window)
            window_data = history_df[
                (history_df['date'] > start_date) & 
                (history_df['date'] <= ref_date)
            ]
            # Guard: ensure we never use any data after ref_date
            assert (window_data['date'] <= ref_date).all(), "Window data must be <= ref_date"
            
            # Aggregate features
            agg_dict = {
                'mobile_eft_all_cnt': ['sum', 'mean', 'std', 'max'],
                'mobile_eft_all_amt': ['sum', 'mean', 'std', 'max'],
                'cc_transaction_all_amt': ['sum', 'mean', 'std', 'max'],
                'cc_transaction_all_cnt': ['sum', 'mean', 'std', 'max'],
                'active_product_category_nbr': ['mean', 'std', 'min', 'max']
            }
            
            window_features = window_data.groupby('cust_id').agg(agg_dict)
            window_features.columns = [
                f'{col[0]}_{col[1]}_{window}m' 
                for col in window_features.columns
            ]
            
            # Add count of active months in window
            window_features[f'active_months_{window}m'] = (
                window_data.groupby('cust_id').size()
            )
            
            features[window] = window_features
        
        # Combine all windows
        all_features = pd.concat(features.values(), axis=1)
        
        return all_features
    
    def create_trend_features(self, history_df, ref_date):
        """
        Calculate velocity/acceleration features
        These are CRITICAL for top 10% discrimination
        """
        history_df['date'] = pd.to_datetime(history_df['date'])
        ref_date = pd.to_datetime(ref_date)
        
        # Get last 6 months
        start_date = ref_date - pd.DateOffset(months=6)
        recent_data = history_df[
            (history_df['date'] > start_date) & 
            (history_df['date'] <= ref_date)
        ].copy()
        # Guard: ensure we never use any data after ref_date
        assert (recent_data['date'] <= ref_date).all(), "Trend data must be <= ref_date"
        
        # Sort by customer and date
        recent_data = recent_data.sort_values(['cust_id', 'date'])
        
        trend_features = {}
        
        for col in ['mobile_eft_all_amt', 'cc_transaction_all_amt', 
                    'active_product_category_nbr']:
            # Calculate month-over-month change
            recent_data[f'{col}_change'] = (
                recent_data.groupby('cust_id')[col]
                .diff()
            )
            
            # Aggregate trends
            trend_agg = recent_data.groupby('cust_id').agg({
                f'{col}_change': ['mean', 'std', 'min', 'max']
            })
            trend_agg.columns = [
                f'{col}_trend_{stat}' 
                for stat in ['mean', 'std', 'min', 'max']
            ]
            
            trend_features[col] = trend_agg
        
        return pd.concat(trend_features.values(), axis=1)
    
    def create_recency_features(self, history_df, ref_date):
        """
        Recency is HUGE for churn - when was last activity?
        """
        history_df['date'] = pd.to_datetime(history_df['date'])
        ref_date = pd.to_datetime(ref_date)
        
        recency_features = {}
        
        # Days since last transaction
        # Use only history up to ref_date to avoid peeking into the future
        last_dates = history_df[history_df['date'] <= ref_date].groupby('cust_id')['date'].max()
        recency_features['days_since_last_activity'] = (
            (ref_date - last_dates).dt.days
        )

        # Days since first transaction (relative to ref_date)
        # This replaces any absolute day counters and makes the horizon relative
        first_dates = history_df[history_df['date'] <= ref_date].groupby('cust_id')['date'].min()
        recency_features['days_since_first_activity_rel'] = (
            (ref_date - first_dates).dt.days
        )
        
        # Days since last EFT transaction (if any)
        eft_data = history_df[(history_df['date'] <= ref_date) & (history_df['mobile_eft_all_cnt'] > 0)]
        if len(eft_data) > 0:
            last_eft = eft_data.groupby('cust_id')['date'].max()
            recency_features['days_since_last_eft'] = (
                (ref_date - last_eft).dt.days
            )
        
        # Days since last CC transaction
        cc_data = history_df[(history_df['date'] <= ref_date) & (history_df['cc_transaction_all_cnt'] > 0)]
        if len(cc_data) > 0:
            last_cc = cc_data.groupby('cust_id')['date'].max()
            recency_features['days_since_last_cc'] = (
                (ref_date - last_cc).dt.days
            )
        
        return pd.DataFrame(recency_features)
    
    def create_behavioral_patterns(self, history_df, ref_date):
        """
        Create patterns that identify at-risk customers
        """
        history_df['date'] = pd.to_datetime(history_df['date'])
        ref_date = pd.to_datetime(ref_date)
        
        # Last 12 months
        start_date = ref_date - pd.DateOffset(months=12)
        data = history_df[
            (history_df['date'] > start_date) & 
            (history_df['date'] <= ref_date)
        ].copy()
        # Guard: ensure we never use any data after ref_date
        assert (data['date'] <= ref_date).all(), "Behavioral window must be <= ref_date"
        
        patterns = {}
        
        # Activity consistency (coefficient of variation)
        for col in ['mobile_eft_all_cnt', 'cc_transaction_all_cnt']:
            agg = data.groupby('cust_id')[col].agg(['mean', 'std'])
            patterns[f'{col}_cv'] = (agg['std'] / (agg['mean'] + 1))
        
        # Ratio features
        totals = data.groupby('cust_id').agg({
            'mobile_eft_all_cnt': 'sum',
            'cc_transaction_all_cnt': 'sum',
            'mobile_eft_all_amt': 'sum',
            'cc_transaction_all_amt': 'sum'
        })
        
        patterns['eft_to_cc_cnt_ratio'] = (
            totals['mobile_eft_all_cnt'] / 
            (totals['cc_transaction_all_cnt'] + 1)
        )
        
        patterns['eft_to_cc_amt_ratio'] = (
            totals['mobile_eft_all_amt'] / 
            (totals['cc_transaction_all_amt'] + 1)
        )
        
        # Average transaction size
        patterns['avg_eft_size'] = (
            totals['mobile_eft_all_amt'] / 
            (totals['mobile_eft_all_cnt'] + 1)
        )
        
        patterns['avg_cc_size'] = (
            totals['cc_transaction_all_amt'] / 
            (totals['cc_transaction_all_cnt'] + 1)
        )
        
        # Product engagement
        patterns['avg_active_products'] = (
            data.groupby('cust_id')['active_product_category_nbr'].mean()
        )
        
        patterns['max_active_products'] = (
            data.groupby('cust_id')['active_product_category_nbr'].max()
        )
        
        # Dropping products? (red flag)
        # Ensure chronological order when computing first vs last
        sorted_data = data.sort_values(['cust_id', 'date'])
        patterns['product_decline'] = (
            sorted_data.groupby('cust_id')['active_product_category_nbr']
            .apply(lambda s: s.iloc[0] - s.iloc[-1] if len(s) > 1 else 0)
        )
        
        base_df = pd.DataFrame(patterns)

        # ------------------------------------------------------------
        # Advanced decile-focused signals using multi-window stats
        # ------------------------------------------------------------
        eps = 1e-6
        # Compute time-window aggregates (1,3,6,12m) to derive composite signals
        win_feats = self.create_time_windows(history_df, ref_date, windows=[1, 3, 6, 12])

        # Helper to safely fetch a column (return NaN series if missing)
        def _col(name):
            return win_feats[name] if name in win_feats.columns else pd.Series(np.nan, index=win_feats.index)

        # Months since last any activity (from full history up to ref_date)
        last_any = history_df[history_df['date'] <= ref_date].groupby('cust_id')['date'].max()
        days_since = (ref_date - last_any).dt.days
        months_since = days_since / 30.0
        months_since = months_since.reindex(win_feats.index)

        # Variable groups and short aliases for cleaner names
        vars_map = {
            'cc_transaction_all_amt': 'cc_amt',
            'cc_transaction_all_cnt': 'cc_cnt',
            'mobile_eft_all_amt': 'eft_amt',
            'mobile_eft_all_cnt': 'eft_cnt'
        }

        derived = pd.DataFrame(index=win_feats.index)

        for v, alias in vars_map.items():
            m1 = _col(f'{v}_mean_1m')
            m3 = _col(f'{v}_mean_3m')
            m6 = _col(f'{v}_mean_6m')
            m12 = _col(f'{v}_mean_12m')
            s3 = _col(f'{v}_std_3m')
            s6 = _col(f'{v}_std_6m')
            s12 = _col(f'{v}_std_12m')

            # Collapse index: (mean_3m - mean_12m) / (mean_12m + eps)
            derived[f'{alias}_collapse_3v12'] = (m3 - m12) / (m12 + eps)
            # Requested: index_3m_12m
            derived[f'{alias}_index_3m_12m'] = (m3 - m12) / (m12 + eps)

            # Volatility spike: (std_3m - std_12m) / (std_12m + eps)
            derived[f'{alias}_vol_spike_3v12'] = (s3 - s12) / (s12 + eps)
            # Requested naming: vol_spike_6m (using 3m vs 12m as specified)
            derived[f'{alias}_vol_spike_6m'] = (s3 - s12) / (s12 + eps)

            # Personal z: (last_1m - mean_12m) / (1 + std_12m)
            derived[f'{alias}_personal_z'] = (m1 - m12) / (1.0 + s12)

            # Recency-intensity (requested decay=0.2): exp(-0.2 * months_since_last_any) * mean_3m
            derived[f'{alias}_recency_intensity'] = np.exp(-0.2 * months_since) * m3

            # Burstiness (Fano): std_6m**2 / (mean_6m + eps)
            derived[f'{alias}_burstiness_6m'] = (s6 ** 2) / (m6 + eps)
            # Requested naming: burst_fano_6m
            derived[f'{alias}_burst_fano_6m'] = (s6 ** 2) / (m6 + eps)

        # Spend/count mix 3m between CC and EFT
        cc_amt_3 = _col('cc_transaction_all_amt_mean_3m')
        eft_amt_3 = _col('mobile_eft_all_amt_mean_3m')
        cc_cnt_3 = _col('cc_transaction_all_cnt_mean_3m')
        eft_cnt_3 = _col('mobile_eft_all_cnt_mean_3m')
        derived['mix_amt_cc_3m'] = cc_amt_3 / (cc_amt_3 + eft_amt_3 + eps)
        derived['mix_cnt_cc_3m'] = cc_cnt_3 / (cc_cnt_3 + eft_cnt_3 + eps)
        # Requested naming
        derived['mix_cc_eft_amt_3m'] = cc_amt_3 / (cc_amt_3 + eft_amt_3 + eps)
        derived['mix_cc_eft_cnt_3m'] = cc_cnt_3 / (cc_cnt_3 + eft_cnt_3 + eps)
        # Provide a generic alias 'mix_cc_eft_3m' mapping to amount-based mix as requested
        derived['mix_cc_eft_3m'] = derived['mix_cc_eft_amt_3m']

        # Join derived into base patterns
        out = base_df.join(derived, how='outer')
        return out
    
    def create_demographic_features(self, customers_df):
        """
        Encode and engineer demographic features
        """
        demo_features = customers_df.copy()
        
        # Age bins (life stages matter for churn)
        demo_features['age_group'] = pd.cut(
            demo_features['age'], 
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )
        
        # Tenure bins
        demo_features['tenure_group'] = pd.cut(
            demo_features['tenure'],
            bins=[0, 12, 36, 60, 120, 400],
            labels=['0-1y', '1-3y', '3-5y', '5-10y', '10y+']
        )
        
        # Label encoding for categorical features
        cat_cols = ['gender', 'province', 'religion', 'work_type', 
                    'work_sector', 'age_group', 'tenure_group']
        
        for col in cat_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                demo_features[f'{col}_encoded'] = (
                    self.label_encoders[col].fit_transform(
                        demo_features[col].astype(str)
                    )
                )
            else:
                demo_features[f'{col}_encoded'] = (
                    self.label_encoders[col].transform(
                        demo_features[col].astype(str)
                    )
                )
        
        # Keep original numeric features
        demo_features = demo_features[[
            'cust_id', 'age', 'tenure',
            'gender_encoded', 'province_encoded', 'religion_encoded',
            'work_type_encoded', 'work_sector_encoded',
            'age_group_encoded', 'tenure_group_encoded'
        ]]
        
        return demo_features
    
    def create_all_features(self, history_df, customers_df, ref_date):
        """
        Main function to create all features
        """
        print(f"Creating features for ref_date: {ref_date}")
        
        # 1. Time window aggregations
        print("  - Time window features...")
        time_features = self.create_time_windows(history_df, ref_date)
        
        # 2. Trend features
        print("  - Trend features...")
        trend_features = self.create_trend_features(history_df, ref_date)
        
        # 3. Recency features
        print("  - Recency features...")
        recency_features = self.create_recency_features(history_df, ref_date)
        
        # 4. Behavioral patterns
        print("  - Behavioral patterns...")
        behavioral_features = self.create_behavioral_patterns(history_df, ref_date)
        
        # 5. Demographic features
        print("  - Demographic features...")
        demo_features = self.create_demographic_features(customers_df)
        
        # Combine all features
        all_features = pd.concat([
            time_features,
            trend_features,
            recency_features,
            behavioral_features
        ], axis=1)
        
        # Merge with demographics
        all_features = all_features.reset_index()
        all_features = all_features.merge(demo_features, on='cust_id', how='left')

        # Seasonality features based on month-of-year (same for train/test)
        # Avoid using raw ref_date-derived ordinal features; use cyclical encoding instead
        ref_dt = pd.to_datetime(ref_date)
        month = ref_dt.month
        angle = 2 * np.pi * (month - 1) / 12.0
        all_features['month_sin'] = np.sin(angle)
        all_features['month_cos'] = np.cos(angle)
        # Never include ref_date itself numerically
        if 'ref_date' in all_features.columns:
            all_features = all_features.drop(columns=['ref_date'], errors='ignore')
        
        # Fill NaN values
        all_features = all_features.fillna(-999)
        
        print(f"  Total features created: {len(all_features.columns) - 1}")
        
        return all_features


# Example usage:
# fe = ChurnFeatureEngineering()
# 
# # For training data
# train_features = fe.create_all_features(
#     customer_history, 
#     customers, 
#     ref_date='2018-03-01'
# )
# 
# # Merge with labels
# train_data = train_features.merge(reference_data, on='cust_id', how='left')