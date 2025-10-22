"""
Feature Analysis for ING Datathon
==================================

This module provides comprehensive feature importance analysis and optimization
recommendations. It analyzes feature importance across multiple models, identifies
redundant features, suggests new interactions, and visualizes feature patterns.

Features:
- Multi-model feature importance aggregation
- Temporal importance trend analysis
- Correlation and redundancy detection
- Feature categorization and analysis
- Actionable optimization recommendations
- Rich visualizations

Author: ING Datathon Team
Date: 2025-10-12
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import json
import warnings
warnings.filterwarnings('ignore')


class FeatureAnalyzer:
    """
    Comprehensive Feature Analysis System

    This class analyzes feature importance across multiple models and runs,
    identifies patterns, detects redundancies, and provides actionable
    recommendations for feature engineering optimization.

    Attributes:
        importance_files (List[Path]): List of feature importance CSV files
        importance_data (List[pd.DataFrame]): Loaded importance data
        feature_importance_agg (pd.DataFrame): Aggregated importance across models
        correlation_matrix (pd.DataFrame): Feature correlation matrix
        feature_categories (Dict): Feature categorization
    """

    def __init__(self):
        """Initialize FeatureAnalyzer."""
        self.importance_files = []
        self.importance_data = []
        self.feature_importance_agg = None
        self.correlation_matrix = None
        self.feature_categories = {}
        self.X_train = None  # For correlation analysis

        print(f"\n{'='*80}")
        print(f"FEATURE ANALYZER INITIALIZED")
        print(f"{'='*80}\n")

    def load_importance_files(self,
                             directory: str = '.',
                             pattern: str = 'feature_importance*.csv') -> None:
        """
        Load feature importance CSV files from directory.

        Args:
            directory: Directory containing feature importance files
            pattern: Glob pattern to match files
        """
        dir_path = Path(directory)
        self.importance_files = sorted(dir_path.glob(pattern))

        if not self.importance_files:
            print(f"[WARN] No feature importance files found matching pattern: {pattern}")
            return

        print(f"\n{'='*80}")
        print(f"LOADING FEATURE IMPORTANCE FILES")
        print(f"{'='*80}\n")

        for file_path in self.importance_files:
            try:
                df = pd.read_csv(file_path)

                # Validate columns
                if 'feature' not in df.columns or 'importance' not in df.columns:
                    print(f"  [WARN] Skipping {file_path.name}: Missing required columns")
                    continue

                # Add metadata
                df['source_file'] = file_path.name
                df['model_name'] = self._extract_model_name(file_path.name)

                self.importance_data.append(df)
                print(f"  [OK] Loaded {file_path.name}: {len(df)} features")

            except Exception as e:
                print(f"  ❌ Error loading {file_path.name}: {str(e)}")

        print(f"\nTotal files loaded: {len(self.importance_data)}")
        print(f"{'='*80}\n")

    def _extract_model_name(self, filename: str) -> str:
        """Extract model name from filename."""
        # Examples: feature_importance_lgb.csv -> lgb
        #           feature_importance_xgb_run2.csv -> xgb
        name = filename.replace('feature_importance_', '').replace('.csv', '')

        for model_type in ['lgb', 'xgb', 'cat', 'catboost', 'two_stage', 'stacking']:
            if model_type in name.lower():
                return model_type

        return name

    def aggregate_importance(self,
                           method: str = 'mean',
                           top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Aggregate feature importance across all loaded files.

        Args:
            method: Aggregation method ('mean', 'median', 'max', 'min')
            top_n: Return top N features only (optional)

        Returns:
            DataFrame with aggregated feature importance
        """
        if not self.importance_data:
            print("No importance data loaded. Call load_importance_files() first.")
            return pd.DataFrame()

        print(f"\n{'='*80}")
        print(f"AGGREGATING FEATURE IMPORTANCE")
        print(f"{'='*80}")
        print(f"Aggregation method: {method}")
        print(f"Number of models: {len(self.importance_data)}")
        print(f"{'='*80}\n")

        # Combine all importance data
        all_importance = []

        for df in self.importance_data:
            model_name = df['model_name'].iloc[0]
            for _, row in df.iterrows():
                all_importance.append({
                    'feature': row['feature'],
                    'importance': row['importance'],
                    'model': model_name
                })

        combined_df = pd.DataFrame(all_importance)

        # Aggregate by feature
        if method == 'mean':
            agg_func = 'mean'
        elif method == 'median':
            agg_func = 'median'
        elif method == 'max':
            agg_func = 'max'
        elif method == 'min':
            agg_func = 'min'
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # Calculate aggregated importance
        agg_importance = combined_df.groupby('feature')['importance'].agg([
            ('importance_' + method, agg_func),
            ('importance_std', 'std'),
            ('importance_min', 'min'),
            ('importance_max', 'max'),
            ('num_models', 'count')
        ]).reset_index()

        # Calculate consistency score (how consistently important across models)
        agg_importance['consistency'] = (
            agg_importance['importance_' + method] /
            (agg_importance['importance_std'] + 1e-6)
        )

        # Sort by importance
        agg_importance = agg_importance.sort_values(
            'importance_' + method, ascending=False
        ).reset_index(drop=True)

        # Add rank
        agg_importance['rank'] = range(1, len(agg_importance) + 1)

        # Store
        self.feature_importance_agg = agg_importance

        # Print summary
        print(f"Total unique features: {len(agg_importance)}")
        print(f"\nTop 10 Features by {method} importance:")
        print("-" * 80)
        for _, row in agg_importance.head(10).iterrows():
            print(f"  {row['rank']:3d}. {row['feature']:40s} "
                  f"Imp: {row['importance_' + method]:.4f} "
                  f"(±{row['importance_std']:.4f})")

        print(f"\n{'='*80}\n")

        if top_n:
            return agg_importance.head(top_n)

        return agg_importance

    def categorize_features(self) -> Dict[str, List[str]]:
        """
        Categorize features by type.

        Returns:
            Dictionary mapping category names to feature lists
        """
        if self.feature_importance_agg is None:
            print("No aggregated importance data. Call aggregate_importance() first.")
            return {}

        features = self.feature_importance_agg['feature'].tolist()

        categories = {
            'interaction': [],
            'ratio': [],
            'aggregation': [],
            'time_based': [],
            'rfm': [],
            'behavioral': [],
            'lifecycle': [],
            'advanced': [],
            'basic': []
        }

        # Categorization patterns
        for feature in features:
            feature_lower = feature.lower()

            # Interaction features (multiplication)
            if '_x_' in feature_lower or '*' in feature:
                categories['interaction'].append(feature)

            # Ratio features (division)
            elif '_div_' in feature_lower or '/' in feature or '_per_' in feature_lower:
                categories['ratio'].append(feature)

            # RFM features
            elif any(x in feature_lower for x in ['recency', 'frequency', 'monetary', 'rfm_']):
                categories['rfm'].append(feature)

            # Behavioral features
            elif any(x in feature_lower for x in ['change_', 'delta_', 'trend_', 'volatility']):
                categories['behavioral'].append(feature)

            # Lifecycle features
            elif any(x in feature_lower for x in ['lifecycle', 'tenure', 'stage', 'adoption']):
                categories['lifecycle'].append(feature)

            # Time-based features
            elif any(x in feature_lower for x in ['days_', 'months_', 'years_', 'date', 'time', 'last_', 'first_']):
                categories['time_based'].append(feature)

            # Aggregation features (mean, sum, max, min, std, etc.)
            elif any(x in feature_lower for x in ['mean', 'sum', 'max', 'min', 'std', 'count', 'total', 'avg']):
                categories['aggregation'].append(feature)

            # Advanced features (from advanced_features.py)
            elif any(x in feature_lower for x in ['intensity', 'consistency', 'decay', 'gap', 'peak']):
                categories['advanced'].append(feature)

            # Basic features
            else:
                categories['basic'].append(feature)

        # Store
        self.feature_categories = categories

        # Print summary
        print(f"\n{'='*80}")
        print(f"FEATURE CATEGORIZATION")
        print(f"{'='*80}\n")

        for category, feature_list in categories.items():
            if feature_list:
                print(f"{category.upper():20s}: {len(feature_list):4d} features")

        print(f"\n{'='*80}\n")

        return categories

    def analyze_consistency(self, top_n: int = 50) -> pd.DataFrame:
        """
        Analyze feature importance consistency across models.

        Args:
            top_n: Number of top features to analyze

        Returns:
            DataFrame with consistency analysis
        """
        if self.feature_importance_agg is None:
            print("No aggregated importance data. Call aggregate_importance() first.")
            return pd.DataFrame()

        print(f"\n{'='*80}")
        print(f"FEATURE CONSISTENCY ANALYSIS")
        print(f"{'='*80}\n")

        top_features = self.feature_importance_agg.head(top_n)

        # Classify consistency
        def classify_consistency(row):
            cv = row['importance_std'] / (row['importance_mean'] + 1e-6)
            if cv < 0.2:
                return 'Very Consistent'
            elif cv < 0.5:
                return 'Consistent'
            elif cv < 1.0:
                return 'Moderate'
            else:
                return 'Inconsistent'

        top_features['consistency_level'] = top_features.apply(classify_consistency, axis=1)

        # Print summary
        print(f"Consistency Analysis (Top {top_n} features):\n")

        consistency_counts = top_features['consistency_level'].value_counts()
        for level in ['Very Consistent', 'Consistent', 'Moderate', 'Inconsistent']:
            count = consistency_counts.get(level, 0)
            print(f"  {level:20s}: {count:3d} features")

        print(f"\nMost Consistent Features (Top 10):")
        print("-" * 80)

        consistent_features = top_features.nlargest(10, 'consistency')
        for _, row in consistent_features.iterrows():
            print(f"  {row['feature']:40s} "
                  f"Consistency: {row['consistency']:.2f} "
                  f"(Imp: {row['importance_mean']:.4f} ±{row['importance_std']:.4f})")

        print(f"\n{'='*80}\n")

        return top_features

    def load_training_data(self, X_train: pd.DataFrame) -> None:
        """
        Load training data for correlation analysis.

        Args:
            X_train: Training feature DataFrame
        """
        self.X_train = X_train
        print(f"[OK] Training data loaded: {X_train.shape}")

    def calculate_correlation_matrix(self,
                                    top_n: int = 50,
                                    method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix for top features.

        Args:
            top_n: Number of top features to include
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Correlation matrix DataFrame
        """
        if self.X_train is None:
            print("[WARN] Training data not loaded. Call load_training_data() first.")
            return pd.DataFrame()

        if self.feature_importance_agg is None:
            print("[WARN] No aggregated importance. Call aggregate_importance() first.")
            return pd.DataFrame()

        print(f"\n{'='*80}")
        print(f"CALCULATING CORRELATION MATRIX")
        print(f"{'='*80}")
        print(f"Method: {method}")
        print(f"Top N features: {top_n}")
        print(f"{'='*80}\n")

        # Get top N features
        top_features = self.feature_importance_agg.head(top_n)['feature'].tolist()

        # Filter to features that exist in X_train
        available_features = [f for f in top_features if f in self.X_train.columns]

        print(f"Features in correlation matrix: {len(available_features)}")

        # Calculate correlation
        X_subset = self.X_train[available_features]
        self.correlation_matrix = X_subset.corr(method=method)

        print(f"[OK] Correlation matrix calculated: {self.correlation_matrix.shape}")
        print(f"\n{'='*80}\n")

        return self.correlation_matrix

    def find_redundant_features(self, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """
        Find highly correlated (potentially redundant) feature pairs.

        Args:
            threshold: Correlation threshold for redundancy (default: 0.95)

        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        if self.correlation_matrix is None:
            print("[WARN] Correlation matrix not calculated. Call calculate_correlation_matrix() first.")
            return []

        print(f"\n{'='*80}")
        print(f"FINDING REDUNDANT FEATURES")
        print(f"{'='*80}")
        print(f"Correlation threshold: {threshold}")
        print(f"{'='*80}\n")

        redundant_pairs = []

        # Get upper triangle of correlation matrix
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                feat1 = self.correlation_matrix.columns[i]
                feat2 = self.correlation_matrix.columns[j]
                corr = abs(self.correlation_matrix.iloc[i, j])

                if corr >= threshold:
                    redundant_pairs.append((feat1, feat2, corr))

        # Sort by correlation (descending)
        redundant_pairs.sort(key=lambda x: x[2], reverse=True)

        if redundant_pairs:
            print(f"Found {len(redundant_pairs)} highly correlated pairs:\n")

            for feat1, feat2, corr in redundant_pairs[:20]:  # Show top 20
                print(f"  {feat1:40s} <-> {feat2:40s} (r={corr:.4f})")

            if len(redundant_pairs) > 20:
                print(f"\n  ... and {len(redundant_pairs) - 20} more pairs")
        else:
            print("No highly correlated feature pairs found.")

        print(f"\n{'='*80}\n")

        return redundant_pairs

    def recommend_features_to_remove(self,
                                    correlation_threshold: float = 0.95,
                                    importance_threshold: float = 0.001) -> List[str]:
        """
        Recommend features to remove based on correlation and importance.

        Strategy:
        1. For highly correlated pairs, remove the less important one
        2. Remove features below importance threshold

        Args:
            correlation_threshold: Threshold for considering features redundant
            importance_threshold: Minimum importance to keep feature

        Returns:
            List of features recommended for removal
        """
        if self.feature_importance_agg is None:
            print("[WARN] No aggregated importance. Call aggregate_importance() first.")
            return []

        print(f"\n{'='*80}")
        print(f"RECOMMENDING FEATURES TO REMOVE")
        print(f"{'='*80}\n")

        features_to_remove = set()

        # Strategy 1: Remove less important feature from correlated pairs
        if self.correlation_matrix is not None:
            redundant_pairs = self.find_redundant_features(correlation_threshold)

            importance_dict = dict(zip(
                self.feature_importance_agg['feature'],
                self.feature_importance_agg['importance_mean']
            ))

            print(f"Analyzing {len(redundant_pairs)} correlated pairs...\n")

            for feat1, feat2, corr in redundant_pairs:
                imp1 = importance_dict.get(feat1, 0)
                imp2 = importance_dict.get(feat2, 0)

                # Remove less important feature
                if imp1 < imp2:
                    features_to_remove.add(feat1)
                    print(f"  Remove {feat1:40s} (keep {feat2}, r={corr:.3f})")
                else:
                    features_to_remove.add(feat2)
                    print(f"  Remove {feat2:40s} (keep {feat1}, r={corr:.3f})")

        # Strategy 2: Remove low importance features
        print(f"\nRemoving features with importance < {importance_threshold}:\n")

        low_importance = self.feature_importance_agg[
            self.feature_importance_agg['importance_mean'] < importance_threshold
        ]

        for _, row in low_importance.iterrows():
            features_to_remove.add(row['feature'])
            print(f"  Remove {row['feature']:40s} (importance: {row['importance_mean']:.6f})")

        print(f"\n{'='*80}")
        print(f"REMOVAL SUMMARY")
        print(f"{'='*80}")
        print(f"Total features recommended for removal: {len(features_to_remove)}")
        print(f"  - Due to redundancy: {len([f for f in features_to_remove if f in [p[0] for p in redundant_pairs] or f in [p[1] for p in redundant_pairs]])}")
        print(f"  - Due to low importance: {len(low_importance)}")
        print(f"{'='*80}\n")

        return list(features_to_remove)

    def suggest_new_interactions(self, top_n: int = 10) -> List[Tuple[str, str]]:
        """
        Suggest new feature interactions to try.

        Strategy:
        - Combine top important features that aren't already interacted
        - Avoid redundant interactions

        Args:
            top_n: Number of top features to consider for interactions

        Returns:
            List of (feature1, feature2) tuples for new interactions
        """
        if self.feature_importance_agg is None:
            print("[WARN] No aggregated importance. Call aggregate_importance() first.")
            return []

        print(f"\n{'='*80}")
        print(f"SUGGESTING NEW FEATURE INTERACTIONS")
        print(f"{'='*80}\n")

        # Get top non-interaction features
        top_features = self.feature_importance_agg.head(top_n * 2)
        base_features = [
            f for f in top_features['feature']
            if '_x_' not in f.lower() and '*' not in f and '_div_' not in f.lower()
        ][:top_n]

        print(f"Top {len(base_features)} base features for interaction:\n")
        for i, feat in enumerate(base_features, 1):
            print(f"  {i:2d}. {feat}")

        # Get existing interactions
        existing_interactions = set()
        for feat in self.feature_importance_agg['feature']:
            if '_x_' in feat.lower() or '*' in feat:
                parts = feat.lower().replace('_x_', '*').split('*')
                if len(parts) == 2:
                    existing_interactions.add(tuple(sorted(parts)))

        # Generate new interaction suggestions
        suggestions = []

        for i in range(len(base_features)):
            for j in range(i + 1, len(base_features)):
                feat1 = base_features[i]
                feat2 = base_features[j]

                # Check if interaction already exists
                pair = tuple(sorted([feat1.lower(), feat2.lower()]))
                if pair not in existing_interactions:
                    suggestions.append((feat1, feat2))

        print(f"\n{'='*80}")
        print(f"NEW INTERACTION SUGGESTIONS")
        print(f"{'='*80}\n")

        print(f"Suggested {len(suggestions[:20])} new interactions:\n")

        for i, (feat1, feat2) in enumerate(suggestions[:20], 1):
            print(f"  {i:2d}. {feat1:35s} × {feat2}")

        if len(suggestions) > 20:
            print(f"\n  ... and {len(suggestions) - 20} more suggestions")

        print(f"\n{'='*80}\n")

        return suggestions

    def generate_recommendations(self) -> Dict[str, any]:
        """
        Generate comprehensive feature engineering recommendations.

        Returns:
            Dictionary with all recommendations
        """
        print(f"\n{'#'*80}")
        print(f"#{'FEATURE ENGINEERING RECOMMENDATIONS'.center(78)}#")
        print(f"{'#'*80}\n")

        recommendations = {}

        # 1. Top features to keep
        if self.feature_importance_agg is not None:
            top_20 = self.feature_importance_agg.head(20)['feature'].tolist()
            recommendations['keep_features'] = top_20

            print(f"{'='*80}")
            print(f"1. TOP 20 FEATURES TO KEEP")
            print(f"{'='*80}\n")
            for i, feat in enumerate(top_20, 1):
                imp = self.feature_importance_agg[
                    self.feature_importance_agg['feature'] == feat
                ]['importance_mean'].values[0]
                print(f"  {i:2d}. {feat:45s} (Importance: {imp:.4f})")
            print()

        # 2. Features to remove
        features_to_remove = self.recommend_features_to_remove()
        recommendations['remove_features'] = features_to_remove

        # 3. New interactions to try
        new_interactions = self.suggest_new_interactions(top_n=10)
        recommendations['new_interactions'] = new_interactions

        # 4. Category analysis
        if self.feature_categories:
            print(f"{'='*80}")
            print(f"2. FEATURE CATEGORY BREAKDOWN")
            print(f"{'='*80}\n")

            # Count top 50 features by category
            if self.feature_importance_agg is not None:
                top_50_features = set(self.feature_importance_agg.head(50)['feature'])

                category_in_top50 = {}
                for category, features in self.feature_categories.items():
                    count = len([f for f in features if f in top_50_features])
                    category_in_top50[category] = count

                # Sort by count
                sorted_categories = sorted(
                    category_in_top50.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                for category, count in sorted_categories:
                    if count > 0:
                        pct = (count / 50) * 100
                        print(f"  {category.upper():20s}: {count:2d} features ({pct:5.1f}%)")

                recommendations['category_breakdown'] = dict(sorted_categories)
                print()

        # 5. Action items
        print(f"{'='*80}")
        print(f"3. ACTION ITEMS")
        print(f"{'='*80}\n")

        print(f"[OK] Keep: {len(recommendations.get('keep_features', []))} most important features")
        print(f"[FAIL] Remove: {len(recommendations.get('remove_features', []))} redundant/low-importance features")
        print(f"+ Try: {len(recommendations.get('new_interactions', []))} new feature interactions")

        print(f"\n{'#'*80}")
        print(f"#{'END OF RECOMMENDATIONS'.center(78)}#")
        print(f"{'#'*80}\n")

        return recommendations

    def visualize_importance(self, top_n: int = 30, save_path: Optional[str] = None) -> None:
        """
        Create feature importance visualization.

        Args:
            top_n: Number of top features to visualize
            save_path: Path to save figure (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("❌ matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")
            return

        if self.feature_importance_agg is None:
            print("[WARN] No aggregated importance. Call aggregate_importance() first.")
            return

        print(f"\nGenerating feature importance visualization...")

        # Get top features
        top_features = self.feature_importance_agg.head(top_n)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.3)))

        # Create horizontal bar chart
        y_pos = np.arange(len(top_features))

        bars = ax.barh(y_pos, top_features['importance_mean'],
                      xerr=top_features['importance_std'],
                      color='#2E86AB', alpha=0.8, error_kw={'linewidth': 1})

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Importance (Mean ± Std)', fontsize=12)
        ax.set_title(f'Top {top_n} Features by Importance', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Feature importance plot saved to: {save_path}")

        plt.show()

    def visualize_correlation_heatmap(self,
                                     top_n: int = 30,
                                     save_path: Optional[str] = None) -> None:
        """
        Create correlation heatmap for top features.

        Args:
            top_n: Number of top features to include
            save_path: Path to save figure (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("❌ matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")
            return

        if self.correlation_matrix is None:
            print("[WARN] Correlation matrix not calculated. Call calculate_correlation_matrix() first.")
            return

        print(f"\nGenerating correlation heatmap...")

        # Get top N features
        corr_subset = self.correlation_matrix.iloc[:top_n, :top_n]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))

        # Create heatmap
        sns.heatmap(corr_subset,
                   cmap='RdBu_r',
                   center=0,
                   vmin=-1, vmax=1,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={'label': 'Correlation'},
                   ax=ax,
                   annot=False)

        ax.set_title(f'Feature Correlation Heatmap (Top {top_n})',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Correlation heatmap saved to: {save_path}")

        plt.show()

    def visualize_category_importance(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization of feature importance by category.

        Args:
            save_path: Path to save figure (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("❌ matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")
            return

        if self.feature_importance_agg is None or not self.feature_categories:
            print("[WARN] Need aggregated importance and categories. Call aggregate_importance() and categorize_features().")
            return

        print(f"\nGenerating category importance visualization...")

        # Calculate average importance by category
        category_importance = {}

        for category, features in self.feature_categories.items():
            if not features:
                continue

            # Get importance for features in this category
            cat_features = self.feature_importance_agg[
                self.feature_importance_agg['feature'].isin(features)
            ]

            if not cat_features.empty:
                category_importance[category] = {
                    'mean_importance': cat_features['importance_mean'].mean(),
                    'total_importance': cat_features['importance_mean'].sum(),
                    'count': len(cat_features)
                }

        # Create DataFrame
        cat_df = pd.DataFrame(category_importance).T.reset_index()
        cat_df.columns = ['category', 'mean_importance', 'total_importance', 'count']
        cat_df = cat_df.sort_values('total_importance', ascending=False)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Total importance by category
        ax1.barh(cat_df['category'], cat_df['total_importance'],
                color='#A23B72', alpha=0.8)
        ax1.set_xlabel('Total Importance', fontsize=12)
        ax1.set_title('Total Feature Importance by Category', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Plot 2: Feature count by category
        ax2.barh(cat_df['category'], cat_df['count'],
                color='#F18F01', alpha=0.8)
        ax2.set_xlabel('Number of Features', fontsize=12)
        ax2.set_title('Feature Count by Category', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Category importance plot saved to: {save_path}")

        plt.show()

    def export_recommendations(self, output_file: str = 'feature_recommendations.json') -> None:
        """
        Export recommendations to JSON file.

        Args:
            output_file: Path to output JSON file
        """
        recommendations = self.generate_recommendations()

        # Convert to JSON-serializable format
        json_recommendations = {
            'keep_features': recommendations.get('keep_features', []),
            'remove_features': recommendations.get('remove_features', []),
            'new_interactions': [
                {'feature1': f1, 'feature2': f2}
                for f1, f2 in recommendations.get('new_interactions', [])
            ],
            'category_breakdown': recommendations.get('category_breakdown', {})
        }

        with open(output_file, 'w') as f:
            json.dump(json_recommendations, f, indent=2)

        print(f"[OK] Recommendations exported to: {output_file}")


def main():
    """
    Example usage of FeatureAnalyzer.
    """
    print("""
    ================================================================================
    FEATURE ANALYZER - EXAMPLE USAGE
    ================================================================================

    This script analyzes feature importance across models and provides optimization
    recommendations.

    Basic Usage:
    -----------

    from analyze_features import FeatureAnalyzer

    # Initialize analyzer
    analyzer = FeatureAnalyzer()

    # Load feature importance files
    analyzer.load_importance_files(
        directory='.',
        pattern='feature_importance*.csv'
    )

    # Aggregate importance across models
    agg_importance = analyzer.aggregate_importance(method='mean')

    # Categorize features
    categories = analyzer.categorize_features()

    # Analyze consistency
    consistency = analyzer.analyze_consistency(top_n=50)

    # Load training data for correlation analysis
    analyzer.load_training_data(X_train)

    # Calculate correlations
    corr_matrix = analyzer.calculate_correlation_matrix(top_n=50)

    # Find redundant features
    redundant = analyzer.find_redundant_features(threshold=0.95)

    # Generate comprehensive recommendations
    recommendations = analyzer.generate_recommendations()

    # Visualizations
    analyzer.visualize_importance(top_n=30, save_path='feature_importance.png')
    analyzer.visualize_correlation_heatmap(top_n=30, save_path='correlation_heatmap.png')
    analyzer.visualize_category_importance(save_path='category_importance.png')

    # Export recommendations
    analyzer.export_recommendations('feature_recommendations.json')

    ================================================================================

    Integration with Training Pipeline:
    ----------------------------------

    # After training models and saving feature importance:

    from analyze_features import FeatureAnalyzer

    analyzer = FeatureAnalyzer()
    analyzer.load_importance_files(pattern='feature_importance*.csv')
    analyzer.aggregate_importance()
    analyzer.categorize_features()

    # Load training data
    analyzer.load_training_data(X_train)
    analyzer.calculate_correlation_matrix(top_n=100)

    # Get actionable recommendations
    recommendations = analyzer.generate_recommendations()

    # Remove low-value features
    features_to_remove = recommendations['remove_features']
    X_train_optimized = X_train.drop(columns=features_to_remove)

    ================================================================================
    """)


if __name__ == "__main__":
    main()
