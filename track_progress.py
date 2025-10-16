"""
Progress Tracker for ING Datathon
==================================

This module provides comprehensive tracking and visualization of competition progress.
It maintains a history of all submissions, tracks CV and LB scores, generates progress
graphs, and provides insights into your journey toward first place.

Features:
- JSON-based submission history
- Automatic detection of new submissions
- CV and LB score tracking
- Progress visualization with matplotlib
- Leaderboard comparison
- Improvement rate analysis
- Terminal-based progress reports

Author: ING Datathon Team
Date: 2025-10-12
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ProgressTracker:
    """
    Competition Progress Tracker

    This class maintains a comprehensive history of all submissions, including
    CV scores, LB scores, ranks, and descriptions. It provides visualization
    and analysis tools to track progress toward winning the competition.

    Attributes:
        history_file (Path): Path to JSON file storing submission history
        submissions (List[Dict]): List of all submissions
        watch_file (Path): File to watch for auto-detection of new submissions
    """

    def __init__(self, history_file: str = 'submission_history.json'):
        """
        Initialize ProgressTracker.

        Args:
            history_file: Path to JSON file for storing submission history
        """
        self.history_file = Path(history_file)
        self.submissions = []
        self.watch_file = None

        # Load existing history
        self._load_history()

        print(f"\n{'='*80}")
        print(f"PROGRESS TRACKER INITIALIZED")
        print(f"{'='*80}")
        print(f"History file: {self.history_file}")
        print(f"Total submissions tracked: {len(self.submissions)}")
        print(f"{'='*80}\n")

    def _load_history(self) -> None:
        """Load submission history from JSON file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.submissions = json.load(f)
            print(f"Loaded {len(self.submissions)} submissions from history.")
        else:
            self.submissions = []
            print("No existing history found. Starting fresh.")

    def _save_history(self) -> None:
        """Save submission history to JSON file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.submissions, f, indent=2)

    def add_submission(self,
                      version_name: str,
                      cv_score: Optional[float] = None,
                      lb_score: Optional[float] = None,
                      rank: Optional[int] = None,
                      description: str = "",
                      filename: Optional[str] = None) -> None:
        """
        Add a new submission to the history.

        Args:
            version_name: Name/identifier for this submission
            cv_score: Cross-validation score (optional)
            lb_score: Leaderboard score (optional)
            rank: Leaderboard rank (optional)
            description: Description of key changes/features
            filename: Path to submission CSV file (optional)
        """
        # Create submission record
        submission = {
            'version_name': version_name,
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cv_score': cv_score,
            'lb_score': lb_score,
            'rank': rank,
            'description': description,
            'filename': filename,
            'id': len(self.submissions) + 1
        }

        # Add to history
        self.submissions.append(submission)

        # Save to file
        self._save_history()

        print(f"\n{'='*80}")
        print(f"NEW SUBMISSION ADDED")
        print(f"{'='*80}")
        print(f"Version: {version_name}")
        print(f"ID: {submission['id']}")
        print(f"Date/Time: {submission['datetime']}")
        if cv_score is not None:
            print(f"CV Score: {cv_score:.6f}")
        if lb_score is not None:
            print(f"LB Score: {lb_score:.6f}")
        if rank is not None:
            print(f"Rank: {rank}")
        if description:
            print(f"Description: {description}")
        print(f"{'='*80}\n")

    def update_lb_score(self,
                       submission_id: Optional[int] = None,
                       version_name: Optional[str] = None,
                       lb_score: float = None,
                       rank: Optional[int] = None) -> None:
        """
        Update the leaderboard score for an existing submission.

        Args:
            submission_id: ID of submission to update (optional)
            version_name: Name of submission to update (optional)
            lb_score: New leaderboard score
            rank: New leaderboard rank (optional)
        """
        # Find submission
        submission = None

        if submission_id is not None:
            for sub in self.submissions:
                if sub['id'] == submission_id:
                    submission = sub
                    break
        elif version_name is not None:
            for sub in self.submissions:
                if sub['version_name'] == version_name:
                    submission = sub
                    break
        else:
            # Update most recent
            submission = self.submissions[-1] if self.submissions else None

        if submission is None:
            print("❌ Submission not found!")
            return

        # Update scores
        old_lb_score = submission.get('lb_score')
        old_rank = submission.get('rank')

        submission['lb_score'] = lb_score
        if rank is not None:
            submission['rank'] = rank

        # Save
        self._save_history()

        # Show update
        print(f"\n{'='*80}")
        print(f"SUBMISSION UPDATED")
        print(f"{'='*80}")
        print(f"Version: {submission['version_name']}")
        print(f"ID: {submission['id']}")

        if old_lb_score is not None:
            diff = lb_score - old_lb_score
            print(f"LB Score: {old_lb_score:.6f} → {lb_score:.6f} ({diff:+.6f})")
        else:
            print(f"LB Score: {lb_score:.6f} (new)")

        if rank is not None:
            if old_rank is not None:
                rank_change = old_rank - rank  # Positive = improved
                print(f"Rank: {old_rank} → {rank} ({rank_change:+d})")
            else:
                print(f"Rank: {rank} (new)")

        print(f"{'='*80}\n")

    def get_submission_by_id(self, submission_id: int) -> Optional[Dict]:
        """Get submission by ID."""
        for sub in self.submissions:
            if sub['id'] == submission_id:
                return sub
        return None

    def get_submission_by_name(self, version_name: str) -> Optional[Dict]:
        """Get submission by version name."""
        for sub in self.submissions:
            if sub['version_name'] == version_name:
                return sub
        return None

    def get_latest_submission(self) -> Optional[Dict]:
        """Get the most recent submission."""
        return self.submissions[-1] if self.submissions else None

    def print_history(self, n_recent: Optional[int] = None) -> None:
        """
        Print submission history in a nice table format.

        Args:
            n_recent: Number of recent submissions to show (None = all)
        """
        if not self.submissions:
            print("No submissions in history.")
            return

        print(f"\n{'='*80}")
        print(f"SUBMISSION HISTORY")
        print(f"{'='*80}\n")

        # Prepare data
        submissions_to_show = self.submissions[-n_recent:] if n_recent else self.submissions

        # Create DataFrame
        df = pd.DataFrame(submissions_to_show)

        # Format columns
        display_cols = ['id', 'version_name', 'datetime', 'cv_score', 'lb_score', 'rank']
        df_display = df[display_cols].copy()

        # Format scores
        for col in ['cv_score', 'lb_score']:
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"
            )

        # Format rank
        df_display['rank'] = df_display['rank'].apply(
            lambda x: str(int(x)) if pd.notna(x) else "N/A"
        )

        print(df_display.to_string(index=False))
        print(f"\n{'='*80}\n")

    def print_leaderboard_comparison(self,
                                     target_rank: int = 1,
                                     target_score: Optional[float] = None) -> None:
        """
        Print comparison with leaderboard target.

        Args:
            target_rank: Target rank to reach (default: 1 for first place)
            target_score: Target score to beat (optional)
        """
        latest = self.get_latest_submission()

        if not latest:
            print("No submissions to compare.")
            return

        print(f"\n{'='*80}")
        print(f"LEADERBOARD COMPARISON")
        print(f"{'='*80}\n")

        print(f"Current Submission: {latest['version_name']}")
        print(f"Date: {latest['datetime']}")
        print(f"-" * 80)

        # CV Score
        if latest['cv_score'] is not None:
            print(f"\nCV Score: {latest['cv_score']:.6f}")

        # LB Score and Rank
        if latest['lb_score'] is not None:
            print(f"LB Score: {latest['lb_score']:.6f}")

            if target_score is not None:
                gap = target_score - latest['lb_score']
                pct_gap = (gap / target_score) * 100
                print(f"Target Score: {target_score:.6f}")
                print(f"Gap to Target: {gap:.6f} ({pct_gap:.2f}%)")
        else:
            print(f"\nLB Score: Not yet submitted")

        if latest['rank'] is not None:
            print(f"\nCurrent Rank: {latest['rank']}")
            print(f"Target Rank: {target_rank}")
            positions_to_climb = latest['rank'] - target_rank
            print(f"Positions to Climb: {positions_to_climb}")
        else:
            print(f"\nRank: Not yet available")

        print(f"\n{'='*80}\n")

    def calculate_improvement_rate(self) -> Dict[str, float]:
        """
        Calculate improvement rate over time.

        Returns:
            Dictionary with improvement statistics
        """
        if len(self.submissions) < 2:
            print("Need at least 2 submissions to calculate improvement rate.")
            return {}

        # Filter submissions with scores
        cv_submissions = [s for s in self.submissions if s['cv_score'] is not None]
        lb_submissions = [s for s in self.submissions if s['lb_score'] is not None]

        stats = {}

        # CV improvement
        if len(cv_submissions) >= 2:
            cv_scores = [s['cv_score'] for s in cv_submissions]
            cv_improvement = cv_scores[-1] - cv_scores[0]
            cv_improvement_pct = (cv_improvement / cv_scores[0]) * 100
            cv_improvement_per_sub = cv_improvement / (len(cv_scores) - 1)

            stats['cv_improvement_total'] = cv_improvement
            stats['cv_improvement_pct'] = cv_improvement_pct
            stats['cv_improvement_per_submission'] = cv_improvement_per_sub
            stats['cv_best'] = max(cv_scores)
            stats['cv_current'] = cv_scores[-1]

        # LB improvement
        if len(lb_submissions) >= 2:
            lb_scores = [s['lb_score'] for s in lb_submissions]
            lb_improvement = lb_scores[-1] - lb_scores[0]
            lb_improvement_pct = (lb_improvement / lb_scores[0]) * 100
            lb_improvement_per_sub = lb_improvement / (len(lb_scores) - 1)

            stats['lb_improvement_total'] = lb_improvement
            stats['lb_improvement_pct'] = lb_improvement_pct
            stats['lb_improvement_per_submission'] = lb_improvement_per_sub
            stats['lb_best'] = max(lb_scores)
            stats['lb_current'] = lb_scores[-1]

        # Rank improvement
        if len(lb_submissions) >= 2:
            ranks = [s['rank'] for s in lb_submissions if s['rank'] is not None]
            if len(ranks) >= 2:
                rank_improvement = ranks[0] - ranks[-1]  # Positive = better
                stats['rank_improvement_total'] = rank_improvement
                stats['rank_best'] = min(ranks)
                stats['rank_current'] = ranks[-1]

        return stats

    def print_improvement_analysis(self) -> None:
        """Print detailed improvement analysis."""
        stats = self.calculate_improvement_rate()

        if not stats:
            print("Insufficient data for improvement analysis.")
            return

        print(f"\n{'='*80}")
        print(f"IMPROVEMENT ANALYSIS")
        print(f"{'='*80}\n")

        # CV Stats
        if 'cv_improvement_total' in stats:
            print(f"Cross-Validation:")
            print(f"  Current Score: {stats['cv_current']:.6f}")
            print(f"  Best Score: {stats['cv_best']:.6f}")
            print(f"  Total Improvement: {stats['cv_improvement_total']:+.6f} ({stats['cv_improvement_pct']:+.2f}%)")
            print(f"  Avg Improvement per Submission: {stats['cv_improvement_per_submission']:+.6f}")
            print()

        # LB Stats
        if 'lb_improvement_total' in stats:
            print(f"Leaderboard:")
            print(f"  Current Score: {stats['lb_current']:.6f}")
            print(f"  Best Score: {stats['lb_best']:.6f}")
            print(f"  Total Improvement: {stats['lb_improvement_total']:+.6f} ({stats['lb_improvement_pct']:+.2f}%)")
            print(f"  Avg Improvement per Submission: {stats['lb_improvement_per_submission']:+.6f}")
            print()

        # Rank Stats
        if 'rank_improvement_total' in stats:
            print(f"Rank:")
            print(f"  Current Rank: {stats['rank_current']}")
            print(f"  Best Rank: {stats['rank_best']}")
            print(f"  Total Positions Climbed: {stats['rank_improvement_total']:+d}")
            print()

        print(f"{'='*80}\n")

    def show_progress_graph(self,
                           save_path: Optional[str] = None,
                           show: bool = True) -> None:
        """
        Generate progress visualization with matplotlib.

        Args:
            save_path: Path to save the figure (optional)
            show: Whether to display the figure (default: True)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            print("❌ matplotlib not installed. Install with: pip install matplotlib")
            return

        if not self.submissions:
            print("No submissions to visualize.")
            return

        # Prepare data
        df = pd.DataFrame(self.submissions)
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('ING Datathon - Progress Tracker', fontsize=16, fontweight='bold')

        # Plot 1: CV and LB Scores over time
        ax1 = axes[0]

        # CV scores
        cv_data = df[df['cv_score'].notna()]
        if not cv_data.empty:
            ax1.plot(cv_data['datetime'], cv_data['cv_score'],
                    marker='o', linestyle='-', linewidth=2, markersize=8,
                    label='CV Score', color='#2E86AB', alpha=0.8)

        # LB scores
        lb_data = df[df['lb_score'].notna()]
        if not lb_data.empty:
            ax1.plot(lb_data['datetime'], lb_data['lb_score'],
                    marker='s', linestyle='-', linewidth=2, markersize=8,
                    label='LB Score', color='#A23B72', alpha=0.8)

        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('CV and LB Scores Over Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Leaderboard Rank over time
        ax2 = axes[1]

        rank_data = df[df['rank'].notna()]
        if not rank_data.empty:
            ax2.plot(rank_data['datetime'], rank_data['rank'],
                    marker='o', linestyle='-', linewidth=2, markersize=8,
                    color='#F18F01', alpha=0.8)

            # Invert y-axis (lower rank = better)
            ax2.invert_yaxis()

            # Add horizontal line at rank 1
            ax2.axhline(y=1, color='green', linestyle='--', linewidth=2,
                       alpha=0.5, label='Target: 1st Place')

            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Rank (Lower is Better)', fontsize=12)
            ax2.set_title('Leaderboard Rank Over Time', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10, loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax2.text(0.5, 0.5, 'No Rank Data Available',
                    ha='center', va='center', fontsize=14, transform=ax2.transAxes)

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Progress graph saved to: {save_path}")

        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()

    def print_progress_report(self,
                             target_rank: int = 1,
                             target_score: Optional[float] = None) -> None:
        """
        Generate comprehensive progress report in terminal.

        Args:
            target_rank: Target rank to reach
            target_score: Target LB score to beat (optional)
        """
        print(f"\n{'#'*80}")
        print(f"#{'ING DATATHON - PROGRESS REPORT'.center(78)}#")
        print(f"{'#'*80}\n")

        # Section 1: Overall Statistics
        print(f"{'='*80}")
        print(f"OVERALL STATISTICS")
        print(f"{'='*80}\n")

        print(f"Total Submissions: {len(self.submissions)}")

        if self.submissions:
            first_date = self.submissions[0]['datetime']
            last_date = self.submissions[-1]['datetime']
            print(f"First Submission: {first_date}")
            print(f"Latest Submission: {last_date}")

            # Calculate days
            first_dt = datetime.strptime(first_date, '%Y-%m-%d %H:%M:%S')
            last_dt = datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S')
            days = (last_dt - first_dt).days
            if days > 0:
                submissions_per_day = len(self.submissions) / days
                print(f"Time Span: {days} days")
                print(f"Submissions per Day: {submissions_per_day:.2f}")

        print()

        # Section 2: Current Status
        latest = self.get_latest_submission()
        if latest:
            print(f"{'='*80}")
            print(f"CURRENT STATUS")
            print(f"{'='*80}\n")

            print(f"Latest Version: {latest['version_name']}")
            print(f"Date: {latest['datetime']}")

            if latest['cv_score'] is not None:
                print(f"\nCV Score: {latest['cv_score']:.6f}")

            if latest['lb_score'] is not None:
                print(f"LB Score: {latest['lb_score']:.6f}")

                # Progress bar for score
                if target_score:
                    progress = (latest['lb_score'] / target_score) * 100
                    self._print_progress_bar("Score Progress", progress, target_score)

            if latest['rank'] is not None:
                print(f"\nCurrent Rank: {latest['rank']}")

                # Progress bar for rank (inverted - closer to 1 is better)
                if latest['rank'] > target_rank:
                    positions_remaining = latest['rank'] - target_rank
                    # Assume starting from rank 100 as baseline
                    total_positions = 100 - target_rank
                    climbed = 100 - latest['rank']
                    progress = (climbed / total_positions) * 100
                    self._print_progress_bar("Rank Progress", progress, target_rank)

            if latest['description']:
                print(f"\nKey Changes: {latest['description']}")

            print()

        # Section 3: Improvement Analysis
        self.print_improvement_analysis()

        # Section 4: Leaderboard Comparison
        if target_rank or target_score:
            self.print_leaderboard_comparison(target_rank, target_score)

        # Section 5: Recent History
        print(f"{'='*80}")
        print(f"RECENT SUBMISSIONS (Last 5)")
        print(f"{'='*80}\n")
        self.print_history(n_recent=5)

        print(f"{'#'*80}")
        print(f"#{'END OF PROGRESS REPORT'.center(78)}#")
        print(f"{'#'*80}\n")

    def _print_progress_bar(self, label: str, percentage: float, target: float) -> None:
        """
        Print a terminal-based progress bar.

        Args:
            label: Label for the progress bar
            percentage: Percentage complete (0-100)
            target: Target value
        """
        bar_length = 40
        filled_length = int(bar_length * percentage / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        print(f"\n{label}:")
        print(f"[{bar}] {percentage:.1f}%")
        print(f"Target: {target}")

    def watch_for_new_submissions(self,
                                  watch_file: str = 'submission.csv',
                                  auto_add: bool = True) -> None:
        """
        Set up auto-detection of new submissions.

        Args:
            watch_file: File to watch for changes
            auto_add: Automatically add to history when detected
        """
        self.watch_file = Path(watch_file)
        print(f"✓ Watching for new submissions: {self.watch_file}")

        if auto_add:
            print(f"  Auto-add enabled: New submissions will be automatically tracked")

    def check_for_new_submission(self,
                                auto_add: bool = True,
                                version_prefix: str = "auto") -> bool:
        """
        Check if watch file has been updated and add if new.

        Args:
            auto_add: Automatically add the submission
            version_prefix: Prefix for auto-generated version names

        Returns:
            True if new submission detected and added, False otherwise
        """
        if self.watch_file is None or not self.watch_file.exists():
            return False

        # Get file modification time
        mod_time = datetime.fromtimestamp(self.watch_file.stat().st_mtime)

        # Check if this is newer than our latest submission
        if self.submissions:
            latest_time = datetime.strptime(self.submissions[-1]['datetime'],
                                          '%Y-%m-%d %H:%M:%S')
            if mod_time <= latest_time:
                return False

        # New submission detected
        if auto_add:
            version_name = f"{version_prefix}_{len(self.submissions) + 1}"
            self.add_submission(
                version_name=version_name,
                description="Auto-detected submission",
                filename=str(self.watch_file)
            )
            print(f"✓ New submission auto-added: {version_name}")
            return True

        return True

    def export_to_csv(self, output_file: str = 'submission_history.csv') -> None:
        """
        Export submission history to CSV.

        Args:
            output_file: Path to output CSV file
        """
        if not self.submissions:
            print("No submissions to export.")
            return

        df = pd.DataFrame(self.submissions)
        df.to_csv(output_file, index=False)
        print(f"✓ Submission history exported to: {output_file}")


def main():
    """
    Example usage of ProgressTracker.
    """
    print("""
    ================================================================================
    PROGRESS TRACKER - EXAMPLE USAGE
    ================================================================================

    This script tracks your competition progress toward first place!

    Basic Usage:
    -----------

    from track_progress import ProgressTracker

    # Initialize tracker
    tracker = ProgressTracker(history_file='submission_history.json')

    # Add a new submission
    tracker.add_submission(
        version_name='baseline_lgb',
        cv_score=0.856234,
        description='LightGBM with basic features'
    )

    # Update with leaderboard results
    tracker.update_lb_score(
        version_name='baseline_lgb',
        lb_score=0.852345,
        rank=45
    )

    # Add another submission
    tracker.add_submission(
        version_name='lgb_xgb_ensemble',
        cv_score=0.862145,
        lb_score=0.858932,
        rank=32,
        description='LightGBM + XGBoost weighted ensemble (0.6/0.4)'
    )

    # View progress
    tracker.print_progress_report(target_rank=1, target_score=0.900000)

    # Show graphs
    tracker.show_progress_graph(save_path='progress.png')

    # View recent submissions
    tracker.print_history(n_recent=5)

    # Improvement analysis
    tracker.print_improvement_analysis()

    # Export to CSV
    tracker.export_to_csv('my_submissions.csv')

    ================================================================================

    Auto-Detection:
    --------------

    # Watch for new submission files
    tracker.watch_for_new_submissions(watch_file='submission.csv', auto_add=True)

    # Check for updates (call this periodically)
    if tracker.check_for_new_submission():
        print("New submission detected!")

    ================================================================================

    Integration with Training Pipeline:
    ----------------------------------

    # At the end of main.py, after creating submission.csv:

    from track_progress import ProgressTracker

    tracker = ProgressTracker()
    tracker.add_submission(
        version_name='four_model_ensemble',
        cv_score=ensemble_score,
        description='LGB + XGB + CAT + Two-Stage (0.30/0.25/0.20/0.25)'
    )

    # Then manually update after submitting to leaderboard:
    # tracker.update_lb_score(lb_score=0.865432, rank=12)

    ================================================================================
    """)


if __name__ == "__main__":
    main()
