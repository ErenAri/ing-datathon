"""
Quick Baseline Test - Verify main.py still produces 1.23

This runs your proven baseline to confirm nothing broke.
Expected OOF score: ~1.23 (same as rank 50 submission)

Usage:
    python test_baseline.py
"""

import subprocess
import sys
import re

print("\n" + "="*70)
print("BASELINE TEST - VERIFYING MAIN.PY STILL WORKS")
print("="*70)
print("\nRunning: python -m src.main --models lgb xgb cat two meta --cat-seeds 5 --with-stacker")
print("\nThis will take 10-15 minutes...")
print("="*70 + "\n")

# Run the baseline command
cmd = [
    sys.executable, "-m", "src.main",
    "--models", "lgb", "xgb", "cat", "two", "meta",
    "--cat-seeds", "5",
    "--with-stacker"
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Try to extract OOF score from output
    output = result.stdout + result.stderr

    # Look for "Blended Ensemble OOF Score: X.XXXXXX"
    match = re.search(r'Blended Ensemble OOF Score:\s+([\d.]+)', output)
    if match:
        oof_score = float(match.group(1))
        print("\n" + "="*70)
        print(f"✅ BASELINE TEST COMPLETE")
        print("="*70)
        print(f"OOF Score: {oof_score:.6f}")

        if oof_score >= 1.22:
            print(f"✅ PASSED - Score is {oof_score:.6f} (expected ~1.23)")
            print("\nYour baseline is working correctly!")
            print("\nNext step: Try Improvement 1 (equal weights)")
        elif oof_score >= 1.20:
            print(f"⚠️  WARNING - Score is {oof_score:.6f} (lower than expected 1.23)")
            print("\nBaseline still works but score dropped slightly.")
            print("This could be due to randomness or code changes.")
        else:
            print(f"❌ FAILED - Score is {oof_score:.6f} (much lower than expected 1.23)")
            print("\nSomething may have changed in the code.")
            print("Check if features or models are different.")
    else:
        print("\n" + "="*70)
        print("⚠️  Could not extract OOF score from output")
        print("="*70)
        print("Check the output above for 'Blended Ensemble OOF Score'")

    print("="*70)

except subprocess.TimeoutExpired:
    print("\n❌ TIMEOUT - Baseline test took longer than 30 minutes")
    print("This may indicate an issue with the code or environment")

except Exception as e:
    print(f"\n❌ ERROR running baseline test: {e}")
    print("\nTry running manually:")
    print("python -m src.main --models lgb xgb cat two meta --cat-seeds 5 --with-stacker")
