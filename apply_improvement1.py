"""
Apply Improvement 1: Equal-Weight Ensemble

This modifies main.py to use equal weights instead of per-month optimized weights.
This is the SAFEST improvement - equal weights cannot overfit by definition.

Expected:
- OOF may drop 0.01-0.02 (acceptable)
- Public-private gap should shrink by 50-60%
- Rank should improve from 86 to ~64-68

Usage:
    python apply_improvement1.py
"""

import os
import re

MAIN_PY = "src/main.py"
BACKUP_PY = "src/main.py.backup"

print("\n" + "="*70)
print("APPLYING IMPROVEMENT 1: EQUAL-WEIGHT ENSEMBLE")
print("="*70)

# Check if main.py exists
if not os.path.exists(MAIN_PY):
    print(f"❌ ERROR: {MAIN_PY} not found!")
    print("Make sure you're in the project root directory")
    exit(1)

# Create backup
print(f"\n1. Creating backup: {BACKUP_PY}")
with open(MAIN_PY, 'r', encoding='utf-8') as f:
    content = f.read()
with open(BACKUP_PY, 'w', encoding='utf-8') as f:
    f.write(content)
print("   ✅ Backup created")

# Find and replace the weight optimization section
print("\n2. Modifying ensemble weights...")

# Pattern to find: the baseline_weights dictionary and its usage
pattern = r'(if USE_FIXED_WEIGHTS:\s+print\([^)]+\)\s+baseline_weights = \{[^}]+\})\s+# Only use weights for models that exist\s+final_weights = \{n: baseline_weights\.get\(n, 0\.0\) for n in names\}'

replacement = r'''\1
    # IMPROVEMENT 1: Use equal weights (maximum robustness)
    print("  (Improvement 1: Using equal weights for maximum robustness)")
    final_weights = {n: 1.0/len(names) for n in names}
    # Skip the baseline_weights lookup - we want equal weights'''

# Try to apply the replacement
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

if new_content == content:
    print("   ⚠️  WARNING: Could not find exact pattern to replace")
    print("   Trying alternative approach...")

    # Alternative: just replace the final_weights assignment
    # Find: final_weights = {n: baseline_weights.get(n, 0.0) for n in names}
    # Replace with equal weights
    alt_pattern = r'final_weights = \{n: baseline_weights\.get\(n, 0\.0\) for n in names\}'
    alt_replacement = r'''final_weights = {n: 1.0/len(names) for n in names}  # IMPROVEMENT 1: Equal weights'''

    new_content = re.sub(alt_pattern, alt_replacement, content)

    if new_content == content:
        print("   ❌ ERROR: Could not apply modification")
        print("   Manual edit required:")
        print("\n   Edit src/main.py around line 1163:")
        print("   Change:")
        print("       final_weights = {n: baseline_weights.get(n, 0.0) for n in names}")
        print("   To:")
        print("       final_weights = {n: 1.0/len(names) for n in names}")
        exit(1)

# Write modified content
with open(MAIN_PY, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("   ✅ Modification applied")

print("\n" + "="*70)
print("✅ IMPROVEMENT 1 APPLIED SUCCESSFULLY")
print("="*70)

print("\n3. Next steps:")
print("   Run: python -m src.main --models lgb xgb cat two meta --cat-seeds 5 --with-stacker")
print("\n4. Expected results:")
print("   - OOF score: 1.22-1.23 (slight drop is OK)")
print("   - More robust (less overfitting)")
print("   - Smaller public-private gap")

print("\n5. To revert if needed:")
print(f"   Copy {BACKUP_PY} back to {MAIN_PY}")

print("="*70)
