import os

FP_DIR = "evaluation/false_positives"

# Results from physics filter (hard-coded from run)
ACCEPTED = 24
REJECTED = 26
TOTAL_FP = ACCEPTED + REJECTED

print("False Positive Analysis Metrics")
print("--------------------------------")
print(f"Total CNN False Positives: {TOTAL_FP}")
print(f"Rejected by Physics Filters: {REJECTED}")
print(f"Remaining False Positives: {ACCEPTED}")

reduction = (REJECTED / TOTAL_FP) * 100
print(f"False Positive Reduction: {reduction:.2f}%")
