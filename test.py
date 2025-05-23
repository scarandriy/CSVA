import json
import re
from collections import Counter

# Initialize statistics
total_files = 0
legit_grade_counts = Counter()
scam_grade_counts = Counter()
failed_parses = 0

# For accuracy metrics
true_positives = 0  # Correctly identified scams (grade >= 3 for scam images)
true_negatives = 0  # Correctly identified legit (grade < 3 for legit images)
false_positives = 0  # Incorrectly flagged as scam (grade >= 3 for legit images)
false_negatives = 0  # Missed scams (grade < 3 for scam images)

with open("logs/evaluation_log_20250519_223449.json", "r") as f:
    data = json.load(f)

for item in data:
    total_files += 1
    image_path = item["image"]
    is_scam = "scam" in image_path.lower()
    print(image_path)
    
    response_str = item["response"]
    # Remove code block markers and whitespace
    response_str = re.sub(r"^```json|```$", "", response_str.strip(), flags=re.MULTILINE).strip()
    try:
        response_json = json.loads(response_str)
        scam_prob = response_json.get("Scam Probability")
        print(f"Scam Probability: {scam_prob}")
        
        if is_scam:
            scam_grade_counts[scam_prob] += 1
            if scam_prob >= 3:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            legit_grade_counts[scam_prob] += 1
            if scam_prob < 3:
                true_negatives += 1
            else:
                false_positives += 1
                
    except Exception as e:
        print(f"Failed to parse response as JSON: {e}")
        failed_parses += 1
    print()

# Print statistics
print("\n=== Statistics ===")
print(f"Total files evaluated: {total_files}")
print(f"Failed parses: {failed_parses}")

# Legit images statistics
print("\nLegitimate Images Grade Distribution:")
total_legit = sum(legit_grade_counts.values())
for grade in sorted(legit_grade_counts.keys()):
    count = legit_grade_counts[grade]
    percentage = (count / total_legit) * 100 if total_legit > 0 else 0
    print(f"Grade {grade}: {count} files ({percentage:.1f}%)")

# Scam images statistics
print("\nScam Images Grade Distribution:")
total_scam = sum(scam_grade_counts.values())
for grade in sorted(scam_grade_counts.keys()):
    count = scam_grade_counts[grade]
    percentage = (count / total_scam) * 100 if total_scam > 0 else 0
    print(f"Grade {grade}: {count} files ({percentage:.1f}%)")

# Accuracy metrics
print("\n=== Accuracy Metrics ===")
total_legit_accuracy = (true_negatives / total_legit * 100) if total_legit > 0 else 0
total_scam_accuracy = (true_positives / total_scam * 100) if total_scam > 0 else 0
total_accuracy = ((true_positives + true_negatives) / total_files * 100) if total_files > 0 else 0

print(f"True Positives (correctly identified scams): {true_positives}")
print(f"True Negatives (correctly identified legit): {true_negatives}")
print(f"False Positives (incorrectly flagged as scam): {false_positives}")
print(f"False Negatives (missed scams): {false_negatives}")
print(f"\nLegitimate Images Accuracy: {total_legit_accuracy:.1f}%")
print(f"Scam Images Accuracy: {total_scam_accuracy:.1f}%")
print(f"Overall Accuracy: {total_accuracy:.1f}%")