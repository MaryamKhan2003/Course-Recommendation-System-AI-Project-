"""
Simple Matplotlib Visualization - Single plot as required
Run: python visualize_model.py
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from recommender import KNNRecommender  # Import the class

# Load model
with open('model/knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create single plot
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
cv_scores = [model.cv_accuracy_mean*100, model.cv_precision_mean*100,
             model.cv_recall_mean*100, model.cv_f1_mean*100]
cv_stds = [model.cv_accuracy_std*100, model.cv_precision_std*100,
           model.cv_recall_std*100, model.cv_f1_std*100]

# Bar chart with error bars
bars = ax.bar(metrics, cv_scores, yerr=cv_stds, capsize=8, 
              color=['#2E86AB', '#A23B72', '#F18F01', '#1B998B'],
              edgecolor='black', linewidth=1.5)

ax.set_ylim(0, 100)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('KNN Model Performance (Mean ± Std Dev)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, score, std in zip(bars, cv_scores, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{score:.1f}%\n±{std:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('knn_performance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Visualization saved to: knn_performance.png")
print("📊 KNN Performance - Mean ± Std Deviation")