import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

df = pd.read_csv('fruit_data.csv')

print("── Shape ───────────────────────────")
print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print()

print("── Missing values ──────────────────")
print(df.isnull().sum())
print()

print("── Quality distribution ────────────")
print(df['quality'].value_counts())
print()

print("── Average per quality ─────────────")
print(df.groupby('quality').mean().round(2))
print()

# Chart 1 — Distribution
plt.figure(figsize=(6, 4))
colors = ['#27ae60', '#f39c12', '#e74c3c']
df['quality'].value_counts().plot(kind='bar', color=colors)
plt.title('Fruit Quality Distribution')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('quality_distribution.png')
plt.show()

# Chart 3 — Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df.drop('quality', axis=1).corr(),
            annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

print("✅ All charts saved!")