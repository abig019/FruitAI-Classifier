import pandas as pd # type: ignore
import numpy as np # type: ignore

np.random.seed(42)
num_fruits = 500

# ── FEATURES ──────────────────────────────────────────────

size          = np.random.uniform(3.0, 10.0, num_fruits)
weight        = np.random.uniform(50.0, 300.0, num_fruits)
colour_score  = np.random.uniform(1.0, 10.0, num_fruits)
firmness      = np.random.uniform(1.0, 10.0, num_fruits)
sugar_content = np.random.uniform(1.0, 10.0, num_fruits)
blemish_score = np.random.uniform(0.0, 10.0, num_fruits)

# ── TARGET ────────────────────────────────────────────────

quality = []
for i in range(num_fruits):
    if (firmness[i] < 3.0 or
        blemish_score[i] > 7.5 or
        colour_score[i] < 2.5):
        quality.append('Rotten')
    elif (firmness[i] > 7.5 and
          colour_score[i] < 4.5 and
          sugar_content[i] < 4.0):
        quality.append('Fresh')
    else:
        quality.append('Ripe')

quality = np.array(quality)

df = pd.DataFrame({
    'size'          : np.round(size, 2),
    'weight'        : np.round(weight, 2),
    'colour_score'  : np.round(colour_score, 2),
    'firmness'      : np.round(firmness, 2),
    'sugar_content' : np.round(sugar_content, 2),
    'blemish_score' : np.round(blemish_score, 2),
    'quality'       : quality
})

df.to_csv('fruit_data.csv', index=False)
print("✅ Dataset created!")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print()
print(df['quality'].value_counts())
print()
print(df.head())