import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean CRSP data
crsp = pd.read_csv(
    "excess_return_2020_2024_updated.csv",
    dtype={'CIK': str},
    parse_dates=['date'],
    converters={
        'RET': lambda x: pd.to_numeric(x, errors='coerce'),
        'sprtrn': lambda x: pd.to_numeric(x, errors='coerce')
    }
)
crsp = crsp.dropna(subset=['RET', 'sprtrn'])
crsp['CIK'] = crsp['CIK'].astype(str).str.zfill(10)

# Load and clean text data (use correct column names)
text = pd.read_csv("final_results.csv", dtype={'CIK': str})
text['CIK'] = text['CIK'].astype(str).str.zfill(10)
text['filing_date'] = pd.to_datetime(text['filename'].str[:8], format='%Y%m%d', errors='coerce')
text = text[text['filename'].str.contains('10-K', case=False, na=False)]

# Use correct column names for word lists
text = text.dropna(subset=['H4N-INF_pct', 'FIN-NEG_pct', 'filing_date'])  # Fix here

# Merge with date tolerance
merged = pd.merge_asof(
    text.sort_values('filing_date'),
    crsp.sort_values('date'),
    left_on='filing_date',
    right_on='date',
    by='CIK',
    direction='nearest',
    tolerance=pd.Timedelta(days=1)
)

if merged.empty:
    raise ValueError("No overlapping CIK/date pairs. Check your data alignment!")

# Calculate excess returns (ensure this aligns with your intended calculation)
merged['excess_ret'] = (
    (1 + merged['RET']).rolling(4, min_periods=1).apply(np.prod) -
    (1 + merged['sprtrn']).rolling(4, min_periods=1).apply(np.prod)
)

# Create quintiles for each word list
merged['quintile_H4N'] = pd.qcut(merged['H4N-INF_pct'], q=5, labels=False, duplicates='drop')  # Fix here
merged['quintile_Fin'] = pd.qcut(merged['FIN-NEG_pct'], q=5, labels=False, duplicates='drop')  # Fix here

# Calculate medians for each quintile
h4n_medians = merged.groupby('quintile_H4N')['excess_ret'].median().reset_index()
h4n_medians.rename(columns={'quintile_H4N': 'quintile'}, inplace=True)
h4n_medians['word_list'] = 'H4N-Inf'

fin_medians = merged.groupby('quintile_Fin')['excess_ret'].median().reset_index()
fin_medians.rename(columns={'quintile_Fin': 'quintile'}, inplace=True)
fin_medians['word_list'] = 'Fin-Neg'

combined_medians = pd.concat([h4n_medians, fin_medians])

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    x='quintile',
    y='excess_ret',
    hue='word_list',
    data=combined_medians,
    marker='o',
    ci=None  # Remove confidence intervals if desired
)
plt.title('Figure 1. Median Filing Period Excess Return by Quintile')
plt.xlabel('Quintile')
plt.ylabel('Median Excess Return')
plt.legend(title='Word List')
plt.savefig('quintile_plot.png', dpi=300, bbox_inches='tight')
plt.show()