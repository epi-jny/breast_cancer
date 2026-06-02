"""Analyse du `difficult_negative_case` au niveau sein (patient_id × laterality),
chez les patients sans cancer.

`difficult_negative_case` = cas négatif difficile à diagnostiquer.
On exclut tout patient avec cancer (où qu'il soit) pour rester sur une
population strictement négative. Granularité = sein : CC et MLO d'un même
sein partagent la même valeur de DNC, donc l'image-level dupliquerait
chaque sein 2× et gonflerait artificiellement les corrélations.

BIRADS dans ce dataset RSNA (train uniquement) :
    0 = follow-up requis  |  1 = négatif  |  2 = normal
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


data = pd.read_csv('data/raw/extract_dataset3/train.csv')

# ─── Patients sans cancer ───────────────────────────────────────────────────
no_cancer_pids = data.groupby('patient_id')['cancer'].max().eq(0)
no_cancer_pids = no_cancer_pids[no_cancer_pids].index

# ─── Agrégation sein-level (patient × laterality) ───────────────────────────
data_neg = data[data['patient_id'].isin(no_cancer_pids)]
breasts_all = data_neg.groupby(['patient_id', 'laterality'], as_index=False).agg({
    'age': 'first',           # patient-level → identique sur les 2 vues du sein
    'density': 'first',
    'implant': 'first',
    'BIRADS': 'max',          # rare mais peut varier image→image
    'difficult_negative_case': 'max',
})
breasts = breasts_all.dropna(subset=['density'])
n_nan = len(breasts_all) - len(breasts)
print(f"Seins sans cancer : {len(breasts_all)}  |  density connue : {len(breasts)}  |  NaN : {n_nan}")
print(f"  dont DNC=True : {breasts['difficult_negative_case'].sum()} "
      f"({100 * breasts['difficult_negative_case'].mean():.1f} %)")

# ─── Analyse des NaN density ────────────────────────────────────────────────
nan_d = breasts_all[breasts_all['density'].isna()]
print("\nProfil des seins sans density renseignée :")
print(f"  taux DNC NaN : {nan_d['difficult_negative_case'].mean():.3f}   vs   density connue : {breasts['difficult_negative_case'].mean():.3f}")
print(f"  age moyen    : {nan_d['age'].mean():.1f}                     vs   density connue : {breasts['age'].mean():.1f}")

# ─── Tests χ² ───────────────────────────────────────────────────────────────
# Capture les associations non-linéaires (que la corrélation Pearson rate).
print("\nTests χ² (DNC × indicateurs) :")
for col in ['density', 'BIRADS', 'implant']:
    tab = pd.crosstab(breasts[col], breasts['difficult_negative_case'])
    chi2, p, _, _ = chi2_contingency(tab)
    print(f"  {col:10s} × DNC : χ²={chi2:7.2f}  p={p:.4g}")

# ─── Heatmap des corrélations ───────────────────────────────────────────────
# BIRADS one-hot car 0/1/2 n'est pas un ordre naturel ici (0=suspect, 1/2=clean).
features = ['age', 'implant', 'difficult_negative_case']
ohe = pd.concat([
    breasts[features].astype(float),
    pd.get_dummies(breasts['density'], prefix='density'),
    pd.get_dummies(breasts['BIRADS'].astype('Int64'), prefix='BIRADS'),
], axis=1)

plt.figure(figsize=(9, 7))
sns.heatmap(ohe.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title("Corrélations chez les seins sans cancer\n(1 ligne = 1 sein)")
plt.tight_layout()
plt.savefig('analyse_exploratoire/image/correlations_difficult_negative.png', dpi=150, bbox_inches='tight')
plt.show()

# ─── Distributions complémentaires ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (a) Taux DNC par densité
rates = breasts.groupby('density')['difficult_negative_case'].mean().sort_index()
axes[0].bar(rates.index, rates.values, color='salmon')
axes[0].set_title("Taux DNC par densité")
axes[0].set_xlabel("density")
axes[0].set_ylabel("proportion DNC")

# (b) Cross-tab BIRADS × DNC (comptes)
ct = pd.crosstab(breasts['BIRADS'].astype('Int64'), breasts['difficult_negative_case'])
sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title("BIRADS × DNC (comptes)\n0=follow-up | 1=négatif | 2=normal")
axes[1].set_xlabel("difficult_negative_case")
axes[1].set_ylabel("BIRADS")

plt.tight_layout()
plt.savefig('analyse_exploratoire/image/distributions_difficult_negative.png', dpi=150, bbox_inches='tight')
plt.show()
