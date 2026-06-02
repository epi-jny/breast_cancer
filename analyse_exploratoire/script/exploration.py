import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


def aggregate_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par patient. MAX sur les indicateurs cliniques (un sein malade
    => patient malade). Patients aux features patient-level incohérentes exclus."""
    patient_level = ['age', 'site_id', 'density', 'implant']
    bad = set()
    for col in patient_level:
        n = df.groupby('patient_id')[col].nunique(dropna=False)
        bad.update(n[n > 1].index)

    df = df[~df['patient_id'].isin(bad)]
    agg = {c: 'first' for c in patient_level} | {
        'cancer': 'max', 'biopsy': 'max', 'invasive': 'max', 'BIRADS': 'max',
    }
    return df.groupby('patient_id', as_index=False).agg(agg)


# ─── Données ────────────────────────────────────────────────────────────────
data = pd.read_csv('data/raw/extract_dataset3/train.csv')
patients_all = aggregate_by_patient(data)
patients = patients_all.dropna(subset=['density'])
n_nan = len(patients_all) - len(patients)
print(f"Patients total : {len(patients_all)}  |  density connue : {len(patients)}  |  NaN : {n_nan}")

# ─── Analyse des NaN density ────────────────────────────────────────────────
# On compare les patients sans density à ceux avec density renseignée pour voir
# si l'absence est informative (biais de site, d'âge, de prévalence cancer…).
nan_d = patients_all[patients_all['density'].isna()]
print("\nProfil des patients sans density renseignée :")
print(f"  taux cancer  NaN : {nan_d['cancer'].mean():.3f}   vs   density connue : {patients['cancer'].mean():.3f}")
print(f"  age moyen    NaN : {nan_d['age'].mean():.1f}     vs   density connue : {patients['age'].mean():.1f}")
print(f"  répartition par site_id (NaN) : {nan_d['site_id'].value_counts().to_dict()}")
print(f"  répartition par site_id (OK)  : {patients['site_id'].value_counts().to_dict()}")

# ─── Tests χ² (associations catégorielles non-linéaires) ────────────────────
# Pearson ne capte que le linéaire ; χ² détecte toute déviation à l'indépendance.
print("\nTests χ² (density × indicateurs cliniques) :")
for col in ['cancer', 'biopsy', 'invasive', 'BIRADS']:
    tab = pd.crosstab(patients['density'], patients[col])
    chi2, p, _, _ = chi2_contingency(tab)
    print(f"  density × {col:10s} : χ²={chi2:7.2f}  p={p:.4g}")

# ─── Heatmap des corrélations (density en one-hot) ──────────────────────────
features = ['age', 'cancer', 'biopsy', 'invasive', 'implant', 'BIRADS']
ohe = pd.concat([patients[features],
                 pd.get_dummies(patients['density'], prefix='density')], axis=1)

plt.figure(figsize=(8, 6))
sns.heatmap(ohe.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title("Corrélations entre variables (1 ligne = 1 patient)")
plt.tight_layout()
plt.savefig('analyse_exploratoire/image/correlations_patient.png', dpi=150, bbox_inches='tight')
plt.show()

# ─── Distributions complémentaires ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) Effectifs par density
counts = patients['density'].value_counts().sort_index()
axes[0].bar(counts.index, counts.values, color='steelblue')
axes[0].set_title("Effectifs par densité")
axes[0].set_xlabel("density")
axes[0].set_ylabel("nb patients")

# (b) Taux de cancer par density
rates = patients.groupby('density')['cancer'].mean().sort_index()
axes[1].bar(rates.index, rates.values, color='salmon')
axes[1].set_title("Taux de cancer par densité")
axes[1].set_xlabel("density")
axes[1].set_ylabel("proportion cancer")

# (c) Age selon density (box plot)
sns.boxplot(data=patients, x='density', y='age',
            order=sorted(patients['density'].unique()), ax=axes[2])
axes[2].set_title("Âge selon la densité")

plt.tight_layout()
plt.savefig('analyse_exploratoire/image/distributions_patient.png', dpi=150, bbox_inches='tight')
plt.show()
