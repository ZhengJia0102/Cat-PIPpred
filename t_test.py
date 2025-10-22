import scipy.stats as stats
import numpy as np
from scipy.stats import shapiro





multifeat = np.array([0.691, 0.508, 0.830, 0.718, 0.360])
cat_pip = np.array([0.691, 0.520, 0.821, 0.738, 0.361])

print("=== shapiro ===")

_, p_multifeat = shapiro(multifeat)
_, p_cat_pip = shapiro(cat_pip)
print(f"MultifeatVotPIP-p: {p_multifeat:.4f}")
print(f"Cat-PIPpred-p: {p_cat_pip:.4f}")


differences = cat_pip - multifeat
_, p_diff = shapiro(differences)
print(f"matched-p: {p_diff:.4f}")

print("\n=== t-test ===")

t_stat, p_value_two_tailed = stats.ttest_rel(cat_pip, multifeat)
p_value_one_tailed = p_value_two_tailed / 2

print(f"t = {t_stat:.4f}")
print(f"p = {p_value_one_tailed:.4f}")
print(f"d = {np.mean(differences):.4f}")


alpha = 0.05
if p_diff > 0.05:  
    if p_value_one_tailed < alpha and t_stat > 0:
        print("Cat-PIPpred》MultifeatVotPIP (p < 0.05)")
    else:
        print("not significant ")
else:  
    print("warning")
    
    
    w_stat, p_wilcoxon = stats.wilcoxon(cat_pip, multifeat, alternative='greater')
    print(f"\n=== Wilcoxon ===")
    print(f"Wilcoxon = {w_stat:.4f}")
    print(f"p = {p_wilcoxon:.4f}")
    
    if p_wilcoxon < alpha:
        print("Cat-PIPpred>MultifeatVotPIP (p < 0.05)")
    else:
        print("not significant")