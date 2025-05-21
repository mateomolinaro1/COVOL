import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from statsmodels.stats.diagnostic import het_arch,acorr_ljungbox
from statsmodels.stats.diagnostic import acorr_ljungbox

"""
Script contenant les méthodes statistiques nécessaires à la réplication des tableaux du papier statistique.
"""


class StatisticalMetrics:

    def mse(self, true, est, axis=1):
        """
        Calcule l'erreur quadratique moyenne (MSE)
        """
        return np.mean((true - est) ** 2, axis=axis)

    def R2(self, true, est, baseline, axis=1):
        """
        Calcule le R² selon la formule :
        """
        mse_diff = self.mse(true, est, axis=axis)
        mse_base = self.mse(true, baseline, axis=axis)
        return np.mean(1 - mse_diff / mse_base)

    def R2_x(self, X_true, X_est):
        """
        Pour le facteur x, la baseline est 1 (un vecteur constant de 1).
        X_true et X_est ont la forme (R, T).
        """
        baseline = np.ones_like(X_true)
        return self.R2(X_true, X_est, baseline, axis=1)

    def R2_s(self, S_true, S_est):
        """
        Pour les facteurs loadings s, la baseline est la moyenne de S_true pour chaque réplique.
        S_true et S_est ont la forme (R, N).
        """
        baseline = np.mean(S_true, keepdims=True) #Calcule la moyenne des S
        baseline = np.broadcast_to(baseline, S_est.shape) #Étend la baseline pour qu’elle ait exactement la même forme que S_est
        return self.R2(S_true, S_est, baseline, axis=1)

    def avg_corr_e2(self, E, X_est_list, S_est_list):
        """
        Calcule la corrélation moyenne des carrés des résidus, réstandardisés par g_hat
        """
        R, T, N = E.shape
        corr_list = []

        for r in range(R):
            e = E[r] # Dimensions (T, N)
            x = X_est_list[r]  # (T,)
            s = S_est_list[r]  # (N,)

            #On réstandardiser les résidus avec les variances conditionnelles estimées
            g = np.outer(x, s) + (1 - s)
            e2_std = (e ** 2) #/ g # Résidus standardisés

            corr_mat = np.corrcoef(e2_std, rowvar=False)  # Corr entre actifs

            # Moyenne des éléments hors diagonale (partie triangulaire sup)
            iu = np.triu_indices(N, k=1)
            corr_list.append(np.mean(corr_mat[iu]))

        return np.mean(corr_list)

    def empirical_v(self, X):
        """
        Calcule la variance empirique de x (pour chaque scénario, x est de taille T)
        puis moyenne sur R scénarios.
        """
        return np.mean(np.var(X, axis=1))

    def compute_z_crit(self,alpha=0.05):
        """ Calcul la valeur critique d'une distribution normale"""
        return norm.ppf(1 - alpha / 2)

    def compute_rejection_rate(E, X_est_list, S_est_list, alpha=0.05):
        """ Calcul le taux de rejet des tests statistiques éfféctués (dans le cas de simulation Monte Carlo)"""

        R = len(E)
        metrics = StatisticalMetrics()
        count_reject = 0

        for r in range(R):
            e = E[r]
            x_est = X_est_list[r]
            s_est = S_est_list[r]
            _, reject_H0, _ = metrics.dstats_xi_test(e, x_est, s_est, alpha)
            if reject_H0:
                count_reject += 1

        rejection_rate = count_reject / R
        print(f"\n📊 Taux de rejet du test de Global COVOL : {rejection_rate:.3f}")
        return rejection_rate

    def dstats_xi_test(self, e, x_est, s_est, alpha=0.05):
        """
        Implémente la statistique de test du papier (section 6) pour tester H0 : pas de Global COVOL.
        """

        T, N = e.shape

        e_std = e

        # Centrage : (e^2 - 1)
        e2_centered = e_std ** 2 - 1  # shape (T, N)

        # Numérateur
        num = 0
        for i in range(N): # Itération sur chaque actif
            for j in range(i): # Itération sur chaque pas de temps T
                num += np.sum(e2_centered[:, i] * e2_centered[:, j])

        # Dénominateur
        denom = np.sum(e2_centered ** 2)

        # Facteur de normalisation
        norm_coeff = np.sqrt((N * T)) / ((N - 1) / 2)

        xi = norm_coeff * (num / denom)

        # Seuil z-critique pour un test bilatéral
        z_crit = self.compute_z_crit(alpha=0.05) # pour alpha = 0.05
        reject_H0 = np.abs(xi) > z_crit
        conclusion = " Global COVOL détecté (H₀ rejetée)" if reject_H0 else " Pas de preuve de Global COVOL (H₀ non rejetée)"

        # Affichage
        print(f"ξ (test statistic) = {xi:.3f}")
        print(f"Seuil critique ±{z_crit}, H₀ est {'rejetée' if reject_H0 else 'non rejetée'}")
        print(f"{conclusion}")

        return xi, reject_H0, conclusion

class DescriptiveStats:
    """Classe contenant les méthodes de calcul des stats utile à la réplication du tableau C.1 en appendix"""
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns

    def min_return(self):
        return self.returns.min() * 100

    def max_return(self):
        return self.returns.max()*100

    def mean_return(self):
        return self.returns.mean() *100

    def std_return(self):
        return self.returns.std() *100

    def skew(self):
        return self.returns.apply(skew)

    def kurtosis(self):
        return self.returns.apply(kurtosis)

    def robust_skewness(self):
        q10 = self.returns.quantile(0.10)
        q90 = self.returns.quantile(0.90)
        q50 = self.returns.quantile(0.50)
        return (q90 + q10 - 2 * q50) / (q90 - q10)

    def robust_kurtosis(self):
        """
        Moors (1988) kurtosis: ((Q7/8-Q5/8)+(Q3/8-Q1/8)) / (Q6/8-Q2/8) - 1.23
        where 1.23 is the value for a Gaussian.
        """
        # quantiles aux octiles
        q125 = self.returns.quantile(0.125)
        q375 = self.returns.quantile(0.375)
        q625 = self.returns.quantile(0.625)
        q875 = self.returns.quantile(0.875)
        # quartiles pour le denom
        q25  = self.returns.quantile(0.25)
        q75  = self.returns.quantile(0.75)

        raw = ( (q875 - q625) + (q375 - q125) ) / (q75 - q25)
        baseline = 1.23  # Moors coefficient pour N(0,1)
        return raw - baseline

    def ar1(self):
        """
        Pour chaque série de self.returns, calcule la statistique Box–Pierce au lag=1
        et sa p-value, comme dans le Tableau C.1 du papier.
        """
        def safe_bp1(x: pd.Series):
            y = x.dropna()
            if len(y) < 3:
                return np.nan
            # acorr_ljungbox avec boxpierce=True renvoie BP et LB
            res = acorr_ljungbox(y, lags=[1], boxpierce=True, return_df=True)
            stat = res['bp_stat'].iloc[0]
            return stat

        return self.returns.apply(safe_bp1)

    def ar1_pvalue(self):
        """p-value pour le test de non autocorrélation AR(1) (Ljung-Box)"""

        def safe_pval(x):
            x_clean = x.dropna()
            if len(x_clean) >= 2:
                return acorr_ljungbox(x_clean, lags=[1], return_df=True)['lb_pvalue'].iloc[0]
            else:
                return np.nan

        return self.returns.apply(safe_pval)

    def arch_test_stat(self):
        """Statistique du test ARCH(1)"""

        def safe_arch_stat(x):
            x_clean = x.dropna()
            if len(x_clean) >= 2:
                return het_arch(x_clean, nlags=1)[0]
            else:
                return np.nan

        return self.returns.apply(safe_arch_stat)

    def arch_test_pvalue(self):
        """p-value du test ARCH(1)"""

        def safe_arch_pval(x):
            x_clean = x.dropna()
            if len(x_clean) >= 2:
                return het_arch(x_clean, nlags=1)[1]
            else:
                return np.nan

        return self.returns.apply(safe_arch_pval)

    def compute_all(self, round_digits=3):
        """
        Regroupe toutes les stats dans une table
        """
        df = pd.DataFrame({
            "Min.": self.min_return().round(round_digits),
            "Mean": self.mean_return().round(round_digits),
            "Max.": self.max_return().round(round_digits),
            "S.D.": self.std_return().round(round_digits),
            "Rob. Sk.": self.robust_skewness().round(round_digits),
            "Rob. Kr.": self.robust_kurtosis().round(round_digits),
            "AR(1)": self.ar1().round(round_digits),
            "p-value AR(1)": self.ar1_pvalue().round(round_digits),
            "ARCH(1)": self.arch_test_stat().round(round_digits),
            "p-value ARCH": self.arch_test_pvalue().round(round_digits),
        })

        print(" Table C.1 – Statistiques descriptives des rendements")
        print(df.to_string())
        return df

