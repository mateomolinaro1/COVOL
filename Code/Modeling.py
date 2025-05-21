import numpy as np
from scipy.stats import rankdata
from numpy.linalg import eigh, norm
from scipy.optimize import minimize
from statsmodels.tsa.ar_model import AutoReg
from arch import arch_model
import pandas as pd
from sklearn.decomposition import PCA

import statsmodels.api as sm

from Fonction_numba import estimate_grid_search
from data_management import ResultManager


"""
Script contenant les différentes classes de modélisation économétrique nécessaires à la réplication du papier.
"""


class GlobalCovolEstimation:
    """
    Classe d'estimation de facteur global COVOL en fonction de la matrice de résidus standardisés fournie
    """

    def __init__(self, e,grid_search = False):
        self.e = e
        self.T, self.N = e.shape
        self.x, self.s = self.initial_estimates()  # Estimation initiale par ACP sur corrélation de rang
        self.normalize_params()
        self.grid_search = grid_search #Possibilité d'utilisé une optimisation grid search


    def initial_estimates(self):
        """
        Initialisation.

        - x0 : moyenne transversale des e_{i,t}^2 pour chaque t,
               puis normalisation pour avoir mean(x0)=1
        - s0 : premier vecteur propre de la matrice de corrélation
               des e_{i,t}^2 (sans clip à [0,1]).
               On prend la valeur absolue ou on gère a minima
               les valeurs négatives si on veut imposer s_i>=0.
        """

        # Construction de x0 par la moyenne transversale
        E2 = self.e ** 2

        x0 = np.zeros(self.T)
        for t in range(self.T):
            valid_idx = ~np.isnan(E2[t])
            if np.any(valid_idx):

                x0[t] = E2[t, valid_idx].mean() # Moyenne des résidus éxistants
            else:
                x0[t] = 1.0 #Si aucun actif valide à la date T (pas probable normalement)

        # Normalisation pour avoir moy(x0)=1
        mean_x0 = x0.mean()
        if mean_x0 > 0:
            x0 /= mean_x0
        else:
            x0[:] = 1.0


        # Construction de s0 par ACP
        E2_df = pd.DataFrame(E2-1.0).dropna(axis=0, how='any') # Pour construire la matrice de corrélation, on ignore chaque NaN => supprime les dates où au moins un actif n’a pas de NA


        if E2_df.shape[0] >= 2:
            # Calcul de la matrice de corrélation
            ranks = np.apply_along_axis(rankdata, 0, E2_df.values)
            R_corr = np.corrcoef(ranks, rowvar=False)

            # Decomposition en valeurs propres
            eigvals, eigvecs = eigh(R_corr)
            idx_max = np.argmax(eigvals)
            s0_raw = eigvecs[:, idx_max]  # vecteur propre principal

            # On prend la valeur absolue pour supprimer l'ambiguïté de signe dans un vecteur propre.
            s0_abs = np.abs(s0_raw)

            # Normalisation
            norm_s0 = np.linalg.norm(s0_abs)
            if norm_s0 > 0:
                s0 = s0_abs / norm_s0
            else:
                # si bug, arrive pas normalement
                s0 = np.ones(self.N) / np.sqrt(self.N)

        else:
            # S’il n’y a pas assez de dates pour faire ACP,
            s0 = np.ones(self.N) / np.sqrt(self.N)

        return x0, s0


    def normalize_params(self):
        """ Normalisation des paramètres """

        self.x /= np.mean(self.x) # moyenne de x égale à 1
        self.s /= np.linalg.norm(self.s) #  s's = 1, norme de s égale à 1


    def log_likelihood(self, s, x):
        """
        Calcule la log-vraisemblance en ignorant les NaN.
        """

        ll_val = 0.0
        for t in range(self.T):
            valid_idx = ~np.isnan(self.e[t]) # actifs disponibles à la date t
            if not np.any(valid_idx):
                continue # Si tous NaN ce jour-là, on l'ignore complètement

            g_t = s[valid_idx] * x[t] + (1 - s[valid_idx]) # Calcul de g pour les actifs valides
            e_t = self.e[t, valid_idx]  # résidus valides

            ll_val += -0.5 * np.sum(np.log(g_t) + (e_t ** 2 / g_t))  # Contribution à la log-vraisemblance

        return ll_val


    def update_x(self, s):
        """
         Mise à jour de x pour chaque période t en optimisant la log-vraisemblance.
        """
        new_x = np.zeros(self.T)

        for t in range(self.T):
            valid_idx = ~np.isnan(self.e[t])
            if not np.any(valid_idx):
                new_x[t] = 1.0 # Si pas d'actifs valides ce jour-là, n'arrive pas normalement
                continue

            #Fonction objective
            def obj(x_val):
                x_sca = np.exp(x_val[0])
                g_t = s[valid_idx] * x_sca + (1 - s[valid_idx])
                e_t = self.e[t, valid_idx]
                ll = 0.5 * np.sum(np.log(g_t) + (e_t ** 2 / g_t))
                return ll

            # Minimisation
            res = minimize(obj, x0=np.array(np.log(self.x[t])), bounds=[(1e-3, 8)], method="L-BFGS-B") #Borne volontairement grande
            new_x[t] = np.exp(res.x[0])

        return new_x

    def update_s(self, x):
        """
        Mise à jour de s pour chaque actif i en optimisant la log-vraisemblance.
        """

        new_s = np.zeros(self.N)

        # Itération sur chaque actif
        for i in range(self.N):
            valid_idx = ~np.isnan(self.e[:, i]) # dates où l'actif i est valide
            if not np.any(valid_idx):
                new_s[i] = 0.0
                continue

            # Fonction objective
            def obj(s_val):
                s_sca = s_val[0]
                e_i = self.e[valid_idx, i] # e_t sur les dates valides
                x_i = x[valid_idx]
                g_i = s_sca * x_i + (1 - s_sca)
                return 0.5 * np.sum(np.log(g_i) + (e_i ** 2 / g_i))

            # Bornes [0,1], contrainte
            res = minimize(obj, x0=np.array(self.s[i]), bounds=[(0, 1)], method="L-BFGS-B")
            new_s[i] = res.x[0]

        return new_s

    def estimate(self):
        """
        Wrapper pour gérer les appels avec Grid search ou scipy
        """

        if self.grid_search: # Lancement de l'estimation avec grid search et numba
            grid_x = np.linspace(0.1, 100, 1000)
            grid_s = np.linspace(0, 1, 100)
            x, s = estimate_grid_search(self.e,  self.s,self.x,  grid_x, grid_s, max_iter=30, tol=1e-6)
        else : # Lancement avec optimisation scipy
            x, s  = self.estimate_scipy()
        return x, s


    def estimate_scipy(self, tol=1e-4, max_iter=150):
        """
        Algorithme d'estimation itératif de type EM
        """

        ll_old = self.log_likelihood(self.s, self.x) #Calcul de la log vraisemblance initial, avec les paramètres initiaux

        # Itération en fonction du paramètre
        for it in range(max_iter):
            print(f"Itération {it}")
            self.x = self.update_x(self.s)
            self.s = self.update_s(self.x)
            self.normalize_params()

            ll_new = self.log_likelihood(self.s, self.x)
            if np.abs(ll_new - ll_old) < tol: # On sort de l'optimisation si la différence sur la nouvelle vraisemblance est minime
                print(f"Convergence à l'itération {it}")
                break
            ll_old = ll_new
        return self.x, self.s


    @staticmethod
    def estimate_global_covol_full_sample(e_df: pd.DataFrame, grid_search=False):
        """
        Calcule le Global COVOL (x_t et s_i) pour T dates et N actifs,
        en une seule fois sur l’échantillon complet.

        Pour les dates où l'actif est ponctuellement indisponible,
        on conserve la ligne (la date) dans e_df, mais on laisse un NaN
        qui sera ignoré dans la somme de log-vraisemblance
        (voir les méthodes).
        """

        # On retire seulement les colonnes (actifs) qui sont 100% NaN
        data_clean = e_df.loc[:, e_df.notna().any(axis=0)].copy()
        print("Dimensions initiales :", e_df.shape)
        print("Dimensions après suppression des actifs 100% NaN :", data_clean.shape)

        # Conversion en numpy
        e = data_clean.values

        # Estimation via GlobalCovolEstimation
        model = GlobalCovolEstimation(e, grid_search=grid_search)
        x_est, s_est = model.estimate()

        # Reconstruit x_est et s_est en pandas Series avec le même index/colonnes
        x_series = pd.Series(x_est, index=data_clean.index, name="Global_COVOL")
        s_series = pd.Series(s_est, index=data_clean.columns, name="Loadings")

        return x_series, s_series


    @staticmethod
    def estimate_loadings_monthly(e_df, grid_search=False, start_date=None):
        """
        Estime des global COVOL loadings s_i à chaque fin de mois, en utilisant
        toutes les données disponibles jusqu'à cette fin de mois (expanding window).
        """

        # On restreint à start_date
        if start_date is not None:
            e_df = e_df.loc[e_df.index >= start_date].copy()

        # On identifie toutes les fins de mois dans l’index
        month_ends = e_df.resample("M").last().index  # liste des dates = fin de mois

        # DataFrame pour stocker les loadings
        s_est_ts = pd.DataFrame(index=month_ends, columns=e_df.columns, dtype=float)

        # Boucle sur chaque fin de mois
        for date in month_ends:
            print(f"Estimation des loadings pour la fin du mois : {date.strftime('%Y-%m-%d')}")

            # Sous-ensemble du DataFrame jusqu'à (et incluant) cette date
            sub_data = e_df.loc[:date]

            # Supprime éventuellement les actifs qui sont 100% NaN
            sub_data = sub_data.loc[:, sub_data.notna().any(axis=0)]

            # Estimation globale
            model = GlobalCovolEstimation(sub_data.values, grid_search=grid_search)
            x_hat, s_hat = model.estimate()

            # Reconstruit s_hat dans une Series pour savoir quelles colonnes on avait
            s_series = pd.Series(s_hat, index=sub_data.columns)

            # Stocke dans s_est_ts, pour la date courante
            s_est_ts.loc[date, sub_data.columns] = s_series

        return s_est_ts




class EconometricModels:
    """
    Classe contenant les méthodes pour ajuster des modèles économétriques
    """

    @staticmethod
    def build_design(y: pd.Series,X_dict,add_const=False,lags = None,diff_list = None,trunc_pos= None) :
        """
        Renvoie un DataFrame propre
        """

        df = pd.DataFrame({"y": y})
        for name, s in X_dict.items():
            s2 = s.copy()
            if diff_list and name in diff_list:
                s2 = s2.diff()
            if trunc_pos and name in trunc_pos:
                s2 = s2.clip(lower=0)
            if lags and name in lags:
                s2 = s2.shift(lags[name])
            df[name] = s2
        if add_const:
            df["const"] = 1.0
        return df.dropna()

    @staticmethod
    def fit_ols(design_df, hac_lags = 1,show = False,use_HAC = False):
        """
        Méthode de régression OLS avec y la première colonne du df
        """
        y = design_df.iloc[:, 0]
        X = design_df.iloc[:, 1:]
        if use_HAC :
            model = sm.OLS(y, X).fit(cov_type="HAC",
                                 cov_kwds={"maxlags": hac_lags})
        else :
            model = sm.OLS(y, X).fit()

        if show:
            print(model.summary())
        return model

    @staticmethod
    def fit_ar1(y: pd.Series,lags =1 ):
        """
        Ajuste un modèle AR(1) sur une série temporelle.
        """
        model = AutoReg(y.dropna(), lags=lags, old_names=False).fit()
        return model.resid

    @staticmethod
    def fit_garch11(y: pd.Series,p=1,q=1):
        """
        Ajuste un modèle GARCH(1,1) directement sur la série (rendements).
        """
        model = arch_model(y.dropna(), vol="Garch", p=p, q=q, mean="Zero")
        res = model.fit(disp="off")
        return res

    @staticmethod
    def compute_standardized_residuals(resid,sigma_t):
        return resid/sigma_t

    @staticmethod
    def compute_pc1_cross_section1(returns: pd.DataFrame): ## Méthode à travailler, je ne suis pas qu'il faut faire comme ça
        """
        Calcule le premier facteur principal (PC1) en cross-section à chaque date t,
        à partir des rendements disponibles des ETF (PCA transversale).
        """
        pc1_series = pd.Series(index=returns.index, dtype=np.float64)

        for t in returns.index:
            row = returns.loc[t].dropna()
            if len(row) < 2:
                continue

            # Centrage du vecteur
            row_centered = row - row.mean()

            # Projection sur la "composante principale" (ici, la direction max de variation)
            # Comme on n'a qu'un vecteur, on peut prendre sa norme directionnelle
            pc1_score = row_centered.dot(row_centered)
            pc1_series[t] = pc1_score

        return pc1_series - pc1_series.mean()

    @staticmethod
    def compute_pc1_cross_section(returns: pd.DataFrame) -> pd.Series:
        """
        returns : DataFrame T×N (dates×actifs), avec éventuellement quelques NaN.
        On les remplace par la moyenne colonne avant PCA.
        Retourne la série PC1 centrée (moyenne nulle).

        """
        # On remplace les NaN par la moyenne de la colonne
        X = returns.copy()
        X = X.fillna(X.mean(axis=0))

        # On centre chaque colonne (rendements) pour que la PCA soit sur cov(X)
        Xc = X - X.mean(axis=0)

        # On lance la PCA pour n_components=1
        pca = PCA(n_components=1)
        pc1_scores = pca.fit_transform(Xc.values).flatten()

        # On met en Series et on centre la série
        pc1 = pd.Series(pc1_scores, index=returns.index)
        return pc1 - pc1.mean()


    @staticmethod
    def estimate_standardized_residuals(y: pd.Series, X_dict: dict):
        """
        Effectue :
        - régression de y sur les variables dans X_dict (par ex : {"acwi": series, "pc1": series, "lag": y.shift(1)})
        - GARCH(1,1) sur les résidus
        - standardisation
        """

        try:
            df_reg = pd.DataFrame(X_dict)
            df_reg["y"] = y
            df_reg = df_reg.dropna()

            reg = sm.OLS(df_reg["y"], sm.add_constant(df_reg.drop(columns=["y"]))).fit() # Régréssion avec constante

            # Modélisation de la variance conditionnelle des résidus avec GARCH
            resid = reg.resid
            rescaled_resid = resid * 100 #On rescale car valeur trop petite pour le modèle
            garch_res = EconometricModels.fit_garch11(rescaled_resid)
            sigma_t = garch_res.conditional_volatility / 100  #Variance conditionnelle des résidus, /100 pou re formater les données dans leur forme d'origine
            e_t = EconometricModels.compute_standardized_residuals(resid, sigma_t) #Résidus standardisés

            return resid, sigma_t, e_t

        except Exception as e:
            print(f"Erreur dans estimation résidus standardisés : {e}")
            return None, None, None


    @staticmethod
    def batch_estimate_residuals(returns: pd.DataFrame, acwi_returns: pd.Series, pc1: pd.Series):
        """
        Applique estimate_standardized_residuals à chaque colonne de `returns`.
        """

        residuals = pd.DataFrame(index=returns.index)
        volatilities = pd.DataFrame(index=returns.index)
        residual_standardized = pd.DataFrame(index=returns.index)

        for col in returns.columns:
            try:
                y = returns[col].dropna() # Recup la série d'ETF par pays
                common_index = y.index.intersection(acwi_returns.index).intersection(pc1.index) # Match les indices
                y_common = y.loc[common_index]

                # Modèle de regréssion, regression des rendements de l'etf, sur la première composante principale, le facteur marché, et le facteur d'autocorrelation
                X_dict = {
                    "acwi": acwi_returns.loc[common_index],
                    "pc1": pc1.loc[common_index],
                    "lag": y_common.shift(1)
                }

                resid, sigma_t, e_t = EconometricModels.estimate_standardized_residuals(y_common, X_dict) # residus, volatilité conditionnelle (GARCH 1,1) et résidus standardisés

                # Stockage
                if resid is not None:
                    residuals[col] = resid.reindex(returns.index)
                    volatilities[col] = sigma_t.reindex(returns.index)
                    residual_standardized[col] = e_t.reindex(returns.index)

            except Exception as e:
                print(f"Erreur pour {col} : {e}")

        return residuals, volatilities, residual_standardized


    @staticmethod
    def mean_cross_corr_monthly(returns: pd.DataFrame,
                                min_pairs: int = 1) -> pd.Series:

        means = []

        # boucle mois par mois
        for date, df_month in returns.resample("M"):
            # matrice de corrélation avec min_periods=2 pairwise
            corr_mat = df_month.corr(min_periods=2)

            # triangle supérieur hors diagonale
            vals = corr_mat.values
            iu = np.triu_indices_from(vals, k=1)
            tri = vals[iu]

            # on ne garde que les valeurs finies
            tri = tri[np.isfinite(tri)]

            if len(tri) >= min_pairs:
                means.append(tri.mean())
            else:
                means.append(np.nan)

        idx = returns.resample("M").last().index
        return pd.Series(means, index=idx, name="mean_corr")

    @staticmethod
    def diagnose_large_xt(x_series,
                          residual_std,
                          volatilities=None,
                          threshold=20,
                          top_n=10,
                          rm: "ResultManager" = None,
                          file_name="diag_xt_large"):
        """
        Crée un rapport Excel listant toutes les dates où x_t > threshold,
        avec les contributions par actif.
        """

        if rm is None:
            rm = ResultManager()
        path = rm.get_path(file_name)

        mask = x_series > threshold
        if mask.sum() == 0:
            print("Aucun x_t au-dessus du seuil.")
            return

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for date in x_series.index[mask]:

                # Infos globales sur la date
                e_t = residual_std.loc[date]
                valid = e_t.dropna()
                n_valid = valid.size

                info = pd.DataFrame({
                    "metric": ["x_t", "Nb actifs valides"],
                    "value": [x_series.loc[date], n_valid]
                })

                if volatilities is not None:
                    sig_t = volatilities.loc[date]
                    info.loc[len(info)] = ["sigma_min", sig_t.min()]


                # Top contributions
                e2 = valid ** 2
                contrib = (e2 / e2.sum()).sort_values(ascending=False)
                top = contrib.head(top_n).to_frame(name="part (%)")
                top["part (%)"] = 100 * top["part (%)"]

                # Écriture dans la feuille Excel
                sheet = date.strftime("%Y-%m-%d")
                info.to_excel(writer, sheet_name=sheet, startrow=0, index=False)
                top.to_excel(writer, sheet_name=sheet,
                             startrow=info.shape[0] + 2,
                             index_label="Actif")

        print(f"Diagnostic terminé ➜ {path}")













