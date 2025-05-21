import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from Statistics import StatisticalMetrics
from data_management import MonteCarlo_Data
from Modeling import GlobalCovolEstimation,EconometricModels

"""
Script contenant toutes les méthodes de visualisation permettant de reproduire les tableaux et graphiques du papier.
"""

class Visualization:
    def __init__(self):
        self.metrics = StatisticalMetrics()

    def table_monte_carlo_1scenario(self,X, E, X_est_list, S_est_list, S_true):

        #Calcul de la moyenne des s
        mean_s = np.mean(S_est_list[0])
        print(f"mean s = {mean_s}")

        print("The true loadings s_i used in the Monte Carlo simulations.\n")
        print(S_est_list[0])
        print("\n============================================\n")

        # Calcule les métriques
        df_metrics = self.table2(X, E, X_est_list, S_est_list, S_true,False)
        return df_metrics

    def table1(self):
        """
        Génère et affiche la Table 1 :
        Les valeurs de chargements s_i pour N=10 et N=50,
        ainsi que leur moyenne (s̄).
        """

        # Simulation pour N=10
        mc10 = MonteCarlo_Data(N=10, T=100, R=1, v=2.0, seed=42)
        s10 = mc10.s
        s10_mean = np.mean(s10)

        # Simulation pour N=50
        mc50 = MonteCarlo_Data(N=50, T=100, R=1, v=2.0, seed=999)
        s50 = mc50.s
        s50_mean = np.mean(s50)

        print("Table 1 : The true loadings s_i, i=1,...,N, used in the Monte Carlo simulations.\n")

        # Affichage pour N=10
        print(f"N=10 ( s̄ = {s10_mean:.3f} )")
        print(" ".join([f"{val:.3f}" for val in s10]))
        print()

        # Affichage pour N=50
        print(f"N=50 ( s̄ = {s50_mean:.3f} )")

        # Affichage par bloc de 10
        s50_str = [f"{val:.3f}" for val in s50]
        for i in range(0, 50, 10):
            block = s50_str[i:i+10]
            print(" ".join(block))
        print()

    def table2(self, X, E, X_est_list, S_est_list, S_true, x_fixed_flag):
        """
        Affiche la Table 2 : statistiques issues des simulations Monte Carlo.
        L'argument x_fixed_flag indique si x a été fixé ou non ("Fixed" ou "Random").
        """

        R2_x_arr = []
        R2_s_arr = []
        R = X.shape[0]
        S_true = S_true / np.linalg.norm(S_true) #Normalisation des simulations (car les estimés sont également normalisé)

        for r in range(R):

            # Calcul du R pour x pour la réplique r
            R2_x_arr.append(self.metrics.R2_x(X[r][None, :], X_est_list[r][None, :]))

            # Calcul du R pour s pour la réplique r
            R2_s_arr.append(self.metrics.R2_s(S_true[None, :], S_est_list[r][None, :]))


        avg_R2_x = np.mean(R2_x_arr)
        avg_R2_s = np.mean(R2_s_arr)
        avg_corr_e2 = self.metrics.avg_corr_e2(E, X_est_list, S_est_list)
        emp_v = self.metrics.empirical_v(X)

        data = {
            'x_type': ["Fixed" if x_fixed_flag else "Random"],
            'T': [X.shape[1]],
            'N': [S_true.shape[0]],
            'R2_s': [avg_R2_s],
            'R2_x': [avg_R2_x],
            'Avg_corr_e2': [avg_corr_e2],
            'Empirical_v': [emp_v]
        }
        df = pd.DataFrame(data)
        print("Table 2 : Empirical statistics from the Monte Carlo simulations. ")
        print(df.to_string(index=False))
        return df

    def fig_1(self, mean_residuals_standardized, vol_residual_standardized, acwi_vol=None):
        """
        Affiche :
        - en haut : la moyenne des résidus standardisés
        - en bas  : à gauche la moyenne de la volatilité des résidus,
                    et à droite la volatilité ACWI si fournie (avec deux axes différents)
        """

        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)


        #Moyenne des résidus standardisés (subplot du haut)
        mean_residuals_standardized.plot(
            ax=ax[0],
            label="Mean standardized residuals",
            color="tab:blue"
        )
        ax[0].set_title("Mean of standardized residuals")
        ax[0].set_ylabel("Mean $e_t$")
        ax[0].legend()


        # Volatilités (subplot du bas, échelle double)
        # Axe principal à gauche pour la moyenne de volatilité des résidus
        vol_residual_standardized.plot(
            ax=ax[1],
            label="Mean volatility of residuals",
            color="tab:blue"
        )
        ax[1].set_ylabel("Mean volatility of residuals")

        # Si on a la série ACWI, on la dessine sur un axe Y secondaire
        if acwi_vol is not None:
            ax2 = ax[1].twinx()
            acwi_vol.plot(
                ax=ax2,
                label="ACWI volatility (GARCH)",
                color="tab:orange",
                linestyle="--",
                alpha=0.8
            )
            ax2.set_ylabel("ACWI volatility")

            # Combine les légendes
            lines1, labels1 = ax[1].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax[1].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        ax[1].set_title("Volatility comparison")
        ax[1].set_xlabel("Date")

        plt.tight_layout()
        plt.show()

    def fig_2(self, residual_standardized, x_est):
        """
        Reproduit la Figure 2 du papier :
        - en gris (points) : les estimations journalières de la racine du facteur global COVOL (sqrt(x_t)) => the magnitude
        - sur un second axe (axe de droite), en noir (courbe) : la moyenne mobile sur 20 jours de x_t
        """


        if not isinstance(x_est, pd.Series):
            x_est = pd.Series(x_est, index=residual_standardized.index)

        # Moyenne mobile 20 jours
        x_rolling = x_est.rolling(window=20).mean()

        # Création d'une figure et d'un axe principal
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Axe principal (à gauche) : Daily x_t, en points gris
        ax1.scatter(x_est.index, np.sqrt(x_est), color='gray', s=10, alpha=0.7, label='Daily $x_t$')
        ax1.set_ylabel('Daily $x_t$ (left axis)', color='gray')
        ax1.tick_params(axis='y', labelcolor='gray')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Création d'un second axe (axe de droite)
        ax2 = ax1.twinx()

        # Sur le second axe : la courbe de la moyenne mobile
        ax2.plot(x_rolling.index, x_rolling, color='black', linewidth=2, label='20-day rolling average')
        ax2.set_ylabel('20-day rolling average (right axis)', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Ajustements du titre, de la légende et du layout
        ax1.set_title("Figure 2: Estimated Global COVOL Factor ($x_t$)")
        ax1.set_xlabel("Date")

        # Pour la légende combinée :
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

        plt.tight_layout()
        plt.show()


    def fig_3(self, s_est: pd.DataFrame):
        """
        Reproduit la Figure 3 du papier : deux panels avec les pays sélectionnés.
        """

        # Date de fins de mois
        s_est.index = pd.to_datetime(s_est.index)
        s_monthly = s_est.resample("M").last()

        # Recalibrage : multiply by √N à chaque date
        s_rescaled = s_monthly.apply(lambda row: row * np.sqrt(row.count()), axis=1)

        # Panel A : Europe centrale et occidentale
        central_and_eastern_europe = {
            "France": "dimgray",
            "Spain": "royalblue",
            "Italy": "crimson",
            "Germany": "black",
            "Netherlands": "purple",
            "Belgium": "deeppink",
            "Sweden": "gold",
            "UK": "forestgreen",
            "Switzerland": "slategrey",
        }

        # Panel B : Autres régions (Amériques, Asie, Afrique)
        other_regions_and_ACWI = {
            "US": "navy",
            "Singapore": "goldenrod",
            "China": "darkred",
            "Japan": "firebrick",
            "Mexico": "hotpink",
            "Brazil": "seagreen",
            "South Africa": "mediumorchid",
            "ACWI": "darkgrey",
            "Canada": "darkslategray",
            "New Zealand": "dodgerblue"
        }

        # SubPlot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

        # Panel A : Europe centrale
        ax1.set_title("Figure 3a: Global COVOL Loadings – Central & Western Europe")
        for country, color in central_and_eastern_europe.items():
            if country in s_rescaled.columns:
                ax1.plot(s_rescaled.index, s_rescaled[country], label=country, color=color)
        ax1.set_ylabel("Rescaled $s_i$")
        ax1.grid(True)
        ax1.legend()

        # Panel B : Monde
        ax2.set_title("Figure 3b: Global COVOL Loadings – Other Regions & ACWI")
        for country, color in other_regions_and_ACWI.items():
            if country in s_rescaled.columns:
                ax2.plot(s_rescaled.index, s_rescaled[country], label=country, color=color)
        ax2.set_ylabel("Rescaled $s_i$")
        ax2.set_xlabel("Date")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def fig_4(self, x_m: pd.Series, epu_m: pd.Series, gpr_m: pd.Series):
        """
        Reproduit la Figure 4 du papier en 2 subplots :
         - Subplot du haut : EPU (axe gauche) vs x (axe droit)
         - Subplot du bas : GPR (axe gauche) vs x (axe droit)
        """

        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # SUBPLOT DU HAUT : EPU & x
        ax_top.plot(epu_m.index, epu_m, color='tab:blue', label='EPU')
        ax_top.set_ylabel("EPU (left axis)", color='tab:blue')
        ax_top.tick_params(axis='y', labelcolor='tab:blue')
        ax_top.grid(True, linestyle='--', alpha=0.5)

        # Axe de droite pour x
        ax_top_2 = ax_top.twinx()
        ax_top_2.plot(x_m.index, x_m, color='black', label='Global COVOL (x)')
        ax_top_2.set_ylabel("Global COVOL (right axis)", color='black')
        ax_top_2.tick_params(axis='y', labelcolor='black')
        ax_top.set_title("Figure 4 - (Top) EPU vs Global COVOL")

        # Légende combinée
        lines_1, labels_1 = ax_top.get_legend_handles_labels()
        lines_2, labels_2 = ax_top_2.get_legend_handles_labels()
        ax_top_2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

        #SUBPLOT DU BAS : GPR & x
        ax_bottom.plot(gpr_m.index, gpr_m, color='tab:red', label='GPR')
        ax_bottom.set_ylabel("GPR (left axis)", color='tab:red')
        ax_bottom.tick_params(axis='y', labelcolor='tab:red')
        ax_bottom.grid(True, linestyle='--', alpha=0.5)

        ax_bottom_2 = ax_bottom.twinx()
        ax_bottom_2.plot(x_m.index, x_m, color='black', label='Global COVOL (x)')
        ax_bottom_2.set_ylabel("Global COVOL (right axis)", color='black')
        ax_bottom_2.tick_params(axis='y', labelcolor='black')
        ax_bottom.set_title("Figure 4 - (Bottom) GPR vs Global COVOL")

        # Légende combinée bas
        lines_3, labels_3 = ax_bottom.get_legend_handles_labels()
        lines_4, labels_4 = ax_bottom_2.get_legend_handles_labels()
        ax_bottom_2.legend(lines_3 + lines_4, labels_3 + labels_4, loc='best')

        # Xlabel commun
        ax_bottom.set_xlabel("Date")

        plt.tight_layout()
        plt.show()

    def fig_5(self, mean_corr_series: pd.Series):
        """
        Figure5 – Moyenne des corrélations cross‑sectionnelles
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        mean_corr_series.plot(ax=ax, color="tab:blue")
        ax.set_title("Figure 5: Moyenne des corrélations cross‑sectionnelles")
        ax.set_ylabel("Mean correlation")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    def table_6_xt_events(self, x_t: pd.Series, etf_returns: pd.DataFrame, us_returns: pd.Series, top_k=20):
        """
        Réplication Table 6 : Classe les plus grandes valeurs de x_t.
        """

        if not isinstance(x_t.index, pd.DatetimeIndex):
            if len(x_t) == len(etf_returns):
                x_t = pd.Series(x_t.values, index=etf_returns.index)
            else:
                raise ValueError(
                    "La longueur de x_t ne correspond pas à celle de etf_returns, impossible d'assigner les dates.")

        # Assure que tout est bien aligné
        etf_returns = etf_returns.loc[x_t.index]
        us_returns = us_returns.loc[x_t.index]

        # Trie les dates par ordre décroissant de x_t
        top_dates = x_t.sort_values(ascending=False).head(top_k).index

        # Construction du tableau
        data = []
        for date in top_dates:
            xt_val = x_t.loc[date]
            etf_r = etf_returns.loc[date].mean() * 100  # moyenne cross-section en %
            us_r = us_returns.loc[date] * 100  # ACWI en %

            data.append({
                "Date": date.strftime("%Y-%m-%d"),
                "x_t": xt_val,
                "Avg ETF return (%)": etf_r,
                "US return (%)": us_r,
                "Event": ""
            })

        df_table6 = pd.DataFrame(data)
        print(" Table 6 – Top événements selon $x_t$")
        print(df_table6.to_string(index=False))
        return df_table6


    def table_regression(self, y, covol2_m, d_epu_m, d_gpr_m,
                         hac_lags=12, print_out=False, add_constant=True):
        """
        Table 8 ou 9 – Régressions OLS Newey-West.
        Affiche les coefficients (p-values entre parenthèses).
        """

        X_list = [
            {"COVOL2": covol2_m},  # (1)
            {"dEPU_T": d_epu_m},  # (2)
            {"dGPR_T": d_gpr_m},  # (3)
            {"COVOL2": covol2_m, "dEPU_T": d_epu_m, "dGPR_T": d_gpr_m},  # (4)
        ]
        col_names = ["(1)", "(2)", "(3)", "(4)"]

        # Variables à reporter
        base_vars = ["COVOL2", "dEPU_T", "dGPR_T"]
        stats_rows = ["Observations", "R²", "Adj R²",
                      "Residual S.E.", "F Statistic"]

        if add_constant:
            display_vars = base_vars + ["Const."]
        else:
            display_vars = base_vars

        table = pd.DataFrame(index=display_vars + stats_rows,
                             columns=col_names)

        for k, Xdict in enumerate(X_list):
            # Design matrix
            design = EconometricModels.build_design(
                y=y,
                X_dict=Xdict,
                add_const=add_constant
            ).dropna()

            # Régression OLS HAC (NW lag = 2 par défaut)
            res = EconometricModels.fit_ols(
                design, hac_lags=hac_lags, use_HAC=False # peut être rendu à True si on veut les erreurs standards Newey-West
            )

            # Remplissage des coefficients
            for var in display_vars:
                key = "const" if var == "Const." else var
                if key in res.params:
                    coef = res.params[key]
                    se = res.bse[key]  # s.e. Newey-West (hac_lags)
                    pval = res.pvalues[key]

                    # étoiles de significativité
                    stars = ""
                    if pval < 0.01:
                        stars = "***"
                    elif pval < 0.05:
                        stars = "**"
                    elif pval < 0.10:
                        stars = "*"

                    table.loc[var, col_names[k]] = f"{coef:6.3f}{stars}({se:6.3f})"

            # Statistiques globales
            se_resid = np.sqrt(res.ssr / res.df_resid)
            table.loc["Observations", col_names[k]] = f"{int(res.nobs)}"
            table.loc["R²", col_names[k]] = f"{res.rsquared:6.3f}"
            table.loc["Adj R²", col_names[k]] = f"{res.rsquared_adj:6.3f}"
            table.loc["Residual S.E.", col_names[k]] = f"{se_resid:6.3f}"
            table.loc["F Statistic", col_names[k]] = f"{res.fvalue:6.3f}"

        if print_out:
            print("\n=== Table de régression ===")
            print(table.to_string())

        return table

    def table_10(self, mean_corr_m, covol2_m, d_epu_T, d_gpr_T,
                 hac_lags: int = 12, print_out: bool = False):

        y = mean_corr_m
        pm_1 = mean_corr_m.shift(1)

        # mêmes quatre spécifications que le papier
        X_list = [
            {"COVOL2": covol2_m, "pm_1": pm_1},  # (1)
            {"dEPU_T": d_epu_T, "pm_1": pm_1},  # (2)
            {"dGPR_T": d_gpr_T, "pm_1": pm_1},  # (3)
            {"COVOL2": covol2_m, "dEPU_T": d_epu_T,
             "dGPR_T": d_gpr_T, "pm_1": pm_1},  # (4)
        ]
        col_names = ["(1)", "(2)", "(3)", "(4)"]
        vars_all = ["COVOL2", "dEPU_T", "dGPR_T", "pm_1", "Const."]

        table = pd.DataFrame(index=vars_all +
                                   ["Observations", "R²", "Adj R²",
                                    "Residual S.E.", "F Statistic"],
                             columns=col_names)

        for k, Xdict in enumerate(X_list):
            design = EconometricModels.build_design(
                y=y, X_dict=Xdict, add_const=True).dropna()

            res = EconometricModels.fit_ols(
                design, hac_lags=hac_lags, use_HAC=False)  #  HAC(2) peut être True

            # coefficients, s.e., étoiles
            for v in ["const", "COVOL2", "dEPU_T", "dGPR_T", "pm_1"]:
                label = "Const." if v == "const" else v
                if v in res.params:
                    coef = res.params[v]
                    se = res.bse[v]
                    pval = res.pvalues[v]

                    stars = ""
                    if pval < 0.01:
                        stars = "***"
                    elif pval < 0.05:
                        stars = "**"
                    elif pval < 0.10:
                        stars = "*"

                    table.loc[label, col_names[k]] = f"{coef:6.3f}{stars}({se:6.3f})"

            # Statistiques globales
            se_resid = np.sqrt(res.ssr / res.df_resid)
            table.loc["Observations", col_names[k]] = f"{int(res.nobs)}"
            table.loc["R²", col_names[k]] = f"{res.rsquared:6.3f}"
            table.loc["Adj R²", col_names[k]] = f"{res.rsquared_adj:6.3f}"
            table.loc["Residual S.E.", col_names[k]] = f"{se_resid:6.3f}"
            table.loc["F Statistic", col_names[k]] = f"{res.fvalue:6.3f}"

        if print_out:
            print("\n=== Table 10 – Explaining mean correlation ===")
            print(table.to_string())
        return table

    def fig_6(self, x_m: pd.Series, vix_m: pd.Series, recessions: list):
        """
        Figure 6 : Global COVOL vs VIX avec zones de récession grisées.

        """
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Axe gauche : Global COVOL
        x_m.plot(ax=ax1, label="Global COVOL", color="black")
        ax1.set_ylabel("Global COVOL", color="black")
        ax1.tick_params(axis='y', labelcolor="black")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Axe droit : VIX
        ax2 = ax1.twinx()
        vix_m.plot(ax=ax2, label="VIX", color="tab:blue", linestyle="--")
        ax2.set_ylabel("VIX", color="tab:blue")
        ax2.tick_params(axis='y', labelcolor="tab:blue")

        # Zones de récession grisées
        for start, end in recessions:
            ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), color="grey", alpha=0.3)

        ax1.set_title("Figure 6: Global COVOL and VIX (Volatility)")
        ax1.set_xlabel("Date")

        # Légende combinée
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

        plt.tight_layout()
        plt.show()

    def table_11(self, covol2_m: pd.Series, hac_lags: int = 12, print_out=True):
        """
        Table 11 – Régression de Global COVOL sur dummies d'expansion/contraction (NBER).
        """
        # Périodes définies dans le papier
        periods = {
            "EXP1": ("2000-06-01", "2001-02-28"),
            "CON1": ("2001-03-01", "2001-11-30"),
            "EXP2": ("2001-12-01", "2007-11-30"),
            "CON2": ("2007-12-01", "2009-06-30"),
            "EXP3": ("2009-07-01", "2020-01-31"),
            "CON3": ("2020-02-01", "2020-04-30"),
            "EXP4": ("2020-05-01", "2021-01-31"),
        }

        # Création des variables indicatrices
        df_dummies = pd.DataFrame(index=covol2_m.index)
        for label, (start, end) in periods.items():
            df_dummies[label] = ((df_dummies.index >= pd.to_datetime(start)) &
                                 (df_dummies.index <= pd.to_datetime(end))).astype(int)

        design_df = EconometricModels.build_design(
            y=covol2_m,
            X_dict={col: df_dummies[col] for col in df_dummies.columns},
            add_const=False
        )

        # Régression avec HAC robust std errors
        res = EconometricModels.fit_ols(design_df, hac_lags=hac_lags, use_HAC=False) #HAC peut être True

        # Construction de la table à afficher
        var_names = df_dummies.columns.tolist()
        table = pd.DataFrame(index=var_names + ["Observations", "R²", "Adj R²", "Residual S.E.", "F Statistic"],
                             columns=["Global COVOL²"])

        for v in var_names:
            if v in res.params:
                coef = res.params[v]
                se = res.bse[v]  # écart-type NW(12)
                pval = res.pvalues[v]  # pour les étoiles

                stars = ""
                if pval < 0.01:
                    stars = "***"
                elif pval < 0.05:
                    stars = "**"
                elif pval < 0.10:
                    stars = "*"

                table.loc[v, "Global COVOL²"] = f"{coef:6.3f}{stars}({se:6.3f})"

        # Statistiques
        se_resid = np.sqrt(res.ssr / res.df_resid)
        table.loc["Observations", "Global COVOL²"] = f"{int(res.nobs)}"
        table.loc["R²", "Global COVOL²"] = f"{res.rsquared:.3f}"
        table.loc["Adj R²", "Global COVOL²"] = f"{res.rsquared_adj:.3f}"
        table.loc["Residual S.E.", "Global COVOL²"] = f"{se_resid:.3f}"
        table.loc["F Statistic", "Global COVOL²"] = f"{res.fvalue:.3f}"

        if print_out:
            print("\n=== Table 11 – Monthly COVOL² and NBER Business Cycle ===")
            print(table.to_string())

        return table


    def table_12_sentiment_regressions(self,
                                       covol2_m: pd.Series,
                                       vix_m: pd.Series,
                                       epu_m: pd.Series,
                                       cci_m: pd.Series,
                                       hac_lags: int = 12,
                                       print_out: bool = True):

        def ms(s):  # index au 1 du mois
            s = s.copy()
            s.index = s.index.to_period("M").to_timestamp()
            return s.sort_index()

        covol2_m = ms(covol2_m)
        dvix = ms(vix_m).diff()
        dgepu = ms(epu_m).diff()
        dcci = ms(cci_m).diff()

        dep_series = {"ΔVIX": dvix, "ΔGEPU": dgepu, "ΔCCI": dcci}
        table = pd.DataFrame(index=["Global COVOL²", "Observations",
                                    "R²", "Adj R²", "Residual S.E.", "F Statistic"],
                             columns=dep_series.keys())

        for name, y in dep_series.items():

            y = y.squeeze()

            # design
            Xdict = {"COVOL2": covol2_m}

            # ajoute AR(3) seulement pour GEPU et CCI
            if name in ("ΔGEPU", "ΔCCI"):
                for l in range(1, 4):
                    Xdict[f"AR{l}"] = y.shift(l)

            # alignement index
            y_aligned, x_aligned = y.align(covol2_m, join="inner")

            design = EconometricModels.build_design(
                y=y_aligned,
                X_dict=Xdict,
                add_const=True
            ).dropna()



            res = EconometricModels.fit_ols(
                design, hac_lags=hac_lags, use_HAC=False) # HAC peut être True

            # coefficient COVOL
            beta = res.params["COVOL2"]
            se = res.bse["COVOL2"]
            pval = res.pvalues["COVOL2"]

            stars = "***" if pval < .01 else "**" if pval < .05 else "*" if pval < .10 else ""
            table.loc["Global COVOL²", name] = f"{beta:6.3f}{stars}({se:6.3f})"

            # statistiques globales
            se_resid = np.sqrt(res.ssr / res.df_resid)
            table.loc["Observations", name] = f"{int(res.nobs)}"
            table.loc["R²", name] = f"{res.rsquared:6.3f}"
            table.loc["Adj R²", name] = f"{res.rsquared_adj:6.3f}"
            table.loc["Residual S.E.", name] = f"{se_resid:6.3f}"
            table.loc["F Statistic", name] = f"{res.fvalue:6.3f}"

        if print_out:
            print("\n=== Table 12 – Sentiment regressions on Global COVOL² ===")
            print(table.to_string())

        return table

    def fig_7(self, x_m: pd.Series, cci_m: pd.Series, recessions: list):
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Axe gauche – CCI
        ax1.plot(cci_m.index, cci_m, label="CCI")
        ax1.set_ylabel("CCI", color="tab:blue")
        ax1.tick_params(axis='y', labelcolor="tab:blue")
        ax1.grid(True, linestyle="--", alpha=0.4)

        # Ajouter les zones de récession
        for start, end in recessions:
            ax1.axvspan(pd.to_datetime(start),
                        pd.to_datetime(end),
                        color="grey", alpha=0.3)

        # Axe droit – Global COVOL
        ax2 = ax1.twinx()
        ax2.plot(x_m.index, x_m, color="black", linewidth=2, label="Global COVOL")
        ax2.set_ylabel("Global COVOL", color="black")
        ax2.tick_params(axis='y', labelcolor="black")

        # Légende
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

        plt.tight_layout()
        plt.show()


    def table_13_macro_reg(self,
                           emp_g: pd.Series,        # log emploi privé
                           ind_g: pd.Series,        # log IP
                           covol2_m: pd.Series,     # COVOL mensuel
                           hac_lags: int = 12,
                           print_out: bool = True):

        # harmonise calendrier (MS = 1er jour)
        covol2_m = covol2_m.copy()
        covol2_m.index = covol2_m.index.to_period("M").to_timestamp()

        dep = {"Δemp_m": emp_g, "Δind_m": ind_g}

        rows = ["Global COVOL²_m",
                "Global COVOL²_m-1",
                "Y_m-1",
                "Const.",
                "Observations", "R²", "Adj R²",
                "Residual S.E.", "F Statistic"]

        table = pd.DataFrame(index=rows, columns=dep.keys())

        for name, y in dep.items():
            y = y.copy().dropna()
            y.index = y.index.to_period("M").to_timestamp()

            # Design : COVOL²_t, COVOL²_{t-1}, Y_{t-1}
            Xdict = {
                "COVOL2":     covol2_m,
                "COVOL2_L1":  covol2_m.shift(1),
                "Y_L1":       y.shift(1)
            }

            design = EconometricModels.build_design(
                y=y, X_dict=Xdict, add_const=True
            ).dropna()

            res = EconometricModels.fit_ols(
                design, hac_lags=hac_lags, use_HAC=False # HAC peut être True
            )

            # utilitaire local
            def put(row_name, key):
                coef, se, p = res.params[key], res.bse[key], res.pvalues[key]
                stars = "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
                table.loc[row_name, name] = f"{coef:7.3f}{stars}({se:7.3f})"

            put("Global COVOL²_m",   "COVOL2")
            put("Global COVOL²_m-1", "COVOL2_L1")
            put("Y_m-1",             "Y_L1")
            put("Const.",            "const")

            # statistiques globales
            se_resid = np.sqrt(res.ssr / res.df_resid)
            table.loc["Observations",  name] = f"{int(res.nobs)}"
            table.loc["R²",            name] = f"{res.rsquared:6.3f}"
            table.loc["Adj R²",        name] = f"{res.rsquared_adj:6.3f}"
            table.loc["Residual S.E.", name] = f"{se_resid:6.3f}"
            table.loc["F Statistic",   name] = f"{res.fvalue:6.3f}"

        if print_out:
            print("\n=== Table 13 – Monthly macro impact of Global COVOL² ===")
            print(table.to_string())

        return table


    def table_14_macro_reg(self,
                           inv_g: pd.Series,        #  log investissement
                           cons_g: pd.Series,       #  log consommation
                           covol2_q: pd.Series,     # COVOL trimestriel
                           hac_lags: int = 4,
                           print_out: bool = True):

        covol2_q = covol2_q.copy()
        covol2_q.index = covol2_q.index.to_period("Q").to_timestamp()

        dep = {"Δinv_q": inv_g, "Δcons_q": cons_g}

        rows = ["Global COVOL²_q",
                "Global COVOL²_q-1",
                "Y_q-1",
                "Const.",
                "Observations", "R²", "Adj R²",
                "Residual S.E.", "F Statistic"]

        table = pd.DataFrame(index=rows, columns=dep.keys())

        for name, y in dep.items():
            y = y.copy().dropna()
            y.index = y.index.to_period("Q").to_timestamp()

            # Design : COVOL²_t, COVOL²_{t-1}, Y_{t-1}
            Xdict = {
                "COVOL2":     covol2_q,
                "COVOL2_L1":  covol2_q.shift(1),
                "Y_L1":       y.shift(1)
            }

            design = EconometricModels.build_design(
                y=y, X_dict=Xdict, add_const=True
            ).dropna()

            res = EconometricModels.fit_ols(
                design, hac_lags=hac_lags, use_HAC=False # HAC peut être True
            )

            def put(row_name, key):
                coef, se, p = res.params[key], res.bse[key], res.pvalues[key]
                stars = "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
                table.loc[row_name, name] = f"{coef:7.3f}{stars}({se:7.3f})"

            put("Global COVOL²_q",   "COVOL2")
            put("Global COVOL²_q-1", "COVOL2_L1")
            put("Y_q-1",             "Y_L1")
            put("Const.",            "const")

            se_resid = np.sqrt(res.ssr / res.df_resid)
            table.loc["Observations",  name] = f"{int(res.nobs)}"
            table.loc["R²",            name] = f"{res.rsquared:6.3f}"
            table.loc["Adj R²",        name] = f"{res.rsquared_adj:6.3f}"
            table.loc["Residual S.E.", name] = f"{se_resid:6.3f}"
            table.loc["F Statistic",   name] = f"{res.fvalue:6.3f}"

        if print_out:
            print("\n=== Table 14 – Quarterly macro impact of Global COVOL² ===")
            print(table.to_string())

        return table
