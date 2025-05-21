import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from data_management import MonteCarlo_Data,DataUtils,Excel_data,ResultManager,MacroDataLoader
from Modeling import GlobalCovolEstimation,EconometricModels
from Visualization_Management import Visualization
from Statistics import StatisticalMetrics,DescriptiveStats


"""Script principal contenant les fonctions de réplication de chaque partie de l’article.
Dans le bloc if __name__ == "__main__":, les appels sont commentés, il suffit de les décommenter pour choisir la réplication à effectuer."""


def main_Monte_Carlo_1scenario():
    """Fonction de lancement de la méthode de Monte Carlo pour un seul scénario (N=50, T=1000, R=150) """

    # Simulation des données de Monte Carlo
    mc = MonteCarlo_Data(N=10, T=1000, R=5, v=2.0, seed=42, x_fixed=False)
    X, E = mc.run_simulations()

    # Listes pour stocker les estimations par réplique
    X_est_list = []
    S_est_list = []
    S_true = mc.s.copy()  # Les vrais factors loadings, dimensions (N,)

    # Statistiques de test
    metrics = StatisticalMetrics()
    xi_list = []

    # Pour chaque réplique (simulation Monte Carlo), estimation du facteur Global COVOL
    R = mc.R
    for r in range(R):
        print(f"Simulation {r + 1}/{R}")
        e = E[r]  # Données de résidus pour la réplique r, dimensions (T x N)
        model = GlobalCovolEstimation(e,grid_search=False)

        x_est, s_est = model.estimate()
        X_est_list.append(x_est)
        S_est_list.append(s_est)

        #Calcul de la statistique de test
        xi, reject_H0, _ = metrics.dstats_xi_test(e, x_est, s_est, alpha=0.05)
        xi_list.append(reject_H0)

    # Création de la table des résultats à l'aide de la classe de visualisation
    viz = Visualization()
    viz.table_monte_carlo_1scenario(X, E, X_est_list, S_est_list, S_true)

    rej_rate = sum(xi_list) / len(xi_list)
    print(f"Taux de rejet empirique du test ξ (alpha = 5%) = {rej_rate:.3f}")



def main_Monte_Carlo():
    """
    Fonction de lancement de toute la partie Monte Carlo, pour obtenir les tables 1 et 2 du papier (Monte Carlo répété sur plusieurs scénarios)
    """

    # Scénarios à tester
    scenarios = [
        {"x_fixed": False, "T": 1000, "N": 10},
        {"x_fixed": False, "T": 1000, "N": 50},
        {"x_fixed": False, "T": 5000, "N": 10},
        {"x_fixed": False, "T": 5000, "N": 50},
        {"x_fixed": True, "T": 1000, "N": 10},
        {"x_fixed": True, "T": 1000, "N": 50},
        {"x_fixed": True, "T": 5000, "N": 10},
        {"x_fixed": True, "T": 5000, "N": 50},
    ]

    results = []  # Pour stocker les DataFrame de chaque scénario
    viz = Visualization()  # Instance pour la visualisation et le calcul des métriques

    # Itère sur chaque scénario
    for sc in scenarios:
        x_fixed = sc["x_fixed"]
        T_val = sc["T"]
        N_val = sc["N"]
        print(f"Scénario: x_fixed = {x_fixed}, T = {T_val}, N = {N_val}")

        # Simulation des données de Monte Carlo pour le scénario
        mc = MonteCarlo_Data(N=N_val, T=T_val, R=150, v=2.0, seed=42, x_fixed=x_fixed)
        X, E = mc.run_simulations()  # X: (R, T), E: (R, T, N)
        S_true = mc.s.copy()  # Chargements vrais, dimensions (N,)

        # Pour chaque réplique, estime le facteur Global COVOL
        X_est_list = []
        S_est_list = []
        R = mc.R

        #Itération sur chaque réplique (simulation Monte Carlo)
        for r in range(R):
            e = E[r]  # (T, N)
            model = GlobalCovolEstimation(e)
            x_est, s_est = model.estimate()
            X_est_list.append(x_est)
            S_est_list.append(s_est)

        #Calcule les métriques pour ce scénario
        df_sc = viz.table2(X, E, X_est_list, S_est_list, S_true, x_fixed)

        # Ajoute les paramètres du scénario dans le DataFrame
        df_sc["x_fixed"] = "Fixed" if x_fixed else "Random"
        df_sc["T"] = T_val
        df_sc["N"] = N_val
        results.append(df_sc)

    # Concat les résultats de tous les scénarios dans un DataFrame final
    final_df = pd.concat(results, ignore_index=True)

    # Réorganise les colonnes dans l'ordre du tableau du papier
    final_df = final_df[["x_fixed", "T", "N", "R2_x", "R2_s", "Avg_corr_e2", "Empirical_v"]]


    # Affichage de la Table 1
    viz.table1()

    #Affichage de la Table 2
    print("\n===== Tableau récapitulatif des scénarios =====")
    print(final_df.to_string(index=False))


def replicate_test_table_Monte_Carlo(N_list, T_list, R=1000, alpha=0.05, v=0, fixed_s=False):
    """
    Réplique les Tables 3, 4, ou 5 selon les paramètres :
    - v = 0 pour Table 3 (H0)
    - v > 0 et fixed_s=True pour Table 4 (H1, effets égaux)
    - v > 0 et fixed_s=False pour Table 5 (H1, effets hétérogènes)
    """

    results = []
    metrics = StatisticalMetrics()

    for T in T_list:
        for N in N_list:
            print(f"\nSimulation: N={N}, T={T}, v={v}, fixed_s={fixed_s}")

            reject_count = 0
            for r in range(R):
                mc = MonteCarlo_Data(N=N, T=T, R=1, v=v, x_fixed=False)
                if fixed_s:
                    mc.s = np.ones(N)  # Si s fixe à 1 pour Table 4

                X, E = mc.run_simulations()
                e = E[0]

                # Estimate x and s (nécessaire pour ξ)
                model = GlobalCovolEstimation(e, grid_search=True)
                x_est, s_est = model.estimate()

                xi, reject_H0, _ = metrics.dstats_xi_test(e, x_est, s_est, alpha=alpha)
                reject_count += int(reject_H0)

            rejection_rate = reject_count / R
            results.append({'T': T, 'N': N, 'alpha': alpha, 'rej_rate': rejection_rate})

    # Affichage
    import pandas as pd
    df = pd.DataFrame(results)
    print(df.pivot(index=['T'], columns='N', values='rej_rate'))
    return df


def etf_country_stats():
    """
    Réplique la Table C.1 – Statistiques descriptives des rendements
    pour les ETF par pays.
    """

    print("\n=== Réplication Table C.1 – Statistiques descriptives ===")

    # Chargement des données
    etf_loader = Excel_data(file_name="Data ETF Country.xlsx", sheet_name="Data")
    bench_loader = Excel_data(file_name="Data Bench.xlsx", sheet_name="Data")

    df_etf = etf_loader.data
    df_bench = bench_loader.data

    # Nettoyage des données pour garder les mêmes dates que le benchmark
    df_etf_clean, df_bench_clean = DataUtils.align_with_benchmark(df_etf, df_bench)

    # Restriction des données à la période souhaitée
    start_date = "2000-06-02"
    end_date = "2021-03-01"
    df_etf_period = df_etf_clean.loc[start_date:end_date]

    # Rendements log pour chaque colonne
    returns_etf = DataUtils.compute_returns(df_etf_period,method="log")

    # Statistiques descriptives
    stats = DescriptiveStats(returns_etf)
    df_table_c1 = stats.compute_all()

    #Sauvegarde de la table statistique
    result_mgr = ResultManager()
    result_mgr.save_dataframe(df_table_c1, "table_c1_stats")


    # Affichage
    print("\n Réplication terminée – Table C.1")
    return df_table_c1



def model_residuals_and_global_covol(data_save = False):
    """
    Etapes d'application sur les etf country : Réplication Fig 1 et  2 et Table 6
    """


    print("\n===  Réplication Section 7 – Modélisation des résidus ===")

    #Instance de la classe de gestion des données sauvegardées
    result_mgr = ResultManager()

    if data_save:
        print("Chargement des données sauvegardées...")
        residuals = result_mgr.load_dataframe("residuals")
        volatilities = result_mgr.load_dataframe("volatilities")
        residual_standardized = result_mgr.load_dataframe("residual_standardized")
        x_est = result_mgr.load_series("x_est")
        s_est = result_mgr.load_dataframe("s_est")
        acwi_vol = result_mgr.load_dataframe("acwi_volatilities")

        # Chargement des données
        etf_loader = Excel_data(file_name="Data ETF Country.xlsx", sheet_name="Data")
        bench_loader = Excel_data(file_name="Data Bench.xlsx", sheet_name="Data")
        df_etf, df_bench = etf_loader.get_data(), bench_loader.get_data()

        # Synchronisation des dates (benchmark comme référence)
        df_etf_aligned, df_bench_aligned = DataUtils.align_with_benchmark(df_etf, df_bench)

        # Restriction période juin 2006 - mars 2021
        start_date = "2000-06-02"
        end_date = "2021-03-01"
        df_etf_aligned = df_etf_aligned.loc[start_date:end_date]
        df_bench_aligned = df_bench_aligned.loc[start_date:end_date]

        returns = DataUtils.compute_returns(df_etf_aligned, method="log")
        acwi_returns = DataUtils.compute_returns(df_bench_aligned, method="log").iloc[:, 0]


    else:

        # Chargement des données
        etf_loader = Excel_data(file_name="Data ETF Country.xlsx", sheet_name="Data")
        bench_loader = Excel_data(file_name="Data Bench.xlsx", sheet_name="Data")
        df_etf, df_bench = etf_loader.get_data(), bench_loader.get_data()

        # Synchronisation des dates (benchmark comme référence)
        df_etf_aligned, df_bench_aligned = DataUtils.align_with_benchmark(df_etf, df_bench)

        # Restriction période juin 2006 - mars 2021
        start_date = "2000-06-01"
        end_date = "2021-03-01"
        df_etf_aligned = df_etf_aligned.loc[start_date:end_date]
        df_bench_aligned=df_bench_aligned.loc[start_date:end_date]

        # Rendements log
        # ETFs
        returns = DataUtils.compute_returns(df_etf_aligned, method="log")

        # ACWI
        acwi_returns = DataUtils.compute_returns(df_bench_aligned, method="log").iloc[:, 0]

        # PC1
        pc1 = EconometricModels.compute_pc1_cross_section(returns)

        # Estimation des résidus standardisés pour chaque serie d'etf, plus ACWI et PC1
        residuals, volatilities, residual_standardized = EconometricModels.batch_estimate_residuals(
            returns, acwi_returns, pc1
        )

        # Résidus standardisés & vol ACWI
        acwi_e = acwi_returns.dropna()
        X_acwi = {
            "lag": acwi_e.shift(1)
        }
        _, acwi_vol, e_acwi = EconometricModels.estimate_standardized_residuals(acwi_e, X_acwi)
        residual_standardized["ACWI"] = e_acwi.reindex(residual_standardized.index)

        # Résidus standardisés PC1
        pc1_e = pc1.dropna()
        X_pc1 = {
            "lag": pc1_e.shift(1)
        }
        _, _, e_pc1 = EconometricModels.estimate_standardized_residuals(pc1_e, X_pc1)
        residual_standardized["PC1"] = e_pc1.reindex(residual_standardized.index)

        print("\n=== Estimation Global COVOL ===")
        x_est, s_est = GlobalCovolEstimation.estimate_global_covol_full_sample(
            residual_standardized,
            grid_search=False  # ou True
        )

        result_mgr.save_dataframe(residuals, "residuals")
        result_mgr.save_dataframe(volatilities, "volatilities")
        result_mgr.save_dataframe(residual_standardized, "residual_standardized")
        result_mgr.save_series(pd.Series(x_est, index=residual_standardized.index), "x_est")
        result_mgr.save_dataframe(s_est, "s_est")
        result_mgr.save_dataframe(acwi_vol, "acwi_volatilities")

    # DIAGNOSTIC DES OUTLIERS (x_t > 20)
    EconometricModels.diagnose_large_xt(
        x_series=x_est,  # ou x_daily si renommé
        residual_std=residual_standardized,
        volatilities=volatilities,  # ou None
        threshold=20,
        top_n=10,
        rm=result_mgr,
        file_name="diag_xt_large"  # => Result/diag_xt_large.xlsx
    )

    # Matrice de covariance des résidus au carré centrés
    z = residual_standardized ** 2 - 1
    Psi = z.cov()

    mask = np.eye(Psi.shape[0], dtype=bool)
    vmax = np.percentile(np.abs(Psi.values[~mask]), 95)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        Psi,
        mask=mask,
        cmap="viridis",
        vmax=vmax,
        cbar_kws={'label': 'Covariance'},
        xticklabels=True,
        yticklabels=True
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Heatmap de Ψ (hors-diagonale masquée, échelle coupée au 95ᵉ percentile)")
    plt.tight_layout()
    plt.show()

    # Affichage des graphiques
    # Fig 1 : Moyennes et vol transversales
    mean_residuals_standardized = residual_standardized.mean(axis=1)
    vol_residual_standardized = volatilities.pow(2).mean(axis=1).apply(np.sqrt)
    viz = Visualization()

    # Fig 1
    acwi_vol = acwi_vol.reindex(mean_residuals_standardized.index)
    viz.fig_1(mean_residuals_standardized, vol_residual_standardized * 100, acwi_vol * 100)

    #Fig. 2 : x_t
    viz.fig_2(residual_standardized,x_est)

    #Table 6 :
    df_table6 = viz.table_6_xt_events(
        x_est,
        returns,
        returns["US"],
        top_k=20
    )

    # sauvegarde des valeurs table 6
    result_mgr.save_dataframe(df_table6, "table6_xt_events")

    return x_est, residual_standardized



def estimate_s_time_series_and_plot_fig3(data_save=False, grid_search=False):

    """
    Fonction de calcul des factors loadings pays, en times series (necessite d'avoir des résidus standardisés sauvegardés)
    """

    print("\n=== Estimation mensuelle des s_i et tracé Figure 3 ===")

    result_mgr = ResultManager()
    viz = Visualization()

    if data_save:
        # On ne recalcule pas, on charge la matrix S_timeseries sauvegardée
        s_est_ts = result_mgr.load_dataframe("S_timeseries_monthly")
    else:

        if not result_mgr.file_exists("residual_standardized"):
            print("Pas de residual_standardized sauvegardé ! Lance d'abord model_residuals_and_global_covol.")
            return

        residual_standardized = result_mgr.load_dataframe("residual_standardized")

        # Estime la time series mensuelle des loadings
        s_est_ts = GlobalCovolEstimation.estimate_loadings_monthly(residual_standardized, grid_search=grid_search)

        result_mgr.save_dataframe(s_est_ts, "S_timeseries_monthly")


    viz.fig_3(s_est_ts)

    print("\n=== Fin estimation mensuelle s_i et tracé Fig 3 ===")

    return s_est_ts




def replicate_partie_8():
    """
    Figure4  (GlobalCOVOL vsEPU & GPR)
    Table7   (loadings s_i  +  s_i * sigma2_i)
    Table8   Regression ACWI_vol
    Table9   Regression ACWI_returns
    """

    #Initialisation des objects
    result_mgr = ResultManager()
    viz = Visualization()

    # Charge x_est
    if not result_mgr.file_exists("x_est"):
        print("Pas de x_est sauvegardé (run model_residuals_and_global_covol d'abord).")
        return
    x_daily = result_mgr.load_series("x_est")

    #Charge EPU et GPR
    epu_m = MacroDataLoader.load_us_epu(file_name="US_EPU_Data.xlsx")
    gpr_m = MacroDataLoader.load_gpr(file_name="Data_GPR.xlsx")

    #Formate les données
    x_ld = (x_daily).resample("MS").first()
    #x_m = (x_daily ** 2).resample("MS").mean()
    x_m = (x_daily.resample("MS").mean())
    x_m.index = x_m.index.to_period("M").to_timestamp(how='start')

    def to_month_start(s):
        """
        Convertit l’index d’un Series/DataFrame au premier jour du mois.
        """
        s2 = s.copy()
        s2.index = s2.index.to_period("M").to_timestamp(how="start")
        return s2.sort_index()

    # harmonisation
    x_ld = to_month_start(x_ld)
    x_m = to_month_start(x_m)
    epu_m = to_month_start(epu_m)
    gpr_m = to_month_start(gpr_m)

    # concaténation
    df_all = pd.concat([x_m, epu_m, gpr_m], axis=1, join="inner")
    df_all.columns = ["x", "EPU", "GPR"]

    #Figure 4
    viz.fig_4(df_all["x"], df_all["EPU"], df_all["GPR"])


    #Table 7 -------
    #Charge les si
    s_series = result_mgr.load_dataframe("s_est").iloc[:,0]

    # Charge les données des ETF Country
    etf_loader = Excel_data(file_name="Data ETF Country.xlsx", sheet_name="Data")
    bench_loader = Excel_data(file_name="Data Bench.xlsx", sheet_name="Data")

    df_etf_raw = etf_loader.get_data()
    df_bench_raw = bench_loader.get_data()

    # Alignement
    df_etf_al, df_bench_al = DataUtils.align_with_benchmark(df_etf_raw, df_bench_raw)
    start, end = "2000-06-02", "2021-03-01"
    df_etf_al = df_etf_al.loc[start:end]
    df_bench_al = df_bench_al.loc[start:end]

    #Calcul des log rendements
    returns_etf = DataUtils.compute_returns(df_etf_al, method="log")
    acwi_ret = DataUtils.compute_returns(df_bench_al, method="log").iloc[:, 0]
    acwi_ret_monthly = acwi_ret.resample("MS").sum()


    returns_etf = returns_etf.apply(pd.to_numeric, errors="coerce")  # force numérique
    returns_etf = returns_etf.loc[:, returns_etf.notna().any(axis=0)]  # vire les 100 % NaN

    # PC1
    pc1_series = EconometricModels.compute_pc1_cross_section(returns_etf)

    #Concat de tout les rendements
    all_ret_pct = pd.concat(
        [returns_etf,  # ETF pays
         acwi_ret.rename("ACWI"),
         pc1_series.rename("PC1")],
        axis=1
    ) * 100  # en pourcentage

    #Calcul des volatilités pour chaque actifs
    sigma2 = all_ret_pct.var(skipna=True) #* 100 # en pourcentage


    #table 7
    table7 = (
        pd.DataFrame({"s_i": s_series})
        .join(sigma2.rename("sigma2_i"))
        .assign(s_i_sigma2=lambda d: d["s_i"] * d["sigma2_i"])
        .sort_values("s_i", ascending=False)
    )

    print("\n=== Table 7 – GlobalCOVOL factor loadings ===")
    print(table7[["s_i", "s_i_sigma2"]].to_string(float_format=lambda x: f"{x:8.3f}"))

    # sauvegarde en Excel/CSV pour LaTeX
    result_mgr.save_dataframe(table7, "table7_loadings")


    # table 8 ------------------------------------------------------------------

    # résidus standardisés ACWI
    if not result_mgr.file_exists("residual_standardized"):
        print("residual_standardized manquant — relance model_residuals_and_global_covol().")
        return
    resid_std = result_mgr.load_dataframe("residual_standardized")

    psi_acwi_m=((resid_std["ACWI"]**2).resample("MS").mean() - 1)#test


    # Variables explicatives
    covol2_m = (((x_daily).resample("MS")).mean() - 1)#**2#test sans reshape
    covol2_m = covol2_m.loc[psi_acwi_m.index]
    d_epu_m = epu_m.diff().loc[psi_acwi_m.index] # variation EPU
    d_gpr_m = gpr_m.diff().loc[psi_acwi_m.index] # variation GPR

    common_idx = psi_acwi_m.index \
        .intersection(covol2_m.index) \
        .intersection(d_epu_m.index) \
        .intersection(d_gpr_m.index)

    common_idx = common_idx.sort_values()[1:-1]  # enlève la toute première et la toute dernière date pour etre à 248 données comme le papier

    psi_acwi_m = psi_acwi_m.loc[common_idx]
    covol2_m = covol2_m.loc[common_idx]
    d_epu_m = d_epu_m.loc[common_idx]
    d_gpr_m = d_gpr_m.loc[common_idx]

    acwi_ret_monthly = (
        acwi_ret.resample("MS").sum()
        .loc[common_idx]  # aligne exactement
    )

    #Régression de vol réalisée sur les mesures de risques global
    table_8 = viz.table_regression(
        y=psi_acwi_m,
        covol2_m=covol2_m,
        d_epu_m=d_epu_m,
        d_gpr_m=d_gpr_m,
        hac_lags=2,
        add_constant = False
    )

    print("\n=== Table 8 – Explaining ACWI_volatility ===")
    print(table_8.to_string())

    #table 9 ----------------------------------------------------------
    #Régression des rendements mensuel sur facteurs de risque
    table_9 = viz.table_regression(
        y=acwi_ret_monthly,
        covol2_m=covol2_m,
        d_epu_m=d_epu_m,
        d_gpr_m=d_gpr_m,
        hac_lags=2,
        add_constant = True
    )

    print("\n=== Table 9 – Explaining ACWI_returns ===")
    print(table_9.to_string())

    #Fig 6
    mean_corr = EconometricModels.mean_cross_corr_monthly(returns_etf)
    viz.fig_5(mean_corr)

    #Table 10
    mean_corr_ms = mean_corr.copy()
    mean_corr_ms.index = mean_corr_ms.index.to_period("M").to_timestamp()
    table_10 = viz.table_10(mean_corr_ms, covol2_m, d_epu_m, d_gpr_m)
    print("\n=== Table 10 – Explaining The averaged correlation of returns ===")
    print(table_10.to_string())

    # === Figure 6 ===
    # Définition des périodes de récession
    recessions = [
        ("2001-03-01", "2001-11-30"),
        ("2007-12-01", "2009-06-30"),
        ("2020-02-01", "2020-04-30")
    ]

    # Charge le VIX
    vix_series = MacroDataLoader.load_vix(file_name="VIX.xlsx")
    vix_m = vix_series.resample("M").mean()
    vix_m.index = vix_m.index.to_period("M").to_timestamp(how="start")

    Monthly_globalcovol = (x_daily).resample("MS").mean()**2
    Monthly_globalcovol.index = Monthly_globalcovol.index.to_period("M").to_timestamp(how='start')

    viz.fig_6(x_m, vix_m.loc[Monthly_globalcovol.index], recessions)

    # Figure 7
    cci_df = MacroDataLoader.load_cci(file_name="CCI.xlsx")
    cci_m = cci_df.iloc[:, 0].resample("M").mean()
    cci_m.index = cci_m.index.to_period("M").to_timestamp()

    viz.fig_7(x_m, cci_m.loc[Monthly_globalcovol.index], recessions)

    #Table 11
    viz.table_11(Monthly_globalcovol)

    # Table 12 – Sentiment regressions
    def to_month_start(s):
        s2 = s.copy()
        s2.index = (s2.index
                    .to_period("M")
                    .to_timestamp())  # 01-MM-YYYY   (how="start" implicite)
        return s2.sort_index()

    cci_raw = MacroDataLoader.load_cci("CCI.xlsx")  # fichier déjà sur disque
    cci_m = to_month_start(cci_raw.resample("ME").mean())  # niveau mensuel, MS

    viz.table_12_sentiment_regressions(
        covol2_m=covol2_m,
        vix_m=vix_m,
        epu_m=epu_m,
        cci_m=cci_m,
        hac_lags=2,
        print_out=True
    )

    #  chargement des niveaux
    emp_lvl = MacroDataLoader.load_emp()  # mensuel
    ind_lvl = MacroDataLoader.load_ind()  # mensuel
    inv_lvl = MacroDataLoader.load_inv()  # trimestriel
    cons_lvl = MacroDataLoader.load_cons()  # trimestriel

    # alignement calendrier

    def to_period_start(s, freq="M"):
        """
        Ramène l’index DatetimeIndex au premier jour de chaque
        mois ('M') ou trimestre ('Q').
        """
        s2 = s.copy()
        s2.index = (s2.index
                    .to_period(freq)
                    .to_timestamp())  # how='start' implicite
        return s2.sort_index()

    emp_lvl = to_period_start(emp_lvl, "M")
    ind_lvl = to_period_start(ind_lvl, "M")
    inv_lvl = to_period_start(inv_lvl, "Q")
    cons_lvl = to_period_start(cons_lvl, "Q")

    # log (taux de croissance)
    emp_g = np.log(emp_lvl).diff()  # mensuel
    ind_g = np.log(ind_lvl).diff()  # mensuel
    inv_g = np.log(inv_lvl).diff()  # trimestriel
    cons_g = np.log(cons_lvl).diff()  # trimestriel

    # option : drop du premier NaN
    emp_g, ind_g, inv_g, cons_g = [s.dropna() for s in (emp_g, ind_g, inv_g, cons_g)]

    covol2_q = (covol2_m).resample("Q").mean()

    # 3) Générer les tableaux
    viz.table_13_macro_reg(emp_g, ind_g, covol2_m)
    viz.table_14_macro_reg(inv_g, cons_g, covol2_q)



if __name__ == '__main__':
    """Test Monte Carlosimple"""
    #main_Monte_Carlo_1scenario()

    """Table 1 et 2"""
    #main_Monte_Carlo()

    """Table 3"""
    #replicate_test_table_Monte_Carlo(N_list=[2, 5, 50], T_list=[100, 1000],R=100000, v=0, fixed_s=False)

    """ Table 4"""
    #replicate_test_table_Monte_Carlo(N_list=[2, 5, 50], T_list=[100, 1000],R=100000, v=0.5, fixed_s=True)

    """Table 5"""
    # replicate_test_table_Monte_Carlo(N_list=[2, 5, 50], T_list=[100, 1000],R=100000, v=1.0, fixed_s=False)

    """Table C.1 Appendix (Summary Statistics)"""
    #etf_country_stats()

    """Fig 1, Fig 2, Table 6 """
    model_residuals_and_global_covol(data_save=False) #22 Itérations

    """Fig 3"""
    #estimate_s_time_series_and_plot_fig3(data_save=False, grid_search=False) #Très Long

    """Fig 4, Table 7, Table 8, Table 9, Fig 5 """
    replicate_partie_8()
