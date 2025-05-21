import numpy as np
import os
import pandas as pd

"""
Script contenant les différentes classes de gestion des données.
"""

class MonteCarlo_Data:
    """
    Classe qui simule les données utilisées pour la simulation Monte Carlo
    """
    def __init__(self, N=10, T=100, R=150, v=2.0, seed=None,x_fixed = False):
        self.N =N #Nombre d'actifs simulés
        self.T = T #Nombre de pas de temps simulés
        self.R = R #Nombre de simulation (réplique)
        self.v = v #Ecart type de la loi normal pour la simulation des x
        self.x_fixed = x_fixed

        if seed is not None: np.random.seed(seed) # Si on veut fixer une seed
        self.s = np.random.uniform(0, 1, size=N) #Tirage des facteurs loadings

        # Génération unique de x si fixé
        if self.x_fixed:
            phi = np.random.normal(0, self.v, size=T)
            x = np.exp(phi)
            self.x = x / np.mean(x)


    def simulate_replication(self):
        """
        Méthode de simulation des x et e pour une réplique du scenario Monte Carlo
        """

        # Génération des innovations ~ N(0,1) pour chaque actif et temps
        eps = np.random.normal(0, 1, size=(self.T, self.N))

        # Génération de x_t (soit fixe, soit spécifique à la réplication)
        if self.x_fixed:
            x = self.x
        else:
            phi = np.random.normal(0, self.v, size=self.T)
            x = np.exp(phi)
            x /= np.mean(x)

        # Calcul de g(s, x) = s*x + (1-s) (différentes expositions selon actif)
        g = self.s * x[:, None] + (1 - self.s)

        # Calcul des résidus : e = sqrt(g(s, x)) * ε
        e = np.sqrt(g) * eps

        return x, e

    def run_simulations(self):
        """
        Méthode de lancement des simulations de Monte Carlo, pour obtenir les données x et e pour chaque réplique
        """
        #Initialisation des tableaux
        X = np.empty((self.R, self.T)) #Taille R*T (nb simul * nb pas de temps), car facteur global covol commun à tous les actifs
        E = np.empty((self.R, self.T, self.N)) # Taille R*T*N (3 dimensions) car résidu propre à chaque actif, pour chaque pas de temps, resimulés à chaque scénario

        #Boucle itérant sur chaque réplique (scénario)
        for r in range(self.R):
            x, e = self.simulate_replication() #Simulation des x et e pour le scénario
            X[r] = x
            E[r] = e
        return X, E



class Excel_data:
    """Classe pour charger les fichiers Excel (ETF)"""

    def __init__(self, file_name, sheet_name=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Chemin du script actuel
        root_dir = os.path.abspath(os.path.join(base_dir, "..")) # Revenir d’un niveau pour pointer vers le dossier
        data_dir = os.path.join(root_dir, "Data")
        self.full_path = os.path.join(data_dir, file_name)

        self.sheet_name = sheet_name
        self.data = None

        self.load_data()

    def load_data(self):
        """
        Charge les données depuis le fichier Excel.
        """
        if not os.path.exists(self.full_path):
            raise FileNotFoundError(f"Fichier non trouvé : {self.full_path}")

        self.data = pd.read_excel(self.full_path,sheet_name=self.sheet_name,
                    header=0,index_col=0)

    def get_data(self):
        """
        Retourne les données chargées.
        """
        return self.data



class DataUtils:
    """
    Classe utilitaire pour effectuer des opérations courantes sur les données :
    """

    @staticmethod
    def compute_returns(df: pd.DataFrame, method="log"):
        """
        Calcule les rendements à partir d’un DataFrame de prix.
        """

        df = df.astype(float)

        if method == "log":
            returns = np.log(df / df.shift(1))
        elif method == "simple":
            returns = df.pct_change()
        else:
            raise ValueError("Méthode invalide : choisir 'log' ou 'simple'")

        return returns.iloc[1:]

    @staticmethod
    def align_with_benchmark(etf_data: pd.DataFrame, bench_data: pd.DataFrame):
        """
        Aligne les données avec celles du benchmark :
        """

        common_dates = bench_data.index.intersection(etf_data.index)
        etf_aligned = etf_data.loc[common_dates].copy()
        bench_aligned = bench_data.loc[common_dates].copy()

        etf_aligned = DataUtils.ffill_after_first_obs(etf_aligned)

        return etf_aligned, bench_aligned


    @staticmethod
    def ffill_after_first_obs(df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill uniquement à partir de la première observation non-NaN
        de chaque colonne.
        """

        df_out = df.copy()
        for col in df_out.columns:
            first_idx = df_out[col].first_valid_index()
            if first_idx is not None:  # l’ETF existe
                df_out.loc[first_idx:, col] = df_out.loc[first_idx:, col].ffill()
        return df_out


class MacroDataLoader:
    """
    Classe dédiée au chargement et gestion des données Macro, EPU, GPR, VIX, CCI ....
    """

    @staticmethod
    def load_us_epu(file_name, sheet_name=None):
        """
        Charge le fichier EPU.
        Renvoie une Series mensuelle indexée par DatetimeIndex (1er jour du mois).
        """

        excel_loader = Excel_data(file_name=file_name, sheet_name="Data")
        df =  excel_loader.get_data()

        df.reset_index(inplace=True)
        df.columns = ["Year", "Month", "1. Economic Policy Uncertainty"]

        df = df.dropna(subset=["Year"])
        df = df[df["Year"].apply(lambda x: str(x).isdigit())]

        # Conversion float + Remplace virgule par point si besoin
        df["Year"] = df["Year"].astype(int)
        df["Month"] = df["Month"].astype(int)

        # Création d'une date = 1er jour du mois
        df["date"] = pd.to_datetime(df["Year"].astype(str) + "-"
                                    + df["Month"].astype(str) + "-01",
                                    format="%Y-%m-%d")

        # On met la colonne "EPU_raw" en float
        df["1. Economic Policy Uncertainty"] = df["1. Economic Policy Uncertainty"].replace(",", ".", regex=True).astype(float)

        # Index = date
        df.set_index("date", inplace=True)

        # On garde juste la colonne EPU_raw, on la renomme:
        df.rename(columns={"1. Economic Policy Uncertainty": "EPU"}, inplace=True)

        # On s'assure qu'elle est ordonnée par date
        df.sort_index(inplace=True)

        # Retourne une Series
        return df["EPU"]

    @staticmethod
    def load_gpr(file_name, sheet_name=None):
        """
        Charge le fichier GPR.
        """

        excel_loader = Excel_data(file_name=file_name, sheet_name="Data")
        df = excel_loader.get_data()

        df.reset_index(inplace=True)
        df.columns = ["Date", "GPR"]
        df["Date"] = pd.to_datetime(df["Date"])

        # On se met sur l'index
        df.set_index("Date", inplace=True)

        df.index = df.index.to_period("M").to_timestamp(how="start")

        return df["GPR"]

    @staticmethod
    def load_vix(file_name="VIX.xlsx"):
        excel_loader = Excel_data(file_name=file_name, sheet_name="Data")
        df = excel_loader.get_data()

        df = df.rename(columns={"VIXCLS": "VIX"})
        df.index = pd.to_datetime(df.index)

        return df

    @staticmethod
    def load_cci(file_name="CCI.xlsx"):
        excel_loader = Excel_data(file_name=file_name, sheet_name="Data")
        df = excel_loader.get_data()
        df.index = pd.to_datetime(df.index)

        return df

    @staticmethod
    def load_emp(file_name="EMP.xlsx", sheet_name="Data"):
        """Total Private Employment, mensuel CVS (FRED : USPRIV)."""
        excel_loader = Excel_data(file_name=file_name,
                                  sheet_name=sheet_name)
        df = excel_loader.get_data()
        df.index = pd.to_datetime(df.index)
        return df.iloc[:, 0].astype(float)  # Series

    @staticmethod
    def load_ind(file_name="IND.xlsx", sheet_name="Data"):
        """Industrial Production Index, mensuel CVS (FRED : INDPRO)."""
        excel_loader = Excel_data(file_name=file_name,
                                  sheet_name=sheet_name)
        df = excel_loader.get_data()
        df.index = pd.to_datetime(df.index)
        return df.iloc[:, 0].astype(float)

    @staticmethod
    def load_inv(file_name="INV.xlsx", sheet_name="Data"):
        """Real Gross Private Fixed Investment, trimestriel CVS (FRED : GPDIC1)."""
        excel_loader = Excel_data(file_name=file_name,
                                  sheet_name=sheet_name)
        df = excel_loader.get_data()
        df.index = pd.to_datetime(df.index)
        return df.iloc[:, 0].astype(float)

    @staticmethod
    def load_cons(file_name="CONS.xlsx", sheet_name="Data"):
        """Real Personal Consumption Expenditures, trimestriel CVS (FRED : PCECC96)."""
        excel_loader = Excel_data(file_name=file_name,
                                  sheet_name=sheet_name)
        df = excel_loader.get_data()
        df.index = pd.to_datetime(df.index)
        return df.iloc[:, 0].astype(float)


class ResultManager:
    """
    Classe pour gérer la sauvegarde et le chargement des résultats dans le dossier ../Result.
    """

    def __init__(self, folder_name="Result"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.result_dir = os.path.abspath(os.path.join(base_dir, "..", folder_name))
        os.makedirs(self.result_dir, exist_ok=True)

    def get_path(self, name):
        return os.path.join(self.result_dir, f"{name}.xlsx")

    def save_dataframe(self, df, name):
        # Si df n'est pas un DataFrame, on le convertit en DataFrame
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        path = self.get_path(name)
        df.to_excel(path)

    def load_dataframe(self, name):
        path = self.get_path(name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier {path} introuvable.")
        return pd.read_excel(path, index_col=0)

    def save_series(self, series, name):
        df = pd.DataFrame(series)
        self.save_dataframe(df, name)

    def load_series(self, name):
        df = self.load_dataframe(name)
        return df.squeeze("columns")

    def file_exists(self, name):
        return os.path.exists(self.get_path(name))



