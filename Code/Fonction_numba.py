from numba import njit
import numpy as np



@njit
def estimate_grid_search(e, s0, x0, grid_x, grid_s, max_iter=30, tol=1e-6):
    """
    Estimation du modèle Global COVOL via grid search entièrement accélérée avec Numba.
    """

    T, N = e.shape
    x = x0.copy()
    s = s0.copy()

    # Calcul initial de log-vraisemblance
    ll_old = -0.5 * np.sum(np.log(np.outer(x, s) + (1 - s)) + e**2 / (np.outer(x, s) + (1 - s)))

    for it in range(max_iter):
        # Met à jour x et s
        x = update_x_grid_search_numba(e, s, grid_x)
        s = update_s_grid_search_numba(e, x, grid_s)

        # Normalisation de x (comme dans normalize_params)
        x /= np.mean(x)

        # Recalcule de la log-vraisemblance
        g = np.outer(x, s) + (1 - s)
        ll_new = -0.5 * np.sum(np.log(g) + e**2 / g)

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    return x, s




@njit
def update_x_grid_search_numba(e, s, grid):
    """
    Met à jour le vecteur x (de taille T) via une recherche en grille.
    Pour chaque instant t, cherche le x_t dans grid qui maximise la log-vraisemblance.
    """

    T, N = e.shape
    new_x = np.zeros(T)

    # Boucle sur chaque période t
    for t in range(T):
        max_ll = -1e10  # Valeur initiale très basse pour initialiser
        best_x = grid[0]  # Valeur par défaut si aucun max trouvé
        # Parcourt toutes les valeurs candidates de la grille
        for x_val in grid:
            ll_val = 0.0
            # Contribution de chaque actif à la log-vraisemblance pour x_t = x_val
            for i in range(N):
                g = s[i] * x_val + (1 - s[i])
                ll_val += np.log(g) + (e[t, i] ** 2) / g
            ll_val = -0.5 * ll_val  # log-vraisemblance négative (à minimiser)
            # Mise à jour du meilleur x_t si la log-vraisemblance est améliorée
            if ll_val > max_ll:
                max_ll = ll_val
                best_x = x_val
        new_x[t] = best_x  # Meilleure valeur trouvée pour x_t
    return new_x


@njit
def update_s_grid_search_numba(e, x, grid):
    """
    Met à jour le vecteur s (de taille N) via une recherche en grille.
    Pour chaque actif i, cherche le s_i dans grid qui maximise la log-vraisemblance.
    """

    T, N = e.shape
    new_s = np.zeros(N)

    # Boucle sur chaque actif i
    for i in range(N):
        max_ll = -1e10
        best_s = grid[0]
        # Parcourt toutes les valeurs candidates de la grille pour s_i
        for s_val in grid:
            ll_val = 0.0
            # Contribution de chaque période t à la log-vraisemblance pour s_i = s_val
            for t in range(T):
                g = s_val * x[t] + (1 - s_val)
                ll_val += np.log(g) + (e[t, i] ** 2) / g
            ll_val = -0.5 * ll_val
            # Mise à jour du meilleur s_i si la log-vraisemblance est améliorée
            if ll_val > max_ll:
                max_ll = ll_val
                best_s = s_val
        new_s[i] = best_s  # Meilleure valeur trouvée pour s_i
    return new_s