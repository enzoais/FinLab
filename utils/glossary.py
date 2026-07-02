"""
Glossaire pédagogique de FinLab (français, sans équations).

Chaque entrée explique en 3 volets : (a) c'est quoi, (b) à quoi ça sert / comment le lire,
(c) comment ça se calcule en mots simples. Affiché au clic via un bouton ⓘ (st.popover)
au contact direct de chaque chiffre/notion — pas de page séparée.

Convention : clés en anglais (snake_case), textes en français.
"""
from __future__ import annotations

GLOSSARY: dict[str, str] = {
    # ---------------------------------------------------------------- Général / marché
    "benchmark": (
        "**C'est quoi ?** Un indice de référence (ex. S&P 500, CAC 40) qui représente « le marché » "
        "ou une catégorie d'actifs.\n\n"
        "**À quoi ça sert ?** À comparer : votre actif ou portefeuille a-t-il fait mieux ou moins bien "
        "que la référence ? C'est l'étalon de mesure de la performance.\n\n"
        "**Comment ça se calcule ?** On prend l'historique de l'indice sur la même période et on met "
        "ses rendements en face des vôtres."
    ),
    "risk_free_rate": (
        "**C'est quoi ?** Le taux « sans risque » : le rendement que l'on obtient sans prendre de risque "
        "(typiquement les bons du Trésor d'État).\n\n"
        "**À quoi ça sert ?** C'est le point de départ de toute décision : un placement risqué doit "
        "rapporter plus que ce taux pour valoir le coup.\n\n"
        "**Comment ça se calcule ?** On le prend directement sur le marché (taux d'État court terme). "
        "Ici c'est un paramètre que vous fixez (ex. 5 % par an)."
    ),
    "log_returns": (
        "**C'est quoi ?** Une façon de mesurer la variation d'un prix d'un jour à l'autre, en « rendements "
        "logarithmiques » plutôt qu'en pourcentage simple.\n\n"
        "**À quoi ça sert ?** Ils s'additionnent proprement dans le temps et se prêtent bien aux calculs "
        "statistiques (moyenne, volatilité).\n\n"
        "**Comment ça se calcule ?** On regarde le rapport entre le prix d'aujourd'hui et celui d'hier, "
        "puis on en prend le logarithme. Pour de petites variations, c'est presque identique au pourcentage."
    ),
    "expected_market_return": (
        "**C'est quoi ?** Le rendement que l'on s'attend à obtenir du marché dans son ensemble.\n\n"
        "**À quoi ça sert ?** C'est un ingrédient du coût des fonds propres (CAPM) : plus on attend du "
        "marché, plus on exige de rendement d'une action risquée.\n\n"
        "**Comment ça se calcule ?** Ici, on l'estime par le rendement moyen historique de l'indice, "
        "ramené à une base annuelle."
    ),

    # ---------------------------------------------------------------- CAPM / Bêta
    "capm": (
        "**C'est quoi ?** Le modèle CAPM relie le rendement attendu d'une action à son risque de marché "
        "(son Bêta).\n\n"
        "**À quoi ça sert ?** À estimer le rendement qu'un investisseur doit exiger pour détenir l'action, "
        "et donc à valoriser une entreprise.\n\n"
        "**Comment ça se calcule ?** Rendement attendu = taux sans risque + Bêta × (rendement du marché − "
        "taux sans risque). Plus le Bêta est grand, plus on exige de rendement."
    ),
    "beta": (
        "**C'est quoi ?** Le Bêta (β) mesure à quel point une action bouge quand le marché bouge.\n\n"
        "**À quoi ça sert ?** À juger le risque de marché : un Bêta de 1,2 veut dire que si le marché monte "
        "de 1 %, l'action tend à monter de 1,2 % (plus volatile) ; 0,8 = moins volatile que le marché.\n\n"
        "**Comment ça se calcule ?** C'est la pente d'une régression entre les rendements de l'action et "
        "ceux du marché sur l'historique."
    ),
    "cost_of_equity": (
        "**C'est quoi ?** Le coût des fonds propres (Kₑ) : le rendement minimum que les actionnaires "
        "exigent pour détenir l'action.\n\n"
        "**À quoi ça sert ?** À valoriser une entreprise (actualiser ses flux futurs) et à décider si un "
        "projet crée de la valeur.\n\n"
        "**Comment ça se calcule ?** Avec le CAPM : taux sans risque + Bêta × prime de risque du marché. "
        "Un Bêta plus élevé donne un coût des fonds propres plus élevé."
    ),
    "jensen_alpha": (
        "**C'est quoi ?** L'Alpha de Jensen mesure la sur- ou sous-performance d'une action par rapport à "
        "ce que le CAPM prévoyait, compte tenu de son risque.\n\n"
        "**À quoi ça sert ?** À juger la qualité d'un gérant ou d'un titre : Alpha positif = a battu le "
        "marché à risque égal ; négatif = a sous-performé.\n\n"
        "**Comment ça se calcule ?** C'est l'ordonnée à l'origine de la régression CAPM (la partie du "
        "rendement non expliquée par le marché), ici annualisée."
    ),
    "r_squared": (
        "**C'est quoi ?** Le R² dit quelle part des mouvements de l'action est expliquée par le marché.\n\n"
        "**À quoi ça sert ?** À juger la fiabilité du Bêta : R² élevé = l'action suit surtout le marché ; "
        "R² faible = beaucoup de mouvements propres à l'entreprise (le Bêta est alors moins fiable).\n\n"
        "**Comment ça se calcule ?** C'est la qualité d'ajustement de la régression, entre 0 % et 100 %."
    ),
    "rolling_beta": (
        "**C'est quoi ?** Le Bêta glissant : le Bêta recalculé sur une fenêtre mobile (ex. les 252 derniers "
        "jours), pour voir son évolution dans le temps.\n\n"
        "**À quoi ça sert ?** À repérer si le risque de marché de l'action a changé (ex. après une fusion, "
        "un changement d'activité).\n\n"
        "**Comment ça se calcule ?** On refait la régression Bêta sur chaque fenêtre glissante et on trace "
        "la suite des valeurs obtenues."
    ),
    "regression_line": (
        "**C'est quoi ?** La droite de régression résume le lien entre les rendements de l'action (axe Y) "
        "et ceux du marché (axe X).\n\n"
        "**À quoi ça sert ?** Sa pente est le Bêta, son point de départ l'Alpha. Plus les points sont "
        "proches de la droite, mieux le marché explique l'action.\n\n"
        "**Comment ça se calcule ?** On cherche la droite qui passe « au plus près » de tous les points "
        "(moindres carrés)."
    ),

    # ---------------------------------------------------------------- Portefeuille / Markowitz
    "markowitz": (
        "**C'est quoi ?** La théorie de Markowitz cherche le meilleur mélange d'actifs pour un couple "
        "rendement / risque donné, grâce à la diversification.\n\n"
        "**À quoi ça sert ?** À répartir son capital entre plusieurs titres de façon à réduire le risque "
        "sans trop sacrifier de rendement.\n\n"
        "**Comment ça se calcule ?** On combine les rendements attendus et les corrélations des actifs, "
        "puis on optimise les poids (ex. maximiser le ratio de Sharpe)."
    ),
    "efficient_frontier": (
        "**C'est quoi ?** La frontière efficiente : l'ensemble des portefeuilles offrant le meilleur "
        "rendement possible pour chaque niveau de risque.\n\n"
        "**À quoi ça sert ?** À visualiser les meilleurs choix : on veut être sur la courbe, pas en "
        "dessous. Aller à droite = plus de risque ; monter = plus de rendement.\n\n"
        "**Comment ça se calcule ?** Pour une série de rendements cibles, on cherche le portefeuille de "
        "risque minimal, et on relie tous ces points."
    ),
    "min_variance": (
        "**C'est quoi ?** Le portefeuille de variance minimale : celui qui a le risque (volatilité) le "
        "plus faible possible.\n\n"
        "**À quoi ça sert ?** Pour un investisseur prudent qui veut avant tout limiter les fluctuations.\n\n"
        "**Comment ça se calcule ?** On optimise les poids pour minimiser la volatilité du portefeuille, "
        "sans viser un rendement particulier."
    ),
    "max_sharpe": (
        "**C'est quoi ?** Le portefeuille qui maximise le ratio de Sharpe : le meilleur rendement par "
        "unité de risque.\n\n"
        "**À quoi ça sert ?** C'est souvent le portefeuille « optimal » de référence : le plus efficace "
        "en rendement ajusté du risque.\n\n"
        "**Comment ça se calcule ?** On cherche les poids qui donnent le plus fort ratio (rendement − taux "
        "sans risque) / volatilité."
    ),
    "sharpe_ratio": (
        "**C'est quoi ?** Le ratio de Sharpe mesure le rendement obtenu par unité de risque pris.\n\n"
        "**À quoi ça sert ?** À comparer des placements : plus il est élevé, mieux c'est. Au-dessus de 1, "
        "c'est généralement considéré comme bon.\n\n"
        "**Comment ça se calcule ?** (Rendement du portefeuille − taux sans risque) divisé par sa "
        "volatilité, le tout annualisé."
    ),
    "portfolio_volatility": (
        "**C'est quoi ?** La volatilité du portefeuille : l'ampleur de ses fluctuations, exprimée en % "
        "par an.\n\n"
        "**À quoi ça sert ?** À mesurer le risque : plus elle est basse, plus la valeur est stable.\n\n"
        "**Comment ça se calcule ?** À partir de l'écart-type des rendements, en tenant compte des poids "
        "et des corrélations entre actifs, annualisé."
    ),
    "expected_return": (
        "**C'est quoi ?** Le rendement attendu : la performance annuelle moyenne espérée du portefeuille.\n\n"
        "**À quoi ça sert ?** À poser ses attentes et à les comparer au taux sans risque.\n\n"
        "**Comment ça se calcule ?** Moyenne des rendements historiques de chaque actif, pondérée par les "
        "poids, ramenée à une base annuelle."
    ),
    "diversification_ratio": (
        "**C'est quoi ?** Le ratio de diversification mesure le bénéfice obtenu en mélangeant des actifs "
        "qui ne bougent pas ensemble.\n\n"
        "**À quoi ça sert ?** Au-dessus de 1, la diversification a bel et bien réduit le risque global.\n\n"
        "**Comment ça se calcule ?** On compare la somme des volatilités individuelles (pondérées) à la "
        "volatilité réelle du portefeuille."
    ),
    "optimal_weights": (
        "**C'est quoi ?** Les poids optimaux : la part de capital à allouer à chaque actif, en % (leur "
        "somme fait 100 %).\n\n"
        "**À quoi ça sert ?** C'est la recommandation d'allocation concrète pour rééquilibrer son "
        "portefeuille.\n\n"
        "**Comment ça se calcule ?** L'optimiseur choisit les poids qui atteignent l'objectif (ex. max "
        "Sharpe) en respectant vos contraintes."
    ),
    "covariance": (
        "**C'est quoi ?** La covariance mesure si deux actifs ont tendance à monter et descendre "
        "ensemble.\n\n"
        "**À quoi ça sert ?** C'est le cœur de la diversification : mélanger des actifs peu liés réduit le "
        "risque total.\n\n"
        "**Comment ça se calcule ?** À partir de l'historique des rendements, on mesure comment deux "
        "séries varient conjointement."
    ),
    "correlation": (
        "**C'est quoi ?** La corrélation, entre −1 et +1, dit à quel point deux actifs bougent ensemble.\n\n"
        "**À quoi ça sert ?** Des corrélations faibles (ou négatives) permettent de mieux réduire le "
        "risque par diversification.\n\n"
        "**Comment ça se calcule ?** C'est la covariance normalisée par les volatilités des deux actifs, "
        "ramenée sur l'échelle −1 à +1."
    ),
    "var": (
        "**C'est quoi ?** La VaR (Value at Risk) est une mesure de perte : le pire rendement attendu dans "
        "les mauvais jours, à un niveau de confiance donné.\n\n"
        "**À quoi ça sert ?** À chiffrer le risque de baisse. Ex. VaR 95 % à −1,5 % / jour : dans les 5 % "
        "de pires jours, on perd au moins 1,5 %.\n\n"
        "**Comment ça se calcule ?** On classe les rendements historiques et on lit le seuil du 5ᵉ centile "
        "(pour une confiance de 95 %)."
    ),
    "cvar": (
        "**C'est quoi ?** La CVaR (Expected Shortfall) est la perte moyenne dans les pires cas, au-delà de "
        "la VaR.\n\n"
        "**À quoi ça sert ?** À mesurer le risque de queue : elle dit « quand ça va mal, à quel point en "
        "moyenne ». Elle est toujours plus sévère que la VaR.\n\n"
        "**Comment ça se calcule ?** On prend la moyenne des rendements situés dans la pire tranche "
        "(ex. les 5 % de pires jours)."
    ),
    "tracking_error": (
        "**C'est quoi ?** La tracking error mesure de combien votre portefeuille s'écarte du benchmark au "
        "fil du temps.\n\n"
        "**À quoi ça sert ?** Faible = vous collez à l'indice ; élevée = gestion très active, plus "
        "éloignée de la référence.\n\n"
        "**Comment ça se calcule ?** C'est la volatilité de la différence (rendement du portefeuille − "
        "rendement du benchmark), annualisée."
    ),
    "information_ratio": (
        "**C'est quoi ?** Le ratio d'information rapporte la surperformance (alpha) au risque pris pour "
        "s'écarter du benchmark.\n\n"
        "**À quoi ça sert ?** À juger l'efficacité d'une gestion active : plus il est élevé, plus la "
        "surperformance est régulière.\n\n"
        "**Comment ça se calcule ?** Alpha divisé par la tracking error."
    ),
    "alpha": (
        "**C'est quoi ?** L'alpha est la surperformance de votre portefeuille par rapport au benchmark.\n\n"
        "**À quoi ça sert ?** Positif = vous avez fait mieux que la référence ; négatif = moins bien.\n\n"
        "**Comment ça se calcule ?** Rendement du portefeuille − rendement du benchmark, sur la même "
        "période."
    ),

    # ---------------------------------------------------------------- Obligations
    "bond_price": (
        "**C'est quoi ?** Le prix d'une obligation : ce qu'elle vaut aujourd'hui, exprimé pour 100 de "
        "nominal.\n\n"
        "**À quoi ça sert ?** À savoir si l'obligation est chère ou bon marché (100 = au pair). On le "
        "compare au prix de marché.\n\n"
        "**Comment ça se calcule ?** On additionne la valeur d'aujourd'hui de tous les flux futurs "
        "(coupons + remboursement), actualisés au taux de rendement (YTM)."
    ),
    "ytm": (
        "**C'est quoi ?** Le taux de rendement actuariel (YTM) : le rendement annuel obtenu si on garde "
        "l'obligation jusqu'à l'échéance.\n\n"
        "**À quoi ça sert ?** À comparer des obligations entre elles. Il bouge à l'inverse du prix : quand "
        "les taux montent, le prix baisse.\n\n"
        "**Comment ça se calcule ?** C'est le taux qui rend la valeur actualisée de tous les flux égale au "
        "prix de l'obligation."
    ),
    "coupon_rate": (
        "**C'est quoi ?** Le taux de coupon : l'intérêt annuel versé par l'obligation, en % du nominal.\n\n"
        "**À quoi ça sert ?** À connaître le revenu régulier reçu (ex. 5 % sur 100 = 5 par an).\n\n"
        "**Comment ça se calcule ?** Il est fixé à l'émission ; le coupon versé = taux de coupon × nominal, "
        "réparti selon la fréquence (annuel, semestriel…)."
    ),
    "face_value": (
        "**C'est quoi ?** Le nominal (valeur faciale) : le montant remboursé à l'échéance.\n\n"
        "**À quoi ça sert ?** C'est la base de calcul des coupons et le capital rendu à la fin.\n\n"
        "**Comment ça se calcule ?** C'est une donnée du contrat (souvent 100 ou 1000)."
    ),
    "spread_vs_rf": (
        "**C'est quoi ?** L'écart de taux (spread) : le YTM de l'obligation moins le taux sans risque.\n\n"
        "**À quoi ça sert ?** À voir la rémunération supplémentaire pour le risque pris (crédit, "
        "liquidité). Positif = l'obligation rapporte plus que le sans-risque.\n\n"
        "**Comment ça se calcule ?** YTM − taux sans risque, exprimé en %."
    ),
    "macaulay_duration": (
        "**C'est quoi ?** La duration de Macaulay : la durée moyenne (en années) au bout de laquelle on "
        "récupère ses flux.\n\n"
        "**À quoi ça sert ?** Plus elle est longue, plus l'obligation est sensible aux variations de "
        "taux.\n\n"
        "**Comment ça se calcule ?** C'est la moyenne des échéances de chaque flux, pondérée par leur "
        "valeur actualisée."
    ),
    "modified_duration": (
        "**C'est quoi ?** La duration modifiée : la variation approximative du prix (en %) pour une hausse "
        "de 1 % des taux.\n\n"
        "**À quoi ça sert ?** C'est la mesure clé du risque de taux. Ex. duration 7 → le prix baisse "
        "d'environ 7 % si les taux montent de 1 %.\n\n"
        "**Comment ça se calcule ?** C'est la duration de Macaulay ajustée par le taux de rendement."
    ),
    "convexity": (
        "**C'est quoi ?** La convexité affine la duration : elle capte l'effet non linéaire quand les taux "
        "bougent beaucoup.\n\n"
        "**À quoi ça sert ?** Une convexité positive est favorable : le prix monte plus quand les taux "
        "baissent qu'il ne baisse quand les taux montent.\n\n"
        "**Comment ça se calcule ?** À partir des flux et de leurs échéances, elle mesure la « courbure » "
        "de la relation prix-taux."
    ),
    "price_yield_curve": (
        "**C'est quoi ?** La courbe prix-taux montre comment le prix de l'obligation évolue quand son "
        "rendement (YTM) change.\n\n"
        "**À quoi ça sert ?** À visualiser le lien inverse prix / taux et à situer votre obligation "
        "dessus.\n\n"
        "**Comment ça se calcule ?** On recalcule le prix pour toute une gamme de taux et on trace la "
        "courbe (bombée = convexe)."
    ),
    "clean_dirty_price": (
        "**C'est quoi ?** Le prix pied de coupon (clean) est le prix affiché ; le prix plein coupon (dirty) "
        "est ce que l'on paie réellement.\n\n"
        "**À quoi ça sert ?** À la transaction : l'acheteur paie le prix clean + les intérêts courus depuis "
        "le dernier coupon.\n\n"
        "**Comment ça se calcule ?** Prix dirty = prix clean + intérêts courus."
    ),
    "accrued_interest": (
        "**C'est quoi ?** Les intérêts courus : la part de coupon déjà « gagnée » depuis le dernier "
        "versement.\n\n"
        "**À quoi ça sert ?** À répartir équitablement le coupon entre vendeur et acheteur lors d'une "
        "vente en cours de période.\n\n"
        "**Comment ça se calcule ?** On prend le coupon de la période au prorata des jours écoulés depuis "
        "le dernier coupon (convention 30/360)."
    ),

    # ---------------------------------------------------------------- Options / Black-Scholes
    "black_scholes": (
        "**C'est quoi ?** Le modèle de Black-Scholes donne le prix théorique d'une option européenne.\n\n"
        "**À quoi ça sert ?** À estimer la juste valeur d'un call ou d'un put et à la comparer au prix de "
        "marché.\n\n"
        "**Comment ça se calcule ?** À partir du prix du sous-jacent, du strike, de l'échéance, du taux "
        "sans risque et de la volatilité, via une formule fermée."
    ),
    "call": (
        "**C'est quoi ?** Un call est le droit (pas l'obligation) d'acheter un actif à un prix fixé "
        "(strike) avant une date donnée.\n\n"
        "**À quoi ça sert ?** À parier sur la hausse ou à se couvrir : on gagne si le sous-jacent monte "
        "au-dessus du strike.\n\n"
        "**Comment ça se calcule ?** À l'échéance, il vaut la différence (prix − strike) si elle est "
        "positive, sinon zéro."
    ),
    "put": (
        "**C'est quoi ?** Un put est le droit de vendre un actif à un prix fixé (strike) avant une date "
        "donnée.\n\n"
        "**À quoi ça sert ?** À parier sur la baisse ou à s'assurer contre une chute : on gagne si le "
        "sous-jacent descend sous le strike.\n\n"
        "**Comment ça se calcule ?** À l'échéance, il vaut la différence (strike − prix) si elle est "
        "positive, sinon zéro."
    ),
    "implied_volatility": (
        "**C'est quoi ?** La volatilité implicite : la volatilité qui rend le prix du modèle égal au prix "
        "de marché de l'option.\n\n"
        "**À quoi ça sert ?** Elle reflète l'incertitude anticipée par le marché. Plus elle est élevée, "
        "plus le marché attend de mouvements.\n\n"
        "**Comment ça se calcule ?** On ajuste la volatilité dans Black-Scholes jusqu'à retrouver "
        "exactement le prix observé."
    ),
    "delta": (
        "**C'est quoi ?** Le Delta (Δ) mesure de combien le prix de l'option bouge quand le sous-jacent "
        "bouge de 1 €.\n\n"
        "**À quoi ça sert ?** À se couvrir : un Delta de 0,5 signifie qu'il faut environ une demi-action "
        "pour neutraliser l'option. Pour un call, il approche aussi la probabilité de finir gagnant.\n\n"
        "**Comment ça se calcule ?** C'est la sensibilité (pente) du prix de l'option par rapport au prix "
        "du sous-jacent."
    ),
    "gamma": (
        "**C'est quoi ?** Le Gamma (Γ) mesure la vitesse à laquelle le Delta change quand le sous-jacent "
        "bouge.\n\n"
        "**À quoi ça sert ?** À savoir à quelle fréquence réajuster une couverture. Il est le plus fort à "
        "la monnaie (prix ≈ strike).\n\n"
        "**Comment ça se calcule ?** C'est la sensibilité du Delta au prix du sous-jacent (une « dérivée "
        "seconde »)."
    ),
    "theta": (
        "**C'est quoi ?** Le Theta (Θ) mesure la perte de valeur de l'option due au temps qui passe, par "
        "jour.\n\n"
        "**À quoi ça sert ?** À chiffrer l'érosion quotidienne : une option perd de la valeur temps en "
        "s'approchant de l'échéance (Theta généralement négatif).\n\n"
        "**Comment ça se calcule ?** C'est la variation du prix de l'option pour un jour de moins jusqu'à "
        "l'échéance."
    ),
    "vega": (
        "**C'est quoi ?** Le Vega mesure la sensibilité du prix de l'option à une hausse de 1 % de la "
        "volatilité.\n\n"
        "**À quoi ça sert ?** À gérer le risque de volatilité : un Vega élevé = l'option gagne beaucoup "
        "si la volatilité monte.\n\n"
        "**Comment ça se calcule ?** C'est la sensibilité du prix à la volatilité ; elle est maximale à "
        "la monnaie."
    ),
    "rho": (
        "**C'est quoi ?** Le Rho mesure la sensibilité du prix de l'option à une hausse de 1 % du taux "
        "sans risque.\n\n"
        "**À quoi ça sert ?** À évaluer l'effet des taux d'intérêt, souvent secondaire pour des options "
        "courtes.\n\n"
        "**Comment ça se calcule ?** C'est la sensibilité du prix de l'option au taux d'intérêt."
    ),
    "spot": (
        "**C'est quoi ?** Le spot (S) est le prix actuel du sous-jacent (l'action, l'indice…).\n\n"
        "**À quoi ça sert ?** C'est l'entrée principale du modèle : la valeur de l'option en dépend "
        "directement.\n\n"
        "**Comment ça se calcule ?** On le lit sur le marché (dernier cours) ou on le saisit à la main."
    ),
    "strike": (
        "**C'est quoi ?** Le strike (K) est le prix d'exercice fixé de l'option.\n\n"
        "**À quoi ça sert ?** Il détermine le gain : un call est « dans la monnaie » si le spot est "
        "au-dessus du strike.\n\n"
        "**Comment ça se calcule ?** C'est une donnée du contrat d'option (vous la choisissez ici)."
    ),
    "time_to_expiry": (
        "**C'est quoi ?** Le temps jusqu'à l'échéance (T), exprimé en années.\n\n"
        "**À quoi ça sert ?** Plus l'échéance est lointaine, plus l'option a de la valeur temps.\n\n"
        "**Comment ça se calcule ?** Nombre de jours restants ramené en fraction d'année (ex. 0,5 = "
        "6 mois)."
    ),
    "volatility": (
        "**C'est quoi ?** La volatilité (σ) mesure l'ampleur des fluctuations du sous-jacent, en % par "
        "an.\n\n"
        "**À quoi ça sert ?** C'est l'ingrédient clé du prix d'une option : plus c'est volatil, plus "
        "l'option vaut cher.\n\n"
        "**Comment ça se calcule ?** À partir de l'écart-type des rendements passés (volatilité "
        "historique) ou déduite du marché (implicite)."
    ),
    "intrinsic_value": (
        "**C'est quoi ?** La valeur intrinsèque : ce que vaudrait l'option si elle expirait maintenant.\n\n"
        "**À quoi ça sert ?** À distinguer la partie « déjà acquise » de la valeur temps (le reste du "
        "prix).\n\n"
        "**Comment ça se calcule ?** Pour un call : prix − strike si positif, sinon 0. Pour un put : "
        "strike − prix si positif, sinon 0."
    ),
    "payoff": (
        "**C'est quoi ?** Le payoff est le gain de l'option à l'échéance selon le prix du sous-jacent.\n\n"
        "**À quoi ça sert ?** À visualiser le résultat final (P&L) : jusqu'où on gagne, à partir de "
        "quand.\n\n"
        "**Comment ça se calcule ?** C'est la valeur intrinsèque à l'échéance (sans valeur temps)."
    ),

    # ---------------------------------------------------------------- Monte Carlo / simulation
    "monte_carlo": (
        "**C'est quoi ?** Une simulation de Monte-Carlo génère des milliers de futurs possibles pour un "
        "capital, en tirant des rendements au hasard.\n\n"
        "**À quoi ça sert ?** À raisonner en distribution plutôt qu'en chiffre unique : « dans 20 ans, "
        "le capital sera probablement entre X et Y ».\n\n"
        "**Comment ça se calcule ?** On simule de nombreuses trajectoires (ici par le modèle GBM) et on "
        "analyse l'ensemble des résultats."
    ),
    "gbm": (
        "**C'est quoi ?** Le mouvement brownien géométrique (GBM) est le modèle standard d'évolution d'un "
        "prix : une tendance (drift) plus du hasard (volatilité).\n\n"
        "**À quoi ça sert ?** À simuler des trajectoires réalistes de prix ou de patrimoine.\n\n"
        "**Comment ça se calcule ?** À chaque pas de temps, le prix évolue selon le drift et un tirage "
        "aléatoire proportionnel à la volatilité."
    ),
    "drift": (
        "**C'est quoi ?** Le drift (μ) est la tendance de fond : le rendement annuel moyen attendu.\n\n"
        "**À quoi ça sert ?** Il oriente la pente générale des trajectoires simulées (à la hausse si "
        "positif).\n\n"
        "**Comment ça se calcule ?** On le fixe à la main, ou on l'estime à partir du rendement moyen "
        "historique d'un titre."
    ),
    "terminal_wealth": (
        "**C'est quoi ?** Le patrimoine terminal : la valeur du capital à la fin de l'horizon simulé.\n\n"
        "**À quoi ça sert ?** C'est le résultat qui compte : on en regarde la moyenne, la médiane et les "
        "extrêmes.\n\n"
        "**Comment ça se calcule ?** C'est la dernière valeur de chaque trajectoire ; on résume ensuite "
        "l'ensemble par des statistiques."
    ),
    "fan_chart": (
        "**C'est quoi ?** Le fan chart (graphe en éventail) montre la fourchette des trajectoires possibles "
        "au fil du temps.\n\n"
        "**À quoi ça sert ?** À communiquer l'incertitude : la bande large regroupe la plupart des "
        "scénarios, la ligne centrale est la trajectoire médiane.\n\n"
        "**Comment ça se calcule ?** À chaque date, on calcule des percentiles (5 %, 25 %, 50 %, 75 %, "
        "95 %) sur toutes les trajectoires."
    ),
    "shortfall_probability": (
        "**C'est quoi ?** La probabilité de shortfall : la chance que le capital final passe sous un seuil "
        "fixé (ex. zéro, ou un objectif).\n\n"
        "**À quoi ça sert ?** À chiffrer un risque concret : « quelle probabilité de ruine ou de ne pas "
        "atteindre mon objectif ? ».\n\n"
        "**Comment ça se calcule ?** C'est la part des trajectoires simulées qui finissent sous le seuil."
    ),
    "max_drawdown": (
        "**C'est quoi ?** Le max drawdown : la pire baisse du sommet au creux subie au cours de la "
        "période.\n\n"
        "**À quoi ça sert ?** À mesurer la douleur maximale à traverser (ex. −30 % avant de remonter).\n\n"
        "**Comment ça se calcule ?** On suit le plus haut atteint, puis on mesure la plus forte chute "
        "sous ce sommet."
    ),

    # ---------------------------------------------------------------- Backtest
    "backtest": (
        "**C'est quoi ?** Un backtest rejoue une stratégie sur des données historiques réelles, comme si "
        "on l'avait suivie.\n\n"
        "**À quoi ça sert ?** À évaluer une allocation sur le passé : rendement, risque, pires baisses, "
        "comparaison au marché.\n\n"
        "**Comment ça se calcule ?** On applique les poids choisis aux rendements historiques et on "
        "construit la courbe de capital dans le temps."
    ),
    "capital_curve": (
        "**C'est quoi ?** La courbe de capital montre l'évolution de la valeur du portefeuille dans le "
        "temps, à partir d'un capital initial.\n\n"
        "**À quoi ça sert ?** À voir d'un coup d'œil la performance et les phases de baisse, et à la "
        "comparer au benchmark.\n\n"
        "**Comment ça se calcule ?** On part du capital de départ et on le fait grandir jour après jour "
        "selon les rendements du portefeuille."
    ),
    "total_return": (
        "**C'est quoi ?** Le rendement total : le gain (ou la perte) sur toute la période, en %.\n\n"
        "**À quoi ça sert ?** À résumer la performance globale entre le début et la fin.\n\n"
        "**Comment ça se calcule ?** (Valeur finale − valeur initiale) / valeur initiale."
    ),
    "cagr": (
        "**C'est quoi ?** Le CAGR est le taux de croissance annuel moyen équivalent sur la période.\n\n"
        "**À quoi ça sert ?** À comparer des stratégies de durées différentes sur une base annuelle "
        "commune.\n\n"
        "**Comment ça se calcule ?** C'est le taux annuel constant qui mène de la valeur initiale à la "
        "valeur finale sur la durée écoulée."
    ),
    "annualized_volatility": (
        "**C'est quoi ?** La volatilité annualisée : l'ampleur des fluctuations du portefeuille, ramenée à "
        "une base annuelle.\n\n"
        "**À quoi ça sert ?** À mesurer le risque et à comparer avec d'autres placements ou le "
        "benchmark.\n\n"
        "**Comment ça se calcule ?** À partir de l'écart-type des rendements quotidiens, multiplié pour "
        "passer à l'année."
    ),
    "var_historical": (
        "**C'est quoi ?** La VaR historique : le pire rendement quotidien attendu dans les mauvais jours, "
        "estimé sur l'historique réel.\n\n"
        "**À quoi ça sert ?** À chiffrer le risque de baisse observé sur la période testée.\n\n"
        "**Comment ça se calcule ?** On classe les rendements passés et on lit le seuil du 5ᵉ centile "
        "(confiance 95 %)."
    ),
    "cvar_historical": (
        "**C'est quoi ?** La CVaR historique : la perte moyenne dans les pires jours observés, au-delà de "
        "la VaR.\n\n"
        "**À quoi ça sert ?** À mesurer la sévérité des mauvais jours réels (risque de queue).\n\n"
        "**Comment ça se calcule ?** Moyenne des rendements de la pire tranche (les 5 % de pires jours)."
    ),
    "rebalancing": (
        "**C'est quoi ?** Le rééquilibrage remet périodiquement le portefeuille aux poids cibles (ex. "
        "chaque mois ou trimestre).\n\n"
        "**À quoi ça sert ?** À garder l'allocation voulue : sans rééquilibrage, les actifs qui montent "
        "prennent trop de place (buy & hold).\n\n"
        "**Comment ça se calcule ?** Aux dates de rééquilibrage, on revend/rachète pour revenir aux "
        "pourcentages cibles."
    ),
    "buy_and_hold": (
        "**C'est quoi ?** Le buy & hold : on achète l'allocation de départ et on ne touche plus rien.\n\n"
        "**À quoi ça sert ?** C'est la stratégie la plus simple ; les poids dérivent naturellement selon "
        "la performance de chaque actif.\n\n"
        "**Comment ça se calcule ?** On laisse chaque position évoluer librement, sans rééquilibrage."
    ),
    "equal_weight": (
        "**C'est quoi ?** L'équipondération : on met la même part de capital sur chaque actif.\n\n"
        "**À quoi ça sert ?** C'est une allocation neutre et robuste, sans pari sur un titre en "
        "particulier.\n\n"
        "**Comment ça se calcule ?** Chaque poids = 100 % divisé par le nombre d'actifs."
    ),
}


def get_glossary_text(term_key: str) -> str:
    """Retourne l'explication d'un terme, ou un message par défaut si la clé est inconnue."""
    return GLOSSARY.get(term_key, "Explication non disponible pour ce terme.")
