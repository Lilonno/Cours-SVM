# SVM - projet sur la classification des étoiles

Ce projet s'inscrit dans une démarche d'apprentissage automatique appliqué à l'astronomie. L'objectif est de prédire la classe d'un objet céleste (étoile, galaxie, quasar) à partir de ses caractéristiques spectroscopiques. Le dataset utilisé provient du Sloan Digital Sky Survey (SDSS), un projet d'observation astronomique lancé en 2000. Le SDSS a permis de collecter des données sur des millions d'objets célestes, ce qui en fait une bonne base pour ce projet. En utilisant le dataset disponible sur Kaggle [Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17), nous cherchons à construire un modèle de classification robuste, capable de distinguer entre trois types d'objets célestes :

- `STAR` : désigne une étoile, un corps céleste composé principalement d'hydrogène et d'hélium, qui produit de la lumière par fusion nucléaire. Les étoiles sont souvent proches (à l’échelle cosmique) et nombreuses dans le ciel.
  
- `GALAXY` : fait référence à une galaxie, un ensemble gigantesque de milliards d’étoiles, de gaz, de poussières et de matière noire. Ce sont des structures lointaines, bien plus grandes que les étoiles individuelles.

- `QSO` *(Quasi-Stellar Object)* : ce sont les quasars, des objets extrêmement lumineux situés au centre de certaines galaxies, alimentés par un trou noir supermassif. Ils ressemblent à des étoiles sur les images mais ont un comportement énergétique très différent.

### Description des variables explicatives

| Colonne                 | Description |
|-------------------------|-------------|
| `obj_ID`                | Identifiant unique de l'objet |
| `alpha`                 | Ascension droite - coordonnée céleste équivalente à la longitude |
| `delta`                 | Déclinaison - équivalent de la latitude céleste |
| `u`, `g`, `r`, `i`, `z` | Magnitudes apparentes - intensité lumineuse mesurée dans différentes bandes de longueurs d'onde (5) |
| `run_ID`                | Identifiant de la session d'observation |
| `rerun_ID`              | Version du traitement de données |
| `cam_col`               | Couleur du canal utilisé pour l'observation |
| `field_ID`              | Identifiant du champ du ciel observé |
| `spec_obj_ID`           | Identifiant de l’objet en spectroscopie |
| `redshift`              | Déplacement vers le rouge, indicateur de distance ou de vitesse |
| `plate`, `MJD`, `fiber_ID` | Détails techniques de l'observation spectroscopique (3) |


## Plan 
1. Ce que nous avons fait
    - Analyse exploratoire des données (EDA)
    - Prétraitement des données
    - Estimation des modèles et évaluation des performances
    - Choix du meilleur modèle
    - Conclusion
2. Ce qui a fonctionné
3. Ce qui n'a pas fonctionné
4. Résultats
5. Conclusion
--- 

## 1. Ce que nous avons fait

**- Analyse exploratoire des données (EDA)**
  
L'analyse exploratoire a débuté par une prise en main du jeu de données : types de variables, valeurs manquantes (aucune détectée), et structure générale. Nous avons observé environ 15 % d’outliers, que nous avons volontairement conservés, car dans le contexte astronomique, ces points extrêmes peuvent représenter des objets rares scientifiquement pertinents, comme des quasars ou des galaxies atypiques.

Nous avons ensuite mené une analyse univariée à l’aide d’histogrammes et de boxplots pour examiner la distribution des variables, ainsi qu’une analyse bivariée à l’aide de scatter plots et d’une matrice de corrélation, afin de comprendre les relations entre variables et la cible.

**- Prétraitement des données**

Le dataset présentait un fort déséquilibre entre les classes. Pour y remédier, nous avons utilisé la technique SMOTE (Synthetic Minority Over-sampling Technique), qui permet de générer des observations synthétiques pour les classes sous-représentées, assurant ainsi un apprentissage moins biaisé.

Nous avons ensuite normalisé les données à l’aide du StandardScaler afin de mettre toutes les variables sur la même échelle, ce qui est indispensable pour les modèles comme le SVM. Une analyse en composantes principales (PCA) a été appliquée sur les données normalisées, permettant de réduire la dimensionnalité tout en visualisant la séparation potentielle entre classes dans un espace 2D.

Enfin, nous avons séparé le jeu de données en deux parties : 

- 80% pour l’entraînement,
- 20% pour le test.

La séparation a été stratifiée (stratify=y) pour conserver les proportions des classes et réplicable via un random_state.

**- Estimation des modèles et évaluation des performances**

Quatre modèles ont été entraînés : 

- SVM
- Random Forest
- XGBoost
- Régression Logistique

Une première série d’entraînements a été réalisée sans réglage d’hyperparamètres pour établir une base comparative. Ensuite, nous avons utilisé GridSearchCV pour affiner les modèles en ajustant leurs hyperparamètres via validation croisée.

Afin d’optimiser la performance tout en réduisant la complexité, une sélection de variables a été effectuée à l’aide de SelectKBest. Les modèles ont ensuite été réentraînés sur les sous-ensembles de variables retenues, avec ou sans optimisation d’hyperparamètres, afin de mesurer les gains en précision, temps de calcul, et généralisation.

**- Choix du meilleur modèle**

Après avoir entraîné plusieurs modèles de classification, avec et sans paramétrage, nous avions comparé leurs performances sur la base de différents indicateurs : l’accuracy (précision globale des prédictions), le F1-score macro (utile pour évaluer la performance sur des classes déséquilibrées), la précision par classe, ainsi que l’analyse de la matrice de confusion, qui permet d’identifier les types d’erreurs les plus fréquents. 

Ces critères nous ont permis d’évaluer la performance générale et la capacité des modèles à bien traiter les classes minoritaires.

Sur la base du modèle sélectionné, nous avons mis en place une analyse approfondie de l'explicabilité des prédictions individuelles. Pour ce faire, nous avons employé plusieurs techniques d'interprétation, à savoir les méthodes ICE (Individual Conditional Expectation), LIME (Local Interpretable Model-agnostic Explanations) et SHAP (SHapley Additive exPlanations)

## 2. Ce qui a fonctionné

Dans ce projet, plusieurs approches ont été testées pour optimiser la performance des modèles de classification des objets célestes. Ce qui a fonctionné inclut :

- L'utilisation de modèles d'ensemble :

Les modèles comme Random Forest et XGBoost ont montré une robustesse naturelle face aux déséquilibres de classes et aux variables peu informatives, offrant de bonnes performances dès le départ. Ces modèles ont été capables de gérer efficacement les outliers présents dans les données astronomiques, comme les quasars, tout en maintenant une performance élevée.

- Le tuning des hyperparamètres :
  
L'optimisation des hyperparamètres via GridSearchCV a permis d'améliorer les performances des modèles, notamment pour des modèles comme le SVM et le Random Forest. Cette étape a révélé l'importance du réglage des paramètres dans l'amélioration des résultats, en particulier pour des modèles sensibles à la configuration par défaut comme le notre.

## 3. Ce qui n'a pas fonctionné

- Sélection de variables : 

Bien que la réduction de la dimensionnalité via SelectKBest ait permis de simplifier certains modèles, cette approche n'a pas apporté d'amélioration notable des performances globales. Dans certains cas, comme pour la régression logistique, cette réduction a même entraîné une légère baisse de la précision. Il aurait peut être fallu davantage réduire le nombre de variables sélectionnées.

- Impact limité du tuning sur certains modèles : 

Bien que le tuning des hyperparamètres ait amélioré les performances de plusieurs modèles, il a eu un impact particulièrement limité sur certains d'entre eux, en particulier pour des modèles simples comme la régression logistique. Dans ce cas, même après ajustement des hyperparamètres, la performance n’a pas montré une grande différence par rapport à la configuration par défaut. Elle a atteint une accuracy de 95.93 % et un score F1 pondéré de 0.96, ce qui reste très satisfaisant, mais légèrement inférieur aux performances de la forêt aléatoire. Cela montre que, malgré sa simplicité, la régression logistique peut bien fonctionner sur ce type de données, mais que son amélioration via le tuning reste marginale comparée à des modèles plus flexibles.

## 4. Résultats 

Au terme de cette étude comparative sur des données astronomiques classifiées en trois catégories (GALAXY, STAR, QSO), plusieurs modèles de machine learning ont été testés selon différentes configurations et ont donnés les résultats suivants en termes d'accuracy:

| Modèle                | Sans tuning, sans sélection | Tuning seul | Sélection seule | Tuning + Sélection |
|----------------------|-----------------------------|------------------|-----------------------|--------------------|
| SVM                  | 96.62%                      | 96.80%           | 96.80%                | 97.41%         |
| Random Forest        | **98.43%**                  | **98.54%**       | **98.38%**               | 97.88%             |
| XGBoost              | 98.26%                      | 98.25%           | 98.20%                | **98.02%**             |
| Régression Logistique| 96.06%                      | 96.48%           | 95.96%                | 96.48%             |

Les modèles Random Forest et XGBoost ont montré des performances exceptionnelles dès le départ, avec des accuracy autour de 98%. En revanche, la régression logistique et le SVM ont obtenu des scores autour de 96%, soulignant leur sensibilité aux configurations par défaut. Le tuning des hyperparamètres (via GridSearchCV) a amélioré les performances de tous les modèles, en particulier du SVM et du Random Forest. La sélection de variables a également été bénéfique pour certains modèles, mais a légèrement réduit la performance de la régression logistique.

La meilleure configuration pour chaque modèle a combiné tuning et sélection de variables, avec des performances optimales pour Random Forest (98.54%), XGBoost (98.02%), et SVM (97.41%). La régression logistique a atteint un plafond à 96.48%, confirmant ses limites dans des scénarios plus complexes.

La meilleure performance a été obtenue par Random Forest avec un score de 98.54 % après tuning des hyperparamètres, seul sur le jeu de données complet. Les meilleurs paramètres trouvés sont : 

- <mark>bootstrap</mark> : `False`
- <mark>max_depth</mark> : `None`
- <mark>min_samples_leaf</mark> : `1`
- <mark>min_samples_split</mark> : `2`
- <mark>n_estimators</mark> : `100`
  
Cela signifie que chaque arbre de la forêt est construit sans échantillonnage avec remplacement, ce qui permet d'éviter une trop grande similarité entre les arbres. En réglant max_depth=None, cela permet aux arbres de se développer autant qu'ils en ont besoin, ce qui peut être utile pour capturer toute la complexité des données, même si parfois ça risque de mener à du surapprentissage. 
Le paramètre min_samples_leaf=1 signifie qu'il n'y a pas de minimum d'échantillons dans les feuilles, donc les arbres peuvent être très spécifiques. min_samples_split=2 indique que les arbres peuvent se diviser même si seulement deux échantillons sont présents, permettant ainsi une segmentation très fine. Enfin, avec n_estimators=100, on a 100 arbres dans la forêt, ce qui est un bon compromis entre performance et temps de calcul. 

En ce qui concerne l'explicabilité des prédictions individuelles, nous relevons les résultats suivants : 

- Les ICE plots montrent comment des variables astrophysiques influencent les prédictions du modèle Random Forest tuné. Le `redshift` a une influence majeure, ce qui est cohérent puisqu’il différencie fortement les galaxies et quasars des étoiles dans ce type de données. Les magnitudes `z` et `g` ont un effet plus progressif, indiquant une sensibilité plus modérée du modèle à ces variables photométriques.
  
- LIME montre que la prédiction locale, à savoir, que la probabilité d’être une GALAXY à 46.7% est surtout due à un redshift modéré, avec un léger renfort des magnitudes photométriques. Le redshift reste alors la variable la plus influente localement.

- L'explication SHAP montre que ces coordonnées célestes et l’identifiant de l’objet ont un impact très limité sur la prédiction. Cela suggère que le modèle ne s’appuie pas fortement sur la position dans le ciel ou l’ID pour classer un objet comme GALAXY, STAR ou QSO, mais plutôt sur d’autres variables comme le `redshift` ou les magnitudes photométriques.

## 5. Conclusion 

Les modèles ont démontré une grande efficacité dans la classification des objets célestes, notamment grâce à leur capacité à traiter les données spectroscopiques et photométriques complexes. Leur aptitude à gérer des "outliers" tels que les quasars est cruciale, car ces objets rares peuvent être atypiques tout en étant scientifiquement pertinents. L'utilisation de modèles robustes, capables de s'adapter à ce type de données, est donc indispensable pour garantir des performances fiables.

En conclusion, bien que la sélection de variables n'ait pas entraîné d'améliorations significatives dans ce cas précis, l'ensemble des modèles testés a démontré une grande capacité d'adaptation aux spécificités des données astronomiques. Leur aptitude à gérer des données complexes, incluant des objets atypiques comme les quasars, confirme leur robustesse et leur pertinence scientifique. Les analyses d'explicabilité via ICE, LIME et SHAP ont permis d’identifier les variables clés, notamment le redshift et les magnitudes photométriques, comme déterminantes dans la classification, tandis que des variables comme les coordonnées célestes ou l’ID ont montré un impact négligeable. Cela renforce la confiance dans le fonctionnement interne du modèle et sa capacité à fonder ses décisions sur des caractéristiques astrophysiquement pertinentes. Ces modèles sont prêts à être déployés pour des applications concrètes dans la classification des objets célestes, avec des performances de haut niveau, et peuvent être davantage optimisés en fonction des caractéristiques des données d'autres études astronomiques.

---
DAHMANI Amel et ONNO Lilou
