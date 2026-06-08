- 0 : 
  - remplacer le chargement de toutes les données par seulement selui des X. 
    Et n'utiliser le Y true seulement quand c'est nécessaire à chaque trouvaille des candidats.  
    => vérifier résultats avec Results_init     :  ✅
  - Utiliser safetensor plutôt que pdframe      :  ❌
  - faire par batch (top k) plutôt que par indiv:  ✅

- 1 :
  - tcheck CV et MLP
    - il faut retrain à chaque cross val l'ensemble des fold. 
    - => c'est surement à adapter  
    - tcheck avec autre repo pour le MLP
  - calcule avec MLP et check les résultats  
  - ❌ : multi-output , tchek remplacement de  `get_features_and_target(df_pool, "Y")` par 
    `get_features_and_target(df_pool, y_size)`
    gerer WiGS SAC


- 2 :
  intégration hydra lightning

- 3 :
  multivariate