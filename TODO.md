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
    => ok mais pas pour toute les métyhodes
    ✅ gerer WiGS SAC
  - Revoir la cross validation avec le init de datamodule

    ```
            (
                self.train_x,
                self.train_y_reg,
                self.train_y_time_reg,
                self.train_y_cls,
                self.train_y_time_cls,
                self.val_x,
                self.val_y_reg,
                self.val_y_time_reg,
                self.val_y_cls,
                self.val_y_time_cls,
            ) = self._split(
                df_x, df_y_reg, df_y_time_reg, df_y_cls, df_y_time_cls, k=k_fold, k_index=k_index
            )
    ```


- 2 :
  intégration hydra lightning ✅

- 3 :
  multivariate ✅ 

  -4 : 
    - ❌ augmenter le batch pour la prediction (et pour le fit?) au max de la capa du GPU 
    - ✅ Faire les post sur la test error  
    - ✅ réinit le model à chaque fit()

  -5 : 
    - merge Romain
    - en // embarquer sur JZ 