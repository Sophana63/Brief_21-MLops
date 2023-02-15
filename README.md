# Brief 21 - MLOPs quand tu nous tiens

## Contexte du projet

Elle n'est cependant pas très mature sur les outils dits "MLOps", qui permettent entre autre, de monitorer les résultats des algorithmes qu'elle conçoit. Algorithmes qui aujourd'hui, peuvent lui poser des problèmes lorsque l'entrainement dégrade les performances de ces derniers (pour diverses raisons).

Elle vous demande de lui présenter l'outil MLFlow à l'aide d'un projet de prédiction de prix. De lui montrer les principales fonctionnalités :

Création d'un fichier MLProject Versionning avec Git Gestion des paramètres entre les runs Exécuter sur un fichier python et/ou un notebook.

## Logiciels et librairies utilisés

Les briefs seront codés avec le langage Python avec l'outil de travail Jupyter Notebook. Certains devront être ouverts avec l'IDE VS code.

Plusieurs **librairies de base** vont être utilisées : pandas, numpy, matplotlib, seaborn et sklearn. Mais pour ce brief, les librairies le plus utilisée seront sklearn et mlflow, souvent pour pouvoir importer plusieurs types de modèles, pipelines, encoders, ect...mais aussi de les visualiser.

```
pip install pandas
pip install numpy
pip install -U scikit-learn
pip install mlflow
```

```
## exemple d'importation de librairies sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

```

Pour utiliser Azure

```
pip install azure-cognitiveservices-vision-computervision
pip install azure-cognitiveservices-vision-face
```

## Partie 1

Pour répondre au brief, J'utilise le dataset des iris pour les prédictions et ensuite les afficher dans mlflow.  
Voici le lien vers le notebook : [tp_mlflow.ipynb](https://github.com/Sophana63/Brief_21-MLops/blob/master/Part_1/tp_mlflow.ipynb)


## Partie 2

### ``Exercice 1``
L'objectif est de prédire le montant du tarif (y compris les péages) pour un trajet en taxi à New York en fonction des lieux de prise en charge et de dépose.

Pas de problème particulier dans cet exercice. Voici le lien du notebook: [ML-Workflow.ipynb](https://github.com/Sophana63/Brief_21-MLops/blob/master/Part_2/01-Kaggle-Taxi-Fare/ML-Workflow.ipynb)

    
### ``Exercice 2  ``  

Refactoriser le problème de prédiction des tarifs de taxi **à l'aide de pipeline**.

De même de le premier exercice, pas de problème particulier dans cet exercice. Voici le lien du notebook: [Taxifare-Pipelines.ipynb](https://github.com/Sophana63/Brief_21-MLops/blob/master/Part_2/02-Taxi-Fare-Pipeline/Taxifare-Pipelines.ipynb)


### ``Exercice 3``

Dans ce défi, nous n'implémenterons pas de nouvelles fonctionnalités. Nous allons réorganiser le code existant de l'exercice 2 en packages et modules.

Nous allons nous concentrer sur la création d'une classe que nous nommerons **Trainer**. Dans celle-ci, il faudra répondre à plusieurs objectifs:

Le `__int__()` comprend:
- toutes les pipelines
- le modèle `LinearRegression()`

La méthode `get_clean_data_train` permettra:
- lire un fichier CSV et le transformer en dataframe
- nettoyer les données 
- les découper avec `train_test_split`

Les deux autres méthodes permettent d'entrainer et de donner le RMSE.

Pour tester l'exercice, il faudra lancer le fichier `test.py` avec VS code. [Lien du dossier](https://github.com/Sophana63/Brief_21-MLops/tree/master/Part_2/03-Notebook-to-package)


### ``Exercice 4``

Pour cet exercice, je vais récupérer ma classe créée auparavant et implémenter des méthodes pour lancer et enregeristrer des expérimantations sur Ml-Flow.

Pour tester l'exercice, il faudra lancer le fichier `ml_flow_test2.py` avec VS code. [Lien du dossier](https://github.com/Sophana63/Brief_21-MLops/tree/master/Part_2/04-MLFlow-quickstart)


### ``Exercice 5``

Je vais changer un peu de code pour utiliser plusieurs modèles dans mes pipelines, créer des Cross Validation et sortir les RMSE.

``` python
# modeles utilisés
self.estimators = [
            ("Linear Regression", LinearRegression()),
            ("Ridge Regression", Ridge())
            ("Decision Tree", DecisionTreeRegressor())
            ("Random Forest", RandomForestRegressor())
            ]

# méthode pour évaluer chaque modele
def evaluate_estimators(self, X_train, y_train, X_test, y_test, X, y):
        results = { 
            "name" : [],
            "score" : [],
            "RMSE" : [],
            "MSE" : [],
            "MAE" : []
        }
        for name, estimator in self.estimators:
            self.pipe_estimators = Pipeline([
                ('preprocessor', self.preproc_pipe),
                (name, estimator)
            ])            
            self.pipe_estimators.fit(X_train, y_train)
            cv_score_mse = cross_val_score(self.pipe_estimators, X, y, scoring="neg_mean_squared_error", cv=5)
            cv_score_mae = cross_val_score(self.pipe_estimators, X, y, scoring="neg_mean_absolute_error", cv=5)
            y_pred = self.pipe_estimators.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            score = self.pipe_estimators.score(X_test, y_test)
            self.scores.append(score)

            mse = np.mean(np.abs(cv_score_mse))
            mae = np.mean(np.abs(cv_score_mae))
            print(f"{name} - Score: {score} - RMSE: {rmse} - Mean MSE: {mse} - Mean MAE: {mae}")

            results["name"].append(name)
            results["score"].append(score)
            results["RMSE"].append(rmse)
            results["MSE"].append(mse)
            results["MAE"].append(mae)

        return results
```

Pour ce brief, je me suis arrêté ici. Le code fonctionne bien, mais j'ai une erreur dans ma boucle quand je veux les enregistrer dans mlflow. Ca me sauvegarde que la première expérimentation.

Pour tester l'exercice, il faudra lancer le fichier `ml_flow_test2.py` avec VS code. [Lien du dossier](https://github.com/Sophana63/Brief_21-MLops/tree/master/Part_2/05-Iterate-with-MLFlow)
