import encoders

data_link = "../../data/train.csv"

# création de mon objet
df = encoders.Trainer(data_link)

# méthode permetant de :
# ouvrir un csv et le transformer en dataframe
# clean les datas et supprime les outliers
# train_test_split des données nettoyées
X_train, X_val, y_train, y_val = df.get_clean_data_train()

# ma classe va créer des pipeline -> self.pipe
# il faudra juste entrer les paramètres X_train et y_train dans la méthode train() de ma classe 
# idem pour la méthode evaluate()
train = df.train(X_train, y_train)
rmse = df.evaluate(X_val, y_val)

# start MLFlow
df.mlflow_log_metric("rmse", rmse)
df.mlflow_log_param("model", "Pipeline")
df.mlflow_log_param("student_name", "phana")

# Afficher un lien sur mlFlow avec le numéro de l'expérimentation
experiment_id = df.mlflow_experiment_id
print(f"experiment URL: http://127.0.0.1:5000/#/experiments/{experiment_id}")
