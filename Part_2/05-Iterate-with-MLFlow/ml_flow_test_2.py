import encoders
import pandas as pd

if __name__ == "__main__":

    data_link = "../data/train.csv"

    # création de mon objet
    df = encoders.Trainer(data_link)

    # méthode permetant de :
    # ouvrir un csv et le transformer en dataframe
    # clean les datas et supprime les outliers
    # train_test_split des données nettoyées
    X_train, X_val, y_train, y_val, X, y = df.get_clean_data_train()

    # ma classe va créer des pipeline -> self.pipe
    # il faudra juste entrer les paramètres X_train et y_train dans la méthode train() de ma classe 
    # idem pour la méthode evaluate()
    # train = df.train(X_train, y_train)
    # rmse = df.evaluate(X_val, y_val)

    # trainer = train
    # trainer = trainer.save_model(trainer)  

    dict = df.evaluate_estimators(X_train, y_train, X_val, y_val, X, y)
    df_results = pd.DataFrame.from_dict(dict)
    print(df_results)

    #start MLFlow
    # df.mlflow_log_metric("rmse", rmse)
    # df.mlflow_log_param("model", "Pipeline")
    # df.mlflow_log_param("student_name", "phana")

    for index, row in df_results.iterrows():
        name = row['name']
        score = row['score']
        rmse = row['RMSE']
        mse = row['MSE']
        mae = row['MAE']
        print(name)
    
        #df.mlflow_run()
        df.mlflow_log_param("model", name)
        df.mlflow_log_param("student_name", "phana")
        df.mlflow_log_metric("score", score)
        df.mlflow_log_metric("rmse", rmse)
        df.mlflow_log_metric("mse", mse)
        df.mlflow_log_metric("mae", mae)
    
    # Afficher un lien sur mlFlow avec le numéro de l'expérimentation
    experiment_id = df.mlflow_experiment_id
    print(f"experiment URL: http://127.0.0.1:5000/#/experiments/{experiment_id}")   
    print("-----------------------------------------")