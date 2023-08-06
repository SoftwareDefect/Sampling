import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef

def tuneparameters(trn_X,trn_y,tst_X,tst_y):
    # 1. Define the Objective Function
    def objective(trial):
        # Define the hyperparameters and their ranges
        c   = trial.suggest_float('C', 1e-4, 1000, log=True)
        tol = trial.suggest_float('tol', 1e-5, 1.0, log=True)
        solver = trial.suggest_categorical('solver', ['liblinear'])
        penalty = trial.suggest_categorical('penalty', ['l1','l2']) #

        # Create and train the logistic regression model ,
        model = LogisticRegression( C=c, tol=tol, solver=solver,penalty=penalty)#

        # Evaluate the model
        model.fit(trn_X, trn_y)
        tst_pred = model.predict(tst_X)
        mcc = matthews_corrcoef(tst_y, tst_pred)
        # Return the evaluation metric
        return mcc


    # 2. Create a Study Object
    study = optuna.create_study(direction='maximize')

    # 3. Run the Optimization Process
    study.optimize(objective, n_trials=100)

    #return best para model
    best_params = study.best_params
    best_model = LogisticRegression(**best_params)
    return best_model




