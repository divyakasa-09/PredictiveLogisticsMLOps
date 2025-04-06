import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import xgboost as xgb
from sklearn.metrics import f1_score
import logging
from .utils import setup_logging

# Setup logging
logger = setup_logging()


def train_logistic_model(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, project_config):
    '''
    Train a Logistic Regression model and log performance metrics using Weights & Biases.

    Parameters:
    - X_train, X_valid, X_test: Feature sets for training, validation, and testing.
    - y_train, y_valid, y_test: Target labels for training, validation, and testing.

    Returns:
    - None
    '''
    # Get the column names of features
    features = X_train.columns
    PROJECT_NAME = project_config['wandb']['wandb_project']
    # Initialize Weights & Biases run
    with wandb.init(project=PROJECT_NAME) as run:
        config = wandb.config

        # Define model parameters
        params = {
            "random_state": 13,
            "class_weight": class_weights 
            }

        # Initialize Logistic Regression model
        model = LogisticRegression(**params)

        # Train the model
        model.fit(X_train, y_train)

        # Train predictions and performance
        y_train_pred = model.predict(X_train)
        train_f1_score = f1_score(y_train, y_train_pred)

        # Validation predictions and performance
        y_valid_pred = model.predict(X_valid)
        valid_f1_score = f1_score(y_valid, y_valid_pred)

        # Test predictions and performance
        y_preds = model.predict(X_test)
        y_probas = model.predict_proba(X_test)
        score = f1_score(y_test, y_preds)

        # Log performance metrics
        print(f"F1_score Train: {round(train_f1_score, 4)}")
        print(f"F1_score Valid: {round(valid_f1_score, 4)}")
        print(f"F1_score Test: {round(score, 4)}")

        wandb.log({"f1_score_train": train_f1_score})
        wandb.log({"f1_score_valid": valid_f1_score})
        wandb.log({"f1_score": score})

        # Plot classifier performance
        wandb.sklearn.plot_classifier(model, X_train, X_test, y_train, y_test,
                                            y_preds, y_probas, labels= None, model_name='LogisticRegression', feature_names=features)

        # Save the trained model
        model_artifact = wandb.Artifact("LogisticRegression", type="model", metadata=dict(config))
        joblib.dump(model, "models/log-truck-model.pkl")
        model_artifact.add_file("models/log-truck-model.pkl")
        wandb.save("models/log-truck-model.pkl")
        run.log_artifact(model_artifact)





def train_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, project_config):
    '''
    Train a Random Forest classifier and log performance metrics using Weights & Biases.

    Parameters:
    - X_train, X_valid, X_test: Feature sets for training, validation, and testing.
    - y_train, y_valid, y_test: Target labels for training, validation, and testing.

    Returns:
    - None
    '''
    features = X_train.columns
    labels = ["delay"]
    PROJECT_NAME = project_config['wandb']['wandb_project']

    # Initialize Weights & Biases run
    with wandb.init(project=PROJECT_NAME) as run:
        config = wandb.config

        # Define model parameters
        model = RandomForestClassifier(n_estimators=20, class_weight=class_weights, random_state=7)

        # Train the model
        model.fit(X_train, y_train)

        # Train predictions and performance
        y_train_pred = model.predict(X_train)
        train_f1_score = f1_score(y_train, y_train_pred)

        # Validation predictions and performance
        y_valid_pred = model.predict(X_valid)
        valid_f1_score = f1_score(y_valid, y_valid_pred)

        # Test predictions and performance
        y_preds = model.predict(X_test)
        y_probas = model.predict_proba(X_test)
        score = f1_score(y_test, y_preds)

        # Log performance metrics
        print(f"F1_score Train: {round(train_f1_score, 4)}")
        print(f"F1_score Valid: {round(valid_f1_score, 4)}")
        print(f"F1_score Test: {round(score, 4)}")

        wandb.log({"f1_score_train": train_f1_score})
        wandb.log({"f1_score_valid": valid_f1_score})
        wandb.log({"f1_score": score})

        # Plot classifier performance
        wandb.sklearn.plot_classifier(model, X_train, X_test, y_train, y_test, y_preds, y_probas, labels=None,
                                      model_name='RandomForestClassifier', feature_names=features)

        # Save the trained model
        model_artifact = wandb.Artifact("RandomForestClassifier", type="model", metadata=dict(config))
        joblib.dump(model, "models/randomf-truck-model.pkl")
        model_artifact.add_file("models/randomf-truck-model.pkl")
        wandb.save("models/randomf-truck-model.pkl")
        run.log_artifact(model_artifact)




def train_xgb_model(X_train, y_train, X_valid, y_valid, X_test, y_test, project_config):
    '''
    Train an XGBoost classifier and log performance metrics using Weights & Biases.

    Parameters:
    - X_train, X_valid, X_test: Feature sets for training, validation, and testing.
    - y_train, y_valid, y_test: Target labels for training, validation, and testing.

    Returns:
    - None
    '''
    PROJECT_NAME = project_config['wandb']['wandb_project']
    features = X_train.columns
    labels = ["delay"]

    # Initialize Weights & Biases run
    with wandb.init(project=PROJECT_NAME) as run:
        config = wandb.config

        # Convert training and test sets to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Train initial model
        params = {'objective': 'multi:softmax', 'num_class': 2}
        num_rounds = 30
        xgbmodel = xgb.train(params, dtrain, num_rounds, evals=[(dvalid, 'validation')],
                             early_stopping_rounds=10)

        # Train predictions and performance
        y_train_pred = xgbmodel.predict(dtrain)
        train_f1_score = f1_score(y_train, y_train_pred)

        # Validation predictions and performance
        y_valid_pred = xgbmodel.predict(dvalid)
        valid_f1_score = f1_score(y_valid, y_valid_pred)

        # Test predictions and performance
        y_preds = xgbmodel.predict(dtest)
        score = f1_score(y_test, y_preds)

        # Log performance metrics
        print(f"F1_score Train: {round(train_f1_score, 4)}")
        print(f"F1_score Valid: {round(valid_f1_score, 4)}")
        print(f"F1_score Test: {round(score, 4)}")

        wandb.log({"f1_score_train": train_f1_score})
        wandb.log({"f1_score_valid": valid_f1_score})
        wandb.log({"f1_score": score})

        # Save the trained model
        model_artifact = wandb.Artifact("XGBoost", type="model", metadata=dict(config))
        joblib.dump(xgbmodel, "models/xgb-truck-model.pkl")
        model_artifact.add_file("models/xgb-truck-model.pkl")
        wandb.save("models/xgb-truck-model.pkl")
        run.log_artifact(model_artifact)






def train_models(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, project_config):
    '''
    Train multiple machine learning models and log performance metrics.

    Parameters:
    - X_train, X_valid, X_test: Feature sets for training, validation, and testing.
    - y_train, y_valid, y_test: Target labels for training, validation, and testing.
    - project_config (dict): Configuration parameters for the project.

    Returns:
    - None
    '''
    try:
        # Train Logistic Regression model
        train_logistic_model(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, project_config)

        # Train Random Forest model
        train_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights, project_config)

        # Train XGBoost model
        train_xgb_model(X_train, y_train, X_valid, y_valid, X_test, y_test, project_config)

    except Exception as e:
        # Log the error
        logger.error(f"An error occurred while training the models: {str(e)}")

