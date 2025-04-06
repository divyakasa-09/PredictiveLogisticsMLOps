import pandas as pd
from sklearn.metrics import f1_score, recall_score, confusion_matrix, roc_auc_score

# Evaluation function
# Columns needed to compare metrics
comparison_columns = ['Model_Name', 'Train_F1score', 'Train_Recall', 'Valid_F1score', 'Valid_Recall', 'Test_F1score', 'Test_Recall']

comparison_df = pd.DataFrame()

def evaluate_models(model_name, model_defined_var, X_train, y_train, X_valid, y_valid, X_test, y_test):
    '''
    This function predicts and evaluates various models for classification
    
    Parameters:
    - model_name (str): Name of the model for identification in the results.
    - model_defined_var: Model instance with defined parameters.
    - X_train, X_valid, X_test: Feature sets for training, validation, and testing.
    - y_train, y_valid, y_test: Target labels for training, validation, and testing.
    
    Returns:
    - final_dict (dict): Dictionary containing evaluation metrics.
    '''

    # Train predictions
    y_train_pred = model_defined_var.predict(X_train)
    # Train performance
    train_f1_score = f1_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)

    # Validation predictions
    y_valid_pred = model_defined_var.predict(X_valid)
    # Validation performance
    valid_f1_score = f1_score(y_valid, y_valid_pred)
    valid_recall = recall_score(y_valid, y_valid_pred)

    # Test predictions
    y_pred = model_defined_var.predict(X_test)
    # Test performance
    test_f1_score = f1_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)

    # Printing performance
    print("Train Results")
    print(f'F1 Score: {train_f1_score}')
    print(f'Recall Score: {train_recall}')
    print(f'Confusion Matrix: \n{confusion_matrix(y_train, y_train_pred)}')
    print(f'Area Under Curve: {roc_auc_score(y_train, y_train_pred)}')

    print(" ")

    print("Validation Results")
    print(f'F1 Score: {valid_f1_score}')
    print(f'Recall Score: {valid_recall}')
    print(f'Confusion Matrix: \n{confusion_matrix(y_valid, y_valid_pred)}')
    print(f'Area Under Curve: {roc_auc_score(y_valid, y_valid_pred)}')

    print(" ")

    print("Test Results")
    print(f'F1 Score: {test_f1_score}')
    print(f'Recall Score: {test_recall}')
    print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
    print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')

    # Saving results
    global comparison_columns
    metric_scores = [model_name, train_f1_score, train_recall, valid_f1_score, valid_recall, test_f1_score, test_recall]
    final_dict = dict(zip(comparison_columns, metric_scores))
    return final_dict

final_list = []

def add_dic_to_final_df(final_dict):
    '''
    Adds a dictionary of evaluation metrics to the final comparison dataframe.
    
    Parameters:
    - final_dict (dict): Dictionary containing evaluation metrics.
    '''
    global final_list
    final_list.append(final_dict)
    global comparison_df
    comparison_df = pd.DataFrame(final_list, columns=comparison_columns)
