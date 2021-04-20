#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os.path

from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from preprocessing.preprocessing import preprocess_whole_pipeline



def load_model(path_model):
    bst = xgb.Booster({'nthread': 8})  # init model
    bst.load_model(path_model)  # load data
    return bst



def get_prediction(model, data_test, ID, label, output_path):    
    dtest = xgb.DMatrix(data_test)
    ypred = [max(0,round(value,2)) for value in list(model.predict(dtest))]
    
    res = pd.DataFrame({"drug_id":ID, label:ypred},columns=["drug_id",label])
    if output_path != "print":
        res.to_csv(path_or_buf=output_path, index=False, float_format='%.2f')
        print("Prediction DONE, results available here: " + output_path)
    else:
        print(res)
    
    
    
def create_model(data_train,label,max_depth,eta,num_round,path_model,NaN_imputation_feature_scaling_PCA_boolean,directory_of_script):
    '''
    Creation of the model using XGBoost
    '''
    print("Training model using: XGBoost")
    df_train, df_test = train_test_split(data_train, test_size=0.2)    
    dtrain = xgb.DMatrix(df_train.drop(label,axis=1), label=df_train[label])
    dtest = xgb.DMatrix(df_test.drop(label,axis=1), label=df_test[label])
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    evals_result = {}    
    param = {'max_depth': max_depth, 'eta': eta, 'objective': 'reg:squarederror'}
    param['nthread'] = 8
    param['eval_metric'] = 'rmse'
    
    bst = xgb.train(param, 
                    dtrain, 
                    num_round, 
                    evallist, 
                    early_stopping_rounds=10, 
                    evals_result=evals_result)
    
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    param_string = 'max_depth_' + str(param['max_depth']) + "_eta_" + str(param['eta']) + "_num_round_" + str(num_round) + "_NaN_imputation_feature_scaling_PCA_usage_" + str(NaN_imputation_feature_scaling_PCA_boolean)
    model_name = param_string + "_" + dt_string
    bst.save_model(path_model + "_" + model_name)
    print("Model is available here: " + path_model + "_" + model_name)
    
    '''
    Get the XGBoost model results and information
    '''       
    print("Plotting validation curve")
    x_axis = range(len(evals_result['train']['rmse']))    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_axis, evals_result['train']['rmse'], label='Train')
    ax.plot(x_axis, evals_result['eval']['rmse'], label='Test')
    ax.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Number of Rounds')
    plt.title('XGBoost RMSE')
    plt.savefig(os.path.join(directory_of_script,"results","Validation Curve" + "_" + model_name + ".png"))
    print("Learning Curve is available here: " + os.path.join(directory_of_script,"results","Validation Curve" + "_" + model_name + ".png"))       
    
    ypred = bst.predict(dtest)    
    RMSE = mean_squared_error(df_test[label], ypred, squared=False)
    print("RMSE: %.4f" % RMSE)
            
    print("Check importance of features\n")
    fig, ax = plt.subplots(figsize=(100, 100))
    ax = xgb.plot_importance(bst,ax=ax)
    ax.figure.savefig(os.path.join(directory_of_script,"results","Feature Importance" + "_" + model_name + ".png"))
    print("Features Importance is available here: " + os.path.join(directory_of_script,"results","Feature Importance" + "_" + model_name + ".png"))
    print("Training DONE")    





def main(args):
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    path_of_output_prediction = args.path_of_output_prediction
    data_directory = args.data_directory
    path_model = args.path_model
    mode = args.mode
    NaN_imputation_feature_scaling_PCA_usage = args.NaN_imputation_feature_scaling_PCA_usage
    max_depth = args.max_depth
    eta = args.eta
    num_round = args.num_round   
    
    FILENAME_DRUGS_TRAIN = "drugs_train.csv"
    FILENAME_DRUGS_TEST = "drugs_test.csv"
    FILENAME_FEATURE_ENG = "drug_label_feature_eng.csv"
    FILENAME_ACTIVE_INGREDIENTS = "active_ingredients.csv"
    
    FEATURE_TO_DUMMY = ["administrative_status",
                        "marketing_status",
                        "approved_for_hospital_use",
                        "dosage_form",
                        "marketing_authorization_status",
                        "marketing_authorization_process",
                        "pharmaceutical_companies"]
    LABEL = "price"
    
    path_drugs_train = os.path.join(data_directory,FILENAME_DRUGS_TRAIN)
    path_feature_eng = os.path.join(data_directory,FILENAME_FEATURE_ENG)
    path_active_ingredients = os.path.join(data_directory,FILENAME_ACTIVE_INGREDIENTS)
    path_drugs_test = args.path_of_data_to_predict
    if path_drugs_test is None : path_drugs_test = os.path.join(data_directory,FILENAME_DRUGS_TEST)

    
    '''
    Reading files
    '''
    print("Reading files...")
    df_drugs_train = pd.read_csv(path_drugs_train, sep=",")
    df_drugs_test = pd.read_csv(path_drugs_test, sep=",")    
    df_feature_eng = pd.read_csv(path_feature_eng, sep=",")
    df_active_ingredients = pd.read_csv(path_active_ingredients, sep=",")

  
    '''
    Preprocessing variables from train and test data, feature engineering 
    and active ingredients data
    '''
    if mode == "prediction" : NaN_imputation_feature_scaling_PCA_usage = "True" in path_model
    data, data_train, data_test, drug_id = preprocess_whole_pipeline(df_active_ingredients=df_active_ingredients,
                                                                      df_feature_eng=df_feature_eng,
                                                                      df_drugs_train=df_drugs_train,
                                                                      df_drugs_test=df_drugs_test,
                                                                      feature_to_dummy=FEATURE_TO_DUMMY,
                                                                      NaN_imputation_feature_scaling_PCA_boolean=NaN_imputation_feature_scaling_PCA_usage,
                                                                      label=LABEL)

    
    #Get again train and test data using preprocessed data (data = data_train + data_test=
    data_train = data[0:data_train.shape[0]]
    data_test = data[data_train.shape[0]:data.shape[0]]
    
    #Remove added price feature (contains only NaN values) from test data
    data_test.drop('price', axis=1, inplace=True)
    
      
    if mode == "training":
        '''
        Creation of the model using XGBoost
        '''
        create_model(data_train=data_train,
                     label=LABEL,
                     max_depth=max_depth,
                     eta=eta,
                     num_round=num_round,
                     path_model=path_model,
                     NaN_imputation_feature_scaling_PCA_boolean=NaN_imputation_feature_scaling_PCA_usage,
                     directory_of_script=directory_of_script)
        
    
    if mode == "prediction":
        '''
        Make prediction of test file using trained model
        ''' 
        bst = load_model(path_model)
        get_prediction(model=bst, 
                       data_test=data_test, 
                       ID=list(drug_id[data_train.shape[0]:data.shape[0]]),
                       label=LABEL,
                       output_path=path_of_output_prediction)
    
    
if __name__ == "__main__":
    
    MODEL_NAME = "XGBoost_drug_price.model"
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_model = os.path.join(directory_of_script,"model")
    directory_of_results = os.path.join(directory_of_script,"results")
    directory_of_data = os.path.join(directory_of_script,"data")
    os.makedirs(directory_of_model,exist_ok=True)
    os.makedirs(directory_of_results,exist_ok=True)
    os.makedirs(directory_of_data,exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_directory", help="Directory of location of the data for training", required=False,  default=directory_of_data, nargs='?')
    parser.add_argument("-mode", help="Mode to use: training to create a model (train and save it and check error rate, learning curve and features importances), or prediction to make prediction using a model", required=True, choices=["training","prediction"])  
    parser.add_argument("-path_model", help="Path of an existing model to use for prediction", required=False, default=os.path.join(directory_of_model,MODEL_NAME), nargs='?')
    parser.add_argument("-path_of_data_to_predict", help="Path of data file to make prediction", required=False)
    parser.add_argument("-path_of_output_prediction", help="Path of output file containing the prediction of data to predict - Use print for a print of results", required=False, default=os.path.join(directory_of_results,"submission.csv"), nargs='?')
    parser.add_argument("-NaN_imputation_feature_scaling_PCA_usage", help="Apply or not NaN imputation, Feature Scaling and PCA", required=False, choices=["False","True"], default="False", nargs='?')
    parser.add_argument("-max_depth", help="Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as hist or gpu_hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree. range: [0,∞] (0 is only accepted in lossguided growing policy when tree_method is set as hist or gpu_hist)", required=False, default=6, type=int, nargs='?')
    parser.add_argument("-eta", help="Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative. range: [0,1]", required=False, default=0.3, type=float, nargs='?')
    parser.add_argument("-num_round", help="The number of rounds for boosting", required=False, default=100, type=int, nargs='?')
    args = parser.parse_args()
    
    main(args)
     