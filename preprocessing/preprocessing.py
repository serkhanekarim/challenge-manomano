#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import re

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def extract_date(data):
    '''
    Create new feature from 'marketing_declaration_date' and 'marketing_authorization_date' features
    and remove 'marketing_declaration_date' and 'marketing_authorization_date' features
    
    Parameters
    ----------
    data : Pandas Dataframe
            
    Returns
    -------
    Inplace Dataframe : Pandas dataframe with new features (years, months, days) extracted 
                        from 'marketing_declaration_date' and 'marketing_authorization_date'
                        
    Examples
    --------
    >>> df_drugs_train = pd.read_csv(path_drugs_train, sep=",")
    >>> extract_date(data = df_drugs_train)
        0       20140101       2014   1   1
        1       20130101       2013   1   1
        2       20000101 ----> 2000   1   1
        3       20050101       2005   1   1
        4       20150101       2015   1   1
    '''
    
    data['marketing_declaration_date_year'] = data['marketing_declaration_date'].apply(lambda date : int(str(date)[0:4]))
    data['marketing_declaration_date_month'] = data['marketing_declaration_date'].apply(lambda date : int(str(date)[4:6]))
    data['marketing_declaration_date_day'] = data['marketing_declaration_date'].apply(lambda date : int(str(date)[6:8]))
    
    data['marketing_authorization_date_year'] = data['marketing_authorization_date'].apply(lambda date : int(str(date)[0:4]))
    data['marketing_authorization_date_month'] = data['marketing_authorization_date'].apply(lambda date : int(str(date)[4:6]))
    data['marketing_authorization_date_day'] = data['marketing_authorization_date'].apply(lambda date : int(str(date)[6:8]))

    data.drop('marketing_declaration_date', axis=1, inplace=True)
    data.drop('marketing_authorization_date', axis=1, inplace=True)


def convert_percent(data):
    
    '''
    Convert percent string into float representation
    
    Parameters
    ----------
    data : Pandas Dataframe
            
    Returns
    -------
    Inplace Dataframe : Pandas dataframe with the percent columns (reimbursement_rate
) converted into float representation
    
    Examples
    --------
    >>> df_drugs_train = pd.read_csv(path_drugs_train, sep=",")
    >>> convert_percent(data = df_drugs_train)
             drug_id  ...  reimbursement_rate       price 
    0        0_train  ...       0.65                2.83
    1        1_train  ...       0.65                14.30
    2        2_train  ...       0.65                5.66
    3        3_train  ...       0.65                24.27
    4        4_train  ...       1.00                59.94 
    '''
    
    data['reimbursement_rate'] = data['reimbursement_rate'].apply(lambda x : float(x.strip('%'))/100)
    
    
def preprocess_active_ingredient_embedding(data,dummy):
    
    '''
    Merge ingredient for each drug_ID
    
    Parameters
    ----------
    data : Pandas Dataframe
    dummy : Boolean, True or False
            
    Returns
    -------
    Dataframe : Pandas dataframe with expanded categorical variable (dummy variable) 
                or without dummy variable that group active ingredient by drug_ID
    
    Examples
    --------
    >>> df_active_ingredients = pd.read_csv(path_active_ingredients, sep=",")
    >>> preprocess_active_ingredient_embedding(data = df_active_ingredients,dummy = False)
                      drug_id                                  active_ingredient
        0          0_test    PANTOPRAZOLE SODIQUE SESQUIHYDRATÉ PANTOPRAZOLE
        1         0_train                                        DÉSOGESTREL
        2       1000_test                                        PARACÉTAMOL
        3      1000_train                                       KÉTOCONAZOLE
        4       1001_test                                LOSARTAN POTASSIQUE

    >>> preprocess_active_ingredient_embedding(data = df_active_ingredients,dummy = True)
                  drug_id  ...  active_ingredient__ÉZÉTIMIBE
        0          0_test  ...                             0
        1         0_train  ...                             0
        2       1000_test  ...                             0
        3      1000_train  ...                             0
        4       1001_test  ...                             0    
    '''
    
    if dummy:
        data = pd.get_dummies(data,columns=["active_ingredient"],prefix_sep="__")
    
    data = data.groupby("drug_id").sum().reset_index()
        
    return data
  
    
def preprocess_feature_embedding(data,feature,more_data=None):
    
    '''
    Make dummy variable on a feature from training and test dataframe containing word separated by comma
    
    Parameters
    ----------
    data : Pandas Dataframe
    more_data : Pandas DataFrame, in any case there is value in test data not 
                present in training data on feature
    feature : feature name (str) to make the dummy variable
            
    Returns
    -------
    Dataframe : Pandas dataframe with expanded categorical variable (dummy variable) 
                from feature
    
    Examples
    --------
    >>> df_drugs_train = pd.read_csv(path_drugs_train, sep=",")
    >>> df_drugs_test = pd.read_csv(path_drugs_test, sep=",")  
    >>> preprocess_feature_embedding(data = df_drugs_train,more_data = df_drugs_test,feature="route_of_administration")
                  drug_id  ...  route_of_administration_orale
        0          0_test  ...                             1
        1         0_train  ...                             1
        2       1000_test  ...                             1
        3      1000_train  ...                             1
        4       1001_test  ...                             1  
    '''
    
    if more_data is not None:
        list_feature = set([word for value in set(pd.concat([data[feature], more_data[feature]], ignore_index=True)) for word in value.split(",")])
    else:
        list_feature = set([word for value in set(data[feature]) for word in value.split(",")])
    
    for value in tqdm(list_feature):
        data[feature + "_" + value] = 0
        for index in range(data.shape[0]):
            if value in data[feature][index]:
                data[feature + "_" + value][index] = 1
                
                
def preprocess_NLP(data,features):
    
    '''
    Cleaning text, remove additional space, lowercase text
    
    Parameters
    ----------
    data : Pandas Dataframe
    features : list of feature name (str) where the text cleaning occurs
            
    Returns
    -------
    Dataframe : Pandas dataframe with the cleaned feature
    
    Examples
    --------
    >>> df_drugs_train = pd.read_csv(path_drugs_train, sep=",")
    >>> df_drugs_test = pd.read_csv(path_drugs_test, sep=",")  
    >>> preprocess_NLP(data = df_drugs_train,feature="description")
                  drug_id  ...  description
        0         0_train  ...  5 flacon(s) en verre de 0,5 ml
        1         1_train  ...  plaquette(s) pvc pvdc aluminium de 28 comprimé(s)
        2         2_train  ...  plaquette(s) pvc pvdc aluminium de 28 comprimé(s)
    '''   
    
    for feature in tqdm(features) :
        data[feature] = data[feature].apply(lambda x : re.sub(r'\s+', ' ',x))    
        data[feature] = data[feature].apply(lambda x : re.sub(r'\\n|^ | $', '',str(x).lower()))   
        
    
def NaN_imputation_FeatureScaling_PCA(data,imputation_strategy):
    '''
    Missing value imputation, StandardScaler and PCA transformation
    '''        
    print("Missing value imputation...")
    imputer = SimpleImputer(missing_values=np.nan,strategy=imputation_strategy)
    imputer = imputer.fit(data)
    data = imputer.transform(data)
    
    print("Feature Scaling...")
    data = StandardScaler().fit_transform(data)
    
    print("PCA transformation...")
    pca = PCA()
    principalComponents = pca.fit_transform(data)
    df_PCA = pd.DataFrame(principalComponents)
    
    print("Data preprocessing DONE")
    return df_PCA


def preprocess_whole_pipeline(df_active_ingredients,df_feature_eng,df_drugs_train,df_drugs_test,feature_to_dummy,NaN_imputation_feature_scaling_PCA_boolean,label):    
    '''
    Preprocessing variables from train and test data, feature engineering 
    and active ingredients data using preprocessor functions
    '''
    print("Preprocessing variables from train, test data, feature engineering and active ingredients data: \n")
    #Clean text from active_ingredients feature (lowercase and remove additional spaces)
    preprocess_NLP(data = df_active_ingredients, features=["active_ingredient"])
    #Group all different active ingredient into their drug_ID and make categorical variable (dummy variable)
    df_active_ingredients = preprocess_active_ingredient_embedding(data = df_active_ingredients,dummy = True)    
    
    #Clean text from description feature (lowercase and remove additional spaces)
    preprocess_NLP(data = df_feature_eng, features=["description"])
    #Remove duplicates row (keep the first one) if they have the same description
    df_feature_eng.drop_duplicates(subset ="description", inplace = True)
       
    # Concatenation of Active Ingredient with train and test data using common drug_ID
    data_train = df_drugs_train.merge(df_active_ingredients, on="drug_id")
    data_test = df_drugs_test.merge(df_active_ingredients, on="drug_id")
    #Clean text from textual feature (lowercase and remove additional spaces)
    preprocess_NLP(data = data_train, features=["description","route_of_administration"] + feature_to_dummy)
    preprocess_NLP(data = data_test, features=["description","route_of_administration"] + feature_to_dummy)
    #Merge Feature_Eng data with train and test data
    data_train = data_train.merge(df_feature_eng, on="description", how="left")
    data_test = data_test.merge(df_feature_eng, on="description", how="left")
    
    #Make dummy variable with "route_of_administration" feature
    preprocess_feature_embedding(data = data_train,more_data=data_test,feature="route_of_administration")
    preprocess_feature_embedding(data = data_test,more_data=data_train,feature="route_of_administration")    
    
    #Convert percent string into float representation
    convert_percent(data_train)
    convert_percent(data_test)
    
    #Extract day, month, year from marketing_declaration_date and marketing_authorization_date features
    extract_date(data_train)
    extract_date(data_test)
    
    #Make dummy variable with string/objcect features by concatening train and test data to avoid missing dummy variable that
    #are present in train or test data and not present in test or train data
    data = pd.get_dummies(pd.concat([data_train, data_test], ignore_index=True),columns=feature_to_dummy,prefix_sep="__")
    #Revove space from column
    data.rename(columns=lambda x: re.sub(r'[\s+()\',\[\]<>]',"_",str(x)),inplace=True)
    #Remove useless feature like drug_ID, route_of_administration, description can be used with Word2Vec/BERT in future
    drug_id = data['drug_id']
    data.drop('drug_id', axis=1, inplace=True)
    data.drop("route_of_administration", axis=1, inplace=True)
    data.drop('description', axis=1, inplace=True)
    
    if NaN_imputation_feature_scaling_PCA_boolean == "True":
        '''
        Missing value imputation, StandardScaler and PCA transformation
        '''
        #Remove temporary LABEL for the preprocessing of data
        data_label = data[label]
        data.drop(label, axis=1, inplace=True)
        data_preprocessed = NaN_imputation_FeatureScaling_PCA(data=data,
                                                   imputation_strategy="mean")        
        data = pd.concat([data_preprocessed, data_label], axis = 1)
    
    return data,data_train,data_test,drug_id  
    