######################################################################################
#                                                                                    #
#                                                                                    #
#                       Manomano Introduction/Instruction                            #
#                                                                                    #
#                                                                                    #
######################################################################################


Introduction:

The algorithm chosen to make the prediction is a Machine Learning algorithm, and specifically the Gradient Boosting (XGBoost) by using the regressor mode (squared error objective) since the label is a continuous variable. This algorithm produces a prediction model using decision trees but unlike other decision trees algorithms, this one works sequentially to update its trees and so to improve itself by minimizing the error rate using the training set and a validation set to avoid the overfitting effect. 

This algorithm can easily overfit data so precautions must be taken, so here I displayed the error rate from the training set versus the validation set among the number of the iteration of the algorithm to face the biais-variance tradeoff. Curves with different model parameters are available here: '/challenge-manomano/results/experiments'

Before applying the algorithm, of course, I preprocessed the data, I needed to make dummy variables from the all the object/string type of some features, some values contained aditional spaces so I applied a regular expression to clean all the string and lowercase them all, it allowed to associate more description from the training set to the feature engineering set.

It allowed us to reduce the number of dummy variables since sometimes some value were the same but the string value were different.

Some features needed also some processing like the reimbursement percentage we converted into float representation or to extract the month, year and day of the marketing declaration and authorization date.

At the end, I added a data preprocessor to fill missing (NaN) value with the median of each related features, then apply a feature scaling (z = (x - u) / s) and a PCA, but the XGBoost algorithm seems to handle better the NaN value as they are.

This algorithm give an important information on feature which is the Feature Importance available here: '/challenge-manomano/results/experiments', I can observe for instance that the price of drug depend mostly of the year of the marketing declaration and authorization of the drug, the count of comprime, seringue, ml of the drug, the reimbursement rate of the drug

The RMSE seems high (~50), the model need to be improved, even after using preprocessors, the RMSE does not decrease, maybe it needs more features related to the price of the drug or maybe by making a word embedding (word2vec or BERT) of strings features like description or active ingredients, or improving the association of description from the training set to the feature engineering set by removing stop words or others special chars like parenthesis or make a similarity sentence of missing description from training set in the feature engineering set.



Instruction:

1 - Download and Unzip the project anywhere or open a terminal and run: 'git clone https://github.com/serkhanekarim/challenge-manomano.git'

2 - First, you will need to install the required libraries, if you do not want to install the required libraries in your base environment, you can create a specific environement by opening a terminal in the 'challenge-manomano' folder and running: 'python3 -m venv $HOME/tmp/venv-challenge-manomano/' and then running: 'source $HOME/tmp/venv-challenge-manomano/bin/activate' to place yourself in the created environmment.

3 - Now, open a terminal in 'challenge-manomano' folder and run: 'make install' and wait for the required libraries to be installed

4 - To get the full help for running the code, open a terminal (still in the 'challenge-manomano' folder) and run: './main.py --help'



Here the two common commands to run:

   1 -	'./main.py -mode prediction'
   2 -	'./main.py -mode training'

The first one is to predict the given default test data (/challenge-manomano/data/drugs_test.csv) by using the default trained model (/challenge-manomano/model/XGBoost_drug_price.model), you can specify another test data file and a specific model to use by running:

	'./main.py -mode prediction -path_of_data_to_predict /path/of/data/to/predict.csv -path_model /path/of/model.bin'
	
By default, the output is here: '/challenge-manomano/results/submission.csv' - You can specify another path or just print the result with the argument: -path_of_output_prediction
	
	
The second one is to create/train a model using the given default training data (from /challenge-manomano/data/), you can specify another directory with the argument: -data_directory
You can also specify parameters for the model and activate or not data preprocessing, here the help:

  -NaN_imputation_feature_scaling_PCA_usage [{False,True}]
                        Apply or not NaN imputation, Feature Scaling and PCA
  -max_depth [MAX_DEPTH]
                        Maximum depth of a tree. Increasing this value will make the model more complex and more
                        likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as
                        hist or gpu_hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes
                        memory when training a deep tree. range: [0,∞] (0 is only accepted in lossguided growing
                        policy when tree_method is set as hist or gpu_hist)
  -eta [ETA]            Step size shrinkage used in update to prevents overfitting. After each boosting step, we can
                        directly get the weights of new features, and eta shrinks the feature weights to make the
                        boosting process more conservative. range: [0,1]
  -num_round [NUM_ROUND]
                        The number of rounds for boosting


By default, model are saved here: '/challenge-manomano/model/'
It will give you plotting of Learning Curves and Features Importances available here: /challenge-manomano/results/


