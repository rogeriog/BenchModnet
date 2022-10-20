
def import_and_featurize(dataset_name,base_feature, target_feature, target_name=None, num_samples=-1):
    #########################################
    #### IMPORT AND FEATURIZE DATASET
    #########################################
    from matminer.datasets import load_dataset
    from modnet.preprocessing import MODData
    import pickle, os
    ''' Loads featurized data if available, otherwise tries to load dataset, otherwise download dataset
    and featurize the data if none are already available'''
    ### save or import file of the featurized dataset
    if os.path.exists(dataset_name+"_featurized.pkl") and os.path.getsize(dataset_name+".pkl") > 0:
        with open(dataset_name+"_featurized.pkl","rb") as file:
            data = pickle.load(file)
    else:
        ### save or import file of the dataset
        if os.path.exists(dataset_name+".pkl") and os.path.getsize(dataset_name+".pkl") > 0:
            with open(dataset_name+".pkl","rb") as file:
                df = pickle.load(file)
        else:
            df = load_dataset(dataset_name)
            with open(dataset_name+".pkl","wb") as file:
                pickle.dump(df, file)

        if base_feature == "composition":
            from pymatgen.core import Composition
            df["composition"] = df["composition"].map(Composition) # maps composition to a pymatgen composition object

        if num_samples != -1 :
            #### just to reduce data to run
            df=df.sample(n=num_samples, random_state=1).reset_index(drop=True)

        if target_name is None:
            target_name = target_feature.replace(' ','_')+"_target"

        data = MODData(
            materials=df[base_feature], # you can provide composition objects to MODData
            targets=df[target_feature], 
            target_names=[target_name]
        )

        data.featurize()
        with open(dataset_name+"_featurized.pkl","wb") as file:
            pickle.dump(data, file)
    return data

def generate_matbench_kfold():
    #########################################
    #### SETTING UP KFOLD
    #########################################
    from sklearn.model_selection import KFold
    ### getting settings for split 
    ### matbench/scripts/mvb01_generate_validation.py 
    d = {  'VALIDATION_METADATA_KEY': {
                "n_splits": 5,
                "random_state": 18012019,
                "shuffle": True, } }
    kfold_config = d['VALIDATION_METADATA_KEY']
    kfold = KFold(**kfold_config)
    return kfold
################################

#########################################
#### NCV FUNCTION FOR MODNET
#########################################
# K-Fold Cross-Validation with MODNet
def NCV_MODNet(data, kf, n_jobs=None, modnet_model=None):
      '''Function to perform MODNet Cross-Validation
       Parameters
       ----------
      data: MODNet Data, default=None
              MODNet Data used for cross-validation
      kf: Kfold method
           These is the KFold split method.
           Use the same across different models to guarantee reliable comparison.
      n_jobs: int, Default=1
          Number of jobs to run in parallel. Training the estimator and computing the score are parallelized over
          the cross-validation splits. 
          None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.  
      modnet_model: MODNet Model, default=None
              This is the MODNet machine learning model to be used for training if it is not defined through
              validation algorithm.

    Returns
       -------
       The function returns a dictionary containing the metrics 'r2_score', 'neg_mean_absolute_error',
       'neg_root_mean_squared_error', 'neg_median_absolute_error' for both training set and validation set.
      '''
      from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
      from modnet.hyper_opt import FitGenetic
      import pickle
      import numpy as np
      results={'test_r2':[],'test_neg_root_mean_squared_error':[],'test_neg_mean_absolute_error':[],
              'test_neg_median_absolute_error':[], 'test_neg_mean_absolute_error_scaled':[],
              'train_r2':[],'train_neg_root_mean_squared_error':[],'train_neg_mean_absolute_error':[],
              'train_neg_median_absolute_error':[], 'train_neg_mean_absolute_error_scaled':[],}
      def evaluate_scores(results, y, y_predict,Train=False):
          if not Train:
              results['test_r2'].append(r2_score(y, y_predict))
              results['test_neg_root_mean_squared_error'].append(
                  -mean_squared_error(y, y_predict, squared=False))
              results['test_neg_mean_absolute_error'].append(-mean_absolute_error(y, y_predict))
              results['test_neg_median_absolute_error'].append(-median_absolute_error(y, y_predict))
              mean_vector=np.empty(len(y))
              mean_vector.fill(y.mean())
              results['test_neg_mean_absolute_error_scaled'].append(-mean_absolute_error(y, y_predict)/mean_absolute_error(y,mean_vector))
          else:
              results['train_r2'].append(r2_score(y, y_predict))
              results['train_neg_root_mean_squared_error'].append(
                  -mean_squared_error(y, y_predict, squared=False))
              results['train_neg_mean_absolute_error'].append(-mean_absolute_error(y, y_predict))
              results['train_neg_median_absolute_error'].append(-median_absolute_error(y, y_predict))
              mean_vector=np.empty(len(y))
              mean_vector.fill(y.mean())
              results['train_neg_mean_absolute_error_scaled'].append(-mean_absolute_error(y, y_predict)/mean_absolute_error(y,mean_vector))
          return results
      X,y=(data.df_featurized,data.df_targets.values) 
      for i_split in range(len(list(kf.split(X)))):
          train, test = data.split(list(kf.split(X))[i_split])
          ### feature selection is necessary for fitgenetic.
          train.feature_selection(n=-1)
          if modnet_model is None:	
              ga = FitGenetic(train)
              modnet_model = ga.run(nested=0, n_jobs=n_jobs) #,fast=True)i
              with open("mymodel.pkl","wb") as file:
                 pickle.dump(modnet_model, file)
              train_each_time=True
          else:
              with open("mymodel.pkl","rb") as file:
                 modnet_model = pickle.load(file)
              train_each_time=False

##       we use the model to predict in the corresponding subset_train to evaluate as the 
##       as the scikit crossvalidation does.
          y_predict_test=modnet_model.predict(test).values.flatten()
          y_predict_train=modnet_model.predict(train).values.flatten()
          y_test=y[list(kf.split(X))[i_split][1]].flatten()
          y_train=y[list(kf.split(X))[i_split][0]].flatten()
          # print(list(kf.split(X))[i_split],y_test, y_train,y_predict_test, y_predict_train)
          results=evaluate_scores(results, y_test, y_predict_test,Train=False)
          results=evaluate_scores(results, y_train, y_predict_train,Train=True)
          print('results_tmp:',results)
          if train_each_time:
              modnet_model = None
            
      ### results must be complete now, we can generate the results dictionary
      return  {"Training R2 scores": results['train_r2'],
              "Mean Training R2": np.array(results['train_r2']).mean(),
              "Training RMS scores": results['train_neg_root_mean_squared_error'],
              "Mean Training RMS": np.array(results['train_neg_root_mean_squared_error']).mean(),
              "Training MAE scores": results['train_neg_mean_absolute_error'],
              "Mean Training MAE": np.array(results['train_neg_mean_absolute_error']).mean(),
              "Training Median AE scores": results['train_neg_median_absolute_error'],
              "Mean Training Median AE Score": np.array(results['train_neg_median_absolute_error']).mean(),
              "Training MAE scaled": np.array(results['train_neg_mean_absolute_error_scaled']),
              "Mean Training MAE scaled": np.array(results['train_neg_mean_absolute_error_scaled']).mean(),
              "Validation R2 scores": results['test_r2'],
              "Mean Validation R2": np.array(results['test_r2']).mean(),
              "Validation RMS scores": results['test_neg_root_mean_squared_error'],
              "Mean RMS Precision": np.array(results['test_neg_root_mean_squared_error']).mean(),
              "Validation MAE scores": results['test_neg_mean_absolute_error'],
              "Mean Validation MAE": np.array(results['test_neg_mean_absolute_error']).mean(),
              "Validation Median AE scores": results['test_neg_median_absolute_error'],
              "Mean Validation Median AE Score": np.array(results['test_neg_median_absolute_error']).mean(),
              "Validation MAE scaled": np.array(results['test_neg_mean_absolute_error_scaled']),
              "Mean validation MAE scaled": np.array(results['test_neg_mean_absolute_error_scaled']).mean(),
              }

#########################################
#### NCV TO TRAIN, VALIDATE AND GET SCORES
#########################################
'''
from BenchModnet import modnet_bench
data = import_and_featurize("matbench_mp_gap","composition","gap pbe", num_samples=100 )
kfold = generate_matbench_kfold()
results_dictionary=NCV_MODNet(data, kfold, n_jobs=-1)
'''
