import warnings
warnings.filterwarnings("ignore")
def split_df(df,n):
    '''Splits a dataframe in n chunks of similar size. 
    Parameters
    ----------
    df : pandas Dataframe,
    n : int, number of chunks to divide Dataframe.
    Returns
    ----------
    list : list of Dataframes,
        subdivided dataframe that if concatenated produce the original DataFrame.
    '''
    ln=len(df) // n
    return [ df[i:i+ln] for i in range(0,df.shape[0],ln) ]

def import_and_featurize(dataset_name,base_feature, target_feature, featurizers=None, id='', mode='general',
                        num_samples=-1, progressive_featurization=False, feat_steps=(1,10,10),
                        model_for_custom_feats=None, n_jobs=None):
    '''Function to import and featurize dataset, it may start from a pickle file for the dataset and featurize it,
    or just read the featurized dataset if it was already produced. Since featurization is very slow for large datasets, 
    this function implements a progressive mode which splits the featurization in small parts to be assembled in a 
    final featurized dataset in the end. 

    Parameters
    ----------
    dataset_name: str, 
        Name of the dataset to be downloaded with matminer or to be loaded from the folder
    base_feature: str, 
        Usually it is either 'structure' or 'composition' depending on the dataset.
    target_feature: str, 
        The name of the feature to be predicted as presented in the raw dataset.
    featurizers: list of matminer featurizers, default=None
        If mode='general' or 'MODNetCustom', list of matminer featurizers to be applied to the dataset, 
        if mode='MODNet' this variable is disregarded, 
    id: str, default='',
        String to append on pkl files that are dumped to help identification.
    mode: str, default='general'
        Mode of featurizer, by default it should apply matminer featurizers as specified to the dataset. 
        May also be 'MODNet' to use MODNet default featurization, or 'MODNetCustom' to implement additional
        featurizers to be passed on the function.
    num_samples: int, default=-1
        In case it is preferable to work with a subset of the dataset, specify the number of samples to be taken.
        These rows will be taken at random.
    progressive_featurization: Boolean, default=False,
        Activate to make the featurization in steps, in each step a ~_featurized_(feat_step).pkl file will be produced.
    feat_steps: (start,stop,total_steps), default=(1,10,10) 
        List to define progressive featurization of the data. 
    model_for_custom_feats: sklearn.model, default=None
        If using MODNetCustom, in this variable a sklearn.model should be passed to fit the features to the target producing
        a processed feature capturing chemical intuition from more complex features that need to be analysed together.
    n_jobs: int, default=None
        Number of jobs.

    Returns
    -------
    The function returns the featurized dataframe. It may be as a MODData class or a pandas dataframe depending on the 
    mode. 
    '''
    from matminer.datasets import load_dataset
    from modnet.preprocessing import MODData
    import pickle, os
    import pandas as pd
    
    #### MODULES FOR FEATURIZATION 
    def default_general_featurization():
        if progressive_featurization :
            # name_featurizer=featurizer.__name__
            start,stop,total_steps = feat_steps
            df_chunks=split_df(df,total_steps)
            for featurizer in featurizers: 
                ## featurize each chunk of the dataframe
                for i in range(start-1,stop):
                    df_chunks[i]=featurizer.featurize_dataframe(df_chunks[i], base_feature)
                    with open(f"{dataset_name}{id}_featurized_{str(i+1)}of{str(total_steps)}.pkl","wb") as file:
                        pickle.dump(df_chunks[i], file)

            ## check if all chunks of dataframe were featurized.
            for i in range(total_steps):
                namechunk=f"{dataset_name}{id}_featurized_{str(i+1)}of{str(total_steps)}.pkl"
                if os.path.exists(namechunk) and os.path.getsize(namechunk) > 0 :
                    with open(namechunk,"rb") as file: 
                        df_chunks[i]=pickle.load(file)
                else:
                    raise RuntimeError(f"The file corresponding to chunk {str(i+1)} of {str(total_steps)} is not present for \
                            the featurized dataset. Featurize the missing data and try again.")
            ## assemble all featurized chunks in the featurized dataframe.
            df=pd.concat(df_chunks, axis=0)
            # save complete featurized dataframe
            with open(f"{dataset_name}{id}_featurized.pkl","wb") as file: 
                        pickle.dump(df, file) 
        else: # featurize in one go.
            for featurizer in featurizers: 
                df=featurizer.featurize_dataframe(df, base_feature)
            # save complete featurized dataframe
            with open(f"{dataset_name}{id}_featurized.pkl","wb") as file: 
                        pickle.dump(df, file) 
        return df
    def default_MODnet_featurization():
        if progressive_featurization :
            start,stop,total_steps = feat_steps
            df_chunks=split_df(df,total_steps)
            target_name = target_feature.replace(' ','_')+"_target"
            for i in range(start-1,stop):
                data = MODData(
                    materials=df_chunks[i][base_feature].reset_index(drop=True), 
                    targets=df_chunks[i][target_feature].reset_index(drop=True),
                    target_names=[target_name]
                )
                data.featurize(n_jobs=n_jobs)
                with open(f"{dataset_name}{id}_featurized_{str(i+1)}of{str(total_steps)}.pkl","wb") as file:
                    pickle.dump(data.df_featurized, file)      
## check if all chunks of dataframe were featurized.
            for i in range(total_steps):
                namechunk=f"{dataset_name}{id}_featurized_{str(i+1)}of{str(total_steps)}.pkl"
                if os.path.exists(namechunk) and os.path.getsize(namechunk) > 0 :
                    with open(namechunk,"rb") as file: 
                        df_chunks[i]=pickle.load(file)
                else:
                    raise RuntimeError(f"The file corresponding to chunk {str(i+1)} of {str(total_steps)} is not present for \
                            the featurized dataset. Featurize the missing data and try again.")
            ## assemble all featurized chunks in the featurized dataframe.
            df_featurized=pd.concat(df_chunks, axis=0)
            ## reinitialize MODData with full dataset and attribute df_featurized
            data = MODData(
                    materials=df[base_feature], 
                    targets=df[target_feature], 
                    target_names=[target_name]
                )
            data.df_featurized=df_featurized 
            # save complete featurized MODData
            with open(f"{dataset_name}{id}_featurized.pkl","wb") as file: 
                        pickle.dump(data, file)
                        start=1 ## reset start to 1
                        stop=total_steps
                        #feat_steps = (start,stop,total_steps)

        else:
            target_name = target_feature.replace(' ','_')+"_target"
            data = MODData(
                materials=df[base_feature], 
                targets=df[target_feature], 
                target_names=[target_name]
            )
            data.featurize(n_jobs=n_jobs)
            with open(dataset_name+id+"_featurized.pkl","wb") as file:
                pickle.dump(data, file)
        return data

    #########################################
    #### IMPORT AND FEATURIZE DATASET
    #########################################
    ''' Loads featurized data if available, otherwise tries to load dataset, otherwise download dataset
    and featurize the data if none are already available'''
    id="_"+id if id != '' else id 
    ### save or import file of the featurized dataset
    if os.path.exists(dataset_name+id+"_featurized.pkl") and os.path.getsize(dataset_name+id+".pkl") > 0:
        with open(dataset_name+id+"_featurized.pkl","rb") as file:
            data = pickle.load(file)
            print("Loading dataset already featurized...")
        return data
    else:
        ### save or import file of the dataset
        if os.path.exists(dataset_name+id+".pkl") and os.path.getsize(dataset_name+id+".pkl") > 0:
            with open(dataset_name+id+".pkl","rb") as file:
                df = pickle.load(file) ## loads raw dataset if available in folder
        else:
            df = load_dataset(dataset_name) ## downloads and then saves the raw dataset
            with open(dataset_name+id+".pkl","wb") as file:
                pickle.dump(df, file)

        if base_feature == "composition":
            from pymatgen.core import Composition
            df["composition"] = df["composition"].map(Composition) # maps composition to a pymatgen composition object

        if num_samples != -1 :
            #### just to reduce data to run
            df=df.sample(n=num_samples, random_state=1).reset_index(drop=True)
        ###########
        #### MODNET
        ###########
        if mode == 'MODNet':
            return default_MODnet_featurization()

        ###########
        #### CUSTOM MODNET
        ###########
        if mode == 'MODNetCustom':
            ## first runs default MODNet_featurization
            data=default_MODnet_featurization()
            ## loads custom featurizers if already calculated
            if os.path.exists(f"{dataset_name}{id}_customfeaturized.pkl") and os.path.getsize(f"{dataset_name}{id}_customfeaturized.pkl",) > 0:
                with open(f"{dataset_name}{id}_customfeaturized.pkl","rb") as file:
                    df = pickle.load(file)
                    print("Loading dataset with custom featurization...")
            else: ## to perform featurization of custom featurizers.
                ## first we featurize with the custom featurizers
                if progressive_featurization :
                    # name_featurizer=featurizer.__name__
                    start,stop,total_steps = feat_steps
                    df_chunks=split_df(df,total_steps)
                    for featurizer in featurizers:
                        ## featurize each chunk of the dataframe
                        for i in range(start-1,stop):
                            df_chunks[i]=featurizer.featurize_dataframe(df_chunks[i], base_feature)
                            with open(f"{dataset_name}{id}_customfeaturized_{str(i+1)}of{str(total_steps)}.pkl","wb") as file:
                                pickle.dump(df_chunks[i], file)

                    ## check if all chunks of dataframe were featurized.
                    for i in range(total_steps):
                        namechunk=f"{dataset_name}{id}_customfeaturized_{str(i+1)}of{str(total_steps)}.pkl"
                        if os.path.exists(namechunk) and os.path.getsize(namechunk) > 0 :
                            with open(namechunk,"rb") as file: 
                                df_chunks[i]=pickle.load(file)
                        else:
                            raise RuntimeError(f"The file corresponding to chunk {str(i+1)} of {str(total_steps)} is not present for \
                                    the featurized dataset. Featurize the missing data and try again.")
                    ## assemble all featurized chunks in the featurized dataframe.
                    df=pd.concat(df_chunks, axis=0)
                    # save complete featurized dataframe
                    with open(f"{dataset_name}{id}_customfeaturized.pkl","wb") as file: 
                                pickle.dump(df, file) 
                else: # featurize in one go.
                    for featurizer in featurizers: 
                        df=featurizer.featurize_dataframe(df, base_feature)
                    # save complete featurized dataframe
                    with open(f"{dataset_name}{id}_customfeaturized.pkl","wb") as file: 
                                pickle.dump(df, file) 
            ## now using the model_for_custom_feats we train the model to predict the target
            for featurizer in featurizers:
                feature_labels=featurizer.feature_labels() 
                X = df[feature_labels]
                y = df[target_feature]
                model_for_custom_feats.fit(X,y)
                feat_predicted=model_for_custom_feats.predict(X)
                ## now we drop the featurizers and insert the feat_predicted in df
                df=df.drop(feature_labels,axis=1)
                feat_name=str(featurizer).split('(')[0]+'_predict'
                #finally add the custom feats to the MODData 
                data.df_featurized[feat_name]=feat_predicted
                with open(f"{dataset_name}{id}_featurized.pkl","wb") as file: 
                            pickle.dump(data, file) 
            return data
        ###########
        #### GENERAL
        ###########       
        if mode == 'general':
            return default_general_featurization()

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
          print(train.optimal_features)
          ## if modnet model is not specified directly through modnet_model variable, 
          ## it will run genetic algorithm to determine the model, the variable train_each_time
          ## is important not to keep the evaluated model in the next splits.
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
          print('Results KSPLIT: ',results)
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

# K-Fold Cross-Validation
def sklearn_CV(model, data, base_feature, target_feature, _kf, n_jobs=1):
      '''Function to perform Folds Cross-Validation
       Parameters
       ----------
      model: Python Class, default=None
              This is the machine learning algorithm to be used for training.
      data: Dataframe
           .
      base_feature: str
           .
      target_feature: str
           .
      _kf: Kfold split
           This is to guarantee the same split in all models evaluated.
      _n_jobs: int
          Number of jobs to run in parallel. Training the estimator and computing the score are parallelized over
          the cross-validation splits. 
          None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.  
      Returns
       -------
       The function returns a dictionary containing the metrics 'accuracy', 'precision',
       'recall', 'f1' for both training set and validation set.
      '''
      from sklearn.model_selection import cross_validate
      import numpy as np
            
      _scoring = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error', 'neg_median_absolute_error']
      _X = data.drop([target_feature,base_feature],axis=1)
      _y = data[target_feature]
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_kf,
                               scoring=_scoring, 
                               return_train_score=True,
                               n_jobs=n_jobs)
      
      mean_vector=np.empty(len(_y))
      mean_vector.fill(_y.mean())
      from sklearn.metrics import mean_absolute_error
      results_dictionary={"Training R2 scores": results['train_r2'],
              "Mean Training R2": results['train_r2'].mean(),
              "Training RMS scores": results['train_neg_root_mean_squared_error'],
              "Mean Training RMS": results['train_neg_root_mean_squared_error'].mean(),
              "Training MAE scores": results['train_neg_mean_absolute_error'],
              "Mean Training MAE": results['train_neg_mean_absolute_error'].mean(),
              "Training Median AE scores": results['train_neg_median_absolute_error'],
              "Mean Training Median AE Score": results['train_neg_median_absolute_error'].mean(),
              "Training MAE scaled": np.array(results['train_neg_mean_absolute_error'])/mean_absolute_error(_y,mean_vector),
              "Mean Training MAE scaled": np.array(np.array(results['train_neg_mean_absolute_error'])/mean_absolute_error(_y,mean_vector)).mean(),
              "Validation R2 scores": results['test_r2'],
              "Mean Validation R2": results['test_r2'].mean(),
              "Validation RMS scores": results['test_neg_root_mean_squared_error'],
              "Mean RMS Precision": results['test_neg_root_mean_squared_error'].mean(),
              "Validation MAE scores": results['test_neg_mean_absolute_error'],
              "Mean Validation MAE": results['test_neg_mean_absolute_error'].mean(),
              "Validation Median AE scores": results['test_neg_median_absolute_error'],
              "Mean Validation Median AE Score": results['test_neg_median_absolute_error'].mean(),
              "Validation MAE scaled": np.array(results['test_neg_mean_absolute_error'])/mean_absolute_error(_y,mean_vector),
              "Mean Validation MAE scaled": np.array(np.array(results['test_neg_mean_absolute_error'])/mean_absolute_error(_y,mean_vector)).mean(),  
            }
      return results_dictionary
