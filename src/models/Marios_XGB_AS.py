""" Amazon Access Challenge Code for ensemble

Marios Michaildis script for Amazon .

xgboost on input data

based on Paul Duan's Script.

"""
from __future__ import division
import numpy as np
from sklearn import  preprocessing
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split

PERCENT_DATA = 0.05
USE_PARTIAL_DATA = 0

SEED = 15  # always use a seed for randomized procedures


def load_data(filename, use_labels=True):
    """
    Load data from CSV files and return them as numpy arrays
    The use_labels parameter indicates whether one should
    read the first column (containing class labels). If false,
    return all 0s. 
    """

    # load column 1 to 8 (ignore last one)
    data = np.loadtxt(open( filename), delimiter=',',
                      usecols=range(1, 9), skiprows=1)
    if use_labels:
        labels = np.loadtxt(open( filename), delimiter=',',
                            usecols=[0], skiprows=1)
    else:
        labels = np.zeros(data.shape[0])
    return labels, data


def save_results(predictions,IDs,  filename):
    with open(filename, 'w') as f:
        f.write("id,TARGET\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (IDs[i], pred))


#X_train,y_train,model, SEED , bagging, X_cv, update_seed=True

def train_and_predict(X_train,y_train,model, myseed, bags, X_cv, update_seed=True):
    
   # create array object to hold predictions 
   baggedpred=[ 0.0  for d in range(0, (X_cv.shape[0]))]
   #loop for as many times as we want bags
   for n in range (0, bags):
        #shuff;e first, aids in increasing variance and forces different results
        #UPDATE THE TWO LINES BELOW SO THAT IT USES BAGGING
        X_train_2 = X_train
        y_train_2 = y_train
        #X_train,y_train=shuffle(Xs,ys, random_state=seed+n)
          
        if update_seed: # update seed if requested, to give a slightly different model
            model.set_params(seed=myseed + n)
        model.fit(X_train_2,y_train_2) # fit model0.0917411475506
        preds=model.predict_proba(X_cv)[:,1] # predict probabilities
        # update bag's array
                 
        baggedpred+=preds
   # divide with number of bags to create an average estimate           
   
   baggedpred = np.true_divide(baggedpred,float(bags))
   # return probabilities            
   return np.array(baggedpred) 
   
   
# using numpy to print results
def printfilcsve(X, filename):

    np.savetxt(filename,X) 
    

def main():
    """
    Fit models and make predictions.
    We'll use one-hot encoding to transform our categorical features
    into binary features.
    y and X will be numpy array objects.
    """
    
    filename="marios_xgb" # nam prefix
    #model = linear_model.LogisticRegression(C=3)  # the classifier we'll use
    
    model=xgb.XGBClassifier(max_depth=5,colsample_bytree=0.7,n_estimators=500, seed=SEED,learning_rate=0.02,subsample=0.7)

    # === load data in memory === #
    print "loading data"

    # === load data into numpy arrays === #
    print ("loading training data")
    X=np.loadtxt("train.csv",skiprows=1, delimiter=",", usecols=range (1, 370)) # 1 is inclusive and 22 exclusive. so it is [1,21] basically. We skip 1 row since the first one is headers
    print ("loading labels")    
    y=np.loadtxt("train.csv",skiprows=1, delimiter=",", usecols=[370]) # The last one - the respnse variable   
    print ("loading test data")
    X_test=np.loadtxt("test.csv",skiprows=1, delimiter=",", usecols=range (1, 370)) # 1 is inclusive and 22 exclusive. so it is [1,21] basically
    print ("loading ids")    
    ids=np.loadtxt("test.csv",skiprows=1, delimiter=",", usecols=[0]) # The first column is the id
    


    if USE_PARTIAL_DATA == 1:
        X, X_discard, y, y_discard = train_test_split(X, y, train_size=PERCENT_DATA, random_state=42)



    # === one-hot encoding === #
    # we want to encode the category IDs encountered both in
    # the training and the test set, so we fit the encoder on both
    # encoder = preprocessing.OneHotEncoder()
    # encoder.fit(np.vstack((X, X_test)))
    # X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    # X_test = encoder.transform(X_test)


    # if you want to create new features, you'll need to compute them
    # before the encoding, and append them to your dataset after

    #create arrays to hold cv an dtest predictions
    train_stacker=[ 0.0  for k in range (0,(X.shape[0])) ] 

    # === training & metrics === #
    mean_auc = 0.0
    bagging=5 # number of models trained with different seeds
    n = 5  # number of folds in strattified cv
    kfolder=StratifiedKFold(y, n_folds= n,shuffle=True, random_state=SEED)     
    i=0
    for train_index, test_index in kfolder: # for each train and test pair of indices in the kfolder object
        # creaning and validation sets
        X_train, X_cv = X[train_index], X[test_index]
        y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
        #print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))

        # if you want to perform feature selection / hyperparameter
        # optimization, this is where you want to do it

        # train model and make predictions 
        preds=train_and_predict(X_train,y_train,model, SEED , bagging, X_cv, update_seed=True)   
        

        # compute AUC metric for this CV fold
        roc_auc = roc_auc_score(y_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
        mean_auc += roc_auc
        
        no=0
        for real_index in test_index:
                 train_stacker[real_index]=(preds[no])
                 no+=1
        i+=1
        

    mean_auc/=n
    print (" Average AUC: %f" % (mean_auc) )
    print (" printing train datasets ")
    printfilcsve(np.array(train_stacker), filename + ".train.csv")          

    # === Predictions === #
    # When making predictions, retrain the model on the whole training set
    preds=train_and_predict(X, y,model, SEED, bagging, X_test, update_seed=True)  


    #create test predictions file to be used in stacking
    printfilcsve(np.array(preds), filename+ ".test.csv")  

    #create training predictions file to be used in stacking 
    printfilcsve(np.array(train_stacker), filename + ".train.csv")

    #create submission file
    save_results(preds, ids,  filename+"_submission_" +str(mean_auc).replace(".","_") + ".csv") # putting the actuall AUC (of cv) in the file's name for reference

if __name__ == '__main__':
    main()
