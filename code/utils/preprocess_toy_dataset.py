import numpy as np
def preprocess(X,Y,training_ratio = 0.8,norm = None):
        '''
	normalizes data and splits it into training/testing
        '''
        #shuffle
        XY = np.concatenate((X,np.expand_dims(Y,axis = 1)),axis = 1)
        np.random.shuffle(XY)
        X = XY[:,:-1]
        Y = XY[:,-1]
        #normalize
        X = np.squeeze(X)
        row_norm = np.linalg.norm(X,axis=1)
        min_norm = np.amin(row_norm)
        max_norm = np.amax(row_norm)
        X = (X-min_norm) / (max_norm-min_norm)
        #split data
        Y = np.squeeze(Y)
        X_train, X_test = X[:int(training_ratio*len(X[:,0])),:],X[int(training_ratio*len(X[:,0])):,:]
        Y_train, Y_test = Y[:int(training_ratio*len(X[:,0]))],Y[int(training_ratio*len(X[:,0])):]

        #remove OOD from testing data
        index = Y_test >= 0
        X_test = X_test[index]
        Y_test = Y_test[index]

        #Create logits from labels
        index = Y_train < 0
        Y_train[index] = Y_train.max()+1 #let the ood data contain class n_classes+1 which we later remove
        logits_train = np.zeros((Y_train.size, int(Y_train.max()+1)))
        logits_train[np.arange(Y_train.size).astype(int),Y_train.astype(int)] = 100
        logits_train = logits_train +1
        logits_train = np.delete(logits_train, -1, axis=1) #delete last column
        logits_test = np.zeros((Y_test.size, int(Y_test.max()+1)))
        logits_test[np.arange(Y_test.size).astype(int),Y_test.astype(int)] = 100
        logits_test = logits_test +1
        return X_train,logits_train,X_test,logits_test
