import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers.core import Dropout,Flatten,Reshape,Activation
from keras.layers.embeddings import Embedding

def get_dataset(StateMatrix,sequence,split_train=0.8):
    
    #print 'Top 50 states according to the number of sites (not the 1st state corresponding to zeros)'
    #idxbeststates = np.argsort(np.sum(StateMatrix,axis=1))[::-1][1:50]
    States =  StateMatrix[1:,:].sum(axis=0)[None,:] # We remove the zeros state
    
    idx1 = np.where(States.sum(axis=0)>0)[0]
    States = States[:,idx1]
    N1 = States.shape[1]
    
    print 'Input Sequence'
    Seq = sequence[idx1]
    SeqOHE = np.zeros((4,len(Seq[0]),N1))
    for i in range(N1):
        for k in range(len(Seq[i])):
            if Seq[i][k]=='A':
                m=0
            if Seq[i][k]=='C':
                m=1
            if Seq[i][k]=='T':
                m=2
            if Seq[i][k]=='G':
                m=3

            SeqOHE[m,k,i] = 1
    
    print 'Including a zeros state with reshuffled bases'
    States = np.concatenate((np.zeros((1,N1)),States),axis=0)
    #Nubi = int(States.sum())
    StatesZ = np.zeros((States.shape[0],N1)) # As many zeros as ones
    StatesZ[0,:]=1
    
    print 'Building Null model'
    SeqOHEZ = Reshuffle_site_Dinucleotide(SeqOHE[:,:,np.random.permutation(N1)[:N1]])
    #SeqOHEZ = Reshuffle_site_LSTM(SeqOHE[:,:,np.random.permutation(N1)[:N1]],CorrStep=5)
    
    print 'Concatenation'
    States = np.concatenate((States,StatesZ),axis=1)
    SeqOHE = np.concatenate((SeqOHE,SeqOHEZ),axis=2)
    
    print 'Transpose matrix in order to get the batch dimension first'
    States = np.transpose(States,(1,0))
    SeqOHE = np.transpose(SeqOHE,(2,1,0))
    
    idxR = np.random.permutation(States.shape[0])
    N_train = int(States.shape[0]*split_train)
    xtrain = SeqOHE[idxR[:N_train],:,:]
    ytrain = States[idxR[:N_train],:]
    
    xtest = SeqOHE[idxR[N_train:],:,:]
    ytest = States[idxR[N_train:],:]
    
    return xtrain,ytrain,xtest,ytest

def get_dataset_CT(matrixCT,sequence,split_train=0.8):
    
    #print 'Top 50 states according to the number of sites (not the 1st state corresponding to zeros)'
    #idxbeststates = np.argsort(np.sum(StateMatrix,axis=1))[::-1][1:50]
    
    idx1 = np.where(matrixCT.sum(axis=0)>0)[0]
    matrixCT = matrixCT[:,idx1]
    N1 = matrixCT.shape[1]
    
    print 'Input Sequence'
    Seq = sequence[idx1]
    SeqOHE = np.zeros((4,len(Seq[0]),N1))
    for i in range(N1):
        for k in range(len(Seq[i])):
            if Seq[i][k]=='A':
                m=0
            if Seq[i][k]=='C':
                m=1
            if Seq[i][k]=='T':
                m=2
            if Seq[i][k]=='G':
                m=3

            SeqOHE[m,k,i] = 1
    
    print 'Transpose matrix in order to get the batch dimension first'
    matrixCT = np.transpose(matrixCT,(1,0))
    SeqOHE = np.transpose(SeqOHE,(2,1,0))
    
    idxR = np.random.permutation(matrixCT.shape[0])
    N_train = int(matrixCT.shape[0]*split_train)
    xtrain = SeqOHE[idxR[:N_train],:,:]
    ytrain = matrixCT[idxR[:N_train],:]
    
    xtest = SeqOHE[idxR[N_train:],:,:]
    ytest = matrixCT[idxR[N_train:],:]
    
    return xtrain,ytrain,xtest,ytest

#def Reshuffle_site(SeqOHE):
    
    #M,N,K = SeqOHE.shape
    #SeqOHE_R = np.zeros((M,N,K))
    
    #for k in range(K):
        #SeqOHE_R[:,:,k] = SeqOHE[:,np.random.permutation(N),k]
    
    #return SeqOHE_R

def Reshuffle_site_Dinucleotide(SeqOHE):
    M,N,K = SeqOHE.shape
    SeqOHE_R = np.zeros((M,N,K))
    N2 = int(N/2)
    #idxDin = [np.array([2*i,2*i+1]) for i in range(N2)]
    idxDin = np.array([2*i for i in range(N2)])
    for k in range(K):
        idxDinR = np.zeros(N,dtype=np.int32)
        idxDinR[idxDin] = idxDin[np.random.permutation(N2)]
        idxDinR[idxDin+1] = idxDinR[idxDin]+1
        SeqOHE_R[:,:,k] = SeqOHE[:,idxDinR,k]
    
    return SeqOHE_R

import multiprocessing as mp
def Reshuffle_site_LSTM(SeqOHE,CorrStep):
    
    M,N,K = SeqOHE.shape #M=4,N=200,K=NumSeq
    
    #train the model
    NMS = LSTM_Seq(CorrStep=CorrStep,NLayer=4)
    NMS.build_model()
    NMS.fit(SeqOHE,epochs_all=5,epochs=5)
    NMS.save_model()
    
    SeqOHE_R = np.zeros((M,N,K))
    #for k in range(K):
        #print str(k) + ' / ' + str(K) 
    start_seq = [SeqOHE[:,np.random.permutation(N)[:CorrStep],k] for k in range(K)]
    args = [(start_seq[k],CorrStep,NMS)for k in range(K)]
    pool = mp.Pool(6)
    SeqOHE_R = zip(*pool.map(wrapper_prediction,args))
    SeqOHE_R = np.hstack(SeqOHE_R)
    
    return SeqOHE_R

def wrapper_prediction(args):
    start_seq,CorrStep,NMS = args
    SeqOHE_R = np.zeros((4,200))
    SeqOHE_R[:,:CorrStep] = start_seq
    for i in range(CorrStep,N-1):
        SeqOHE_R[:,i+1] = NMS.best_model.predict((SeqOHE_R[:,(i-CorrStep):i].T)[None,:,:])
    
    return SeqOHE_R
    
class LSTM_Seq:
    def __init__(self,CorrStep=10,NLayer=4,filepath='/users/invites/jmorlot/Documents/Sequence_Embedding/model.hdf5'):
        
        self.model = []
        self.CorrStep = CorrStep
        self.NLayer = NLayer
        self.validation_split=0.12
        #Best accuracy on validation set model
        self.best_model=[]
        self.best_accuracy = 0
        
        self.filepath = filepath
        
        self.current_epoch = 0
        
    def build_model(self):
        self.model = Sequential()
        self.model.add(Reshape((4*self.CorrStep,),input_shape=(self.CorrStep,4)))
        self.model.add(Embedding(4*self.CorrStep, self.CorrStep))
        self.model.add(LSTM(self.NLayer))
        self.model.add(Dense(4,activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    def load_model(self):
        self.model.load_weights(self.filepath, by_name=False)
    
    def save_model(self):
        self.model.save_weights(self.filepath)
        
    def fit(self,xtrain,epochs_all=15,epochs=5):
        
        print 'Building Training Data'
        xtrain = np.concatenate(np.transpose(xtrain,(2,1,0)),axis=0)
        xtrain = xtrain[:10000]
        #Building validation set
        Nval = int(xtrain.shape[0]*self.validation_split)
        xval = xtrain[:Nval]
        xtrain = xtrain[Nval:]
        
        # Model Fitting
        NEpochs = epochs_all/epochs
        
        Nbatch = int(xtrain.shape[0])
        
        for i in range(NEpochs):
            self.current_epoch = self.current_epoch + epochs
            print '\nCurrent Epoch = ' + str(self.current_epoch)
            self.model.fit_generator(generator = self.data_generator(xtrain), 
                                     epochs=epochs,steps_per_epoch=Nbatch)
            print 'Evaluate the model on validataion set'
            self.evaluate(xval)
            
            
    def evaluate(self,xval):
        
        loss,accuracy = self.model.evaluate_generator(generator = self.data_generator(xval),steps=int(xval.shape[0]))
        print 'loss = ' + str(loss)
        print 'accuracy = ' + str(accuracy)
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = self.model
    
    def data_generator(self,xtrain):
        while True:
            for i in np.arange(self.CorrStep,xtrain.shape[0]):
                yield xtrain[(i-self.CorrStep):i][None,:,:],xtrain[i][None,:]





def generate_param_CNN_States(Nparam):
    
    params = []
    params_keys = ['MotifLen','Nmotif','NHidden','LearningRate','Momentum','DropoutRate','optimizer','batch_size']
    
    for i in range(Nparam):
        dict_param = {
            'MotifLen':np.random.choice([12,24,36,48]),
            'Nmotif':np.random.choice([64,128,200,256]),
            'NHidden':np.random.choice([2,8,32,64]),
            'LearningRate':np.random.choice(np.logspace(5e-4,5e-2,5)),
            'Momentum':np.random.choice(0.95 + np.sqrt(np.linspace(0,0.0025,5))),
            'DropoutRate':np.random.choice([0.25,0.5,0.75]),
            'optimizer':np.random.choice(['rmsprop','adam','nadam']),
            'batch_size':np.random.choice([64,128,256,512]),
            }
        params.append(dict_param)
    
    return params





'''
    Neural Network from all states
'''

def get_dataset_all_states(States,sequence,split_train=0.8,uniformisation=False):
    
    Nstates = States.shape[0]
    print 'Top ' + str(Nstates) +' states according to the number of sites (not the 1st state corresponding to zeros)'
    
    idx1 = np.where(States.sum(axis=0)>0)[0]
    N1 = len(idx1)
    States = States[:,idx1]
    
    print 'Input Sequence'
    Seq = sequence[idx1]
    SeqOHE = np.zeros((4,len(Seq[0]),N1))
    for i in range(N1):
        for k in range(len(Seq[i])):
            if Seq[i][k]=='A':
                m=0
            if Seq[i][k]=='C':
                m=1
            if Seq[i][k]=='T':
                m=2
            if Seq[i][k]=='G':
                m=3

            SeqOHE[m,k,i] = 1
   
    print 'Transpose matrix in order to get the batch dimension first'
    States = np.transpose(States,(1,0)) #Sequence x State
    SeqOHE = np.transpose(SeqOHE,(2,1,0)) #Sequences x 200 bp x 4 nucl.
    
    
    print 'Split the data set in train & test set'
    idxR = np.random.permutation(States.shape[0])
    N_train = int(States.shape[0]*split_train)
    xtrain = SeqOHE[idxR[:N_train]]
    ytrain = States[idxR[:N_train]]
    
    xtest = SeqOHE[idxR[N_train:]]
    ytest = States[idxR[N_train:]]
    
    
    if uniformisation==True:
        xtrain,ytrain = Uniformisation(xtrain,ytrain)
    
    return xtrain,ytrain,xtest,ytest

def Uniformisation(xtrain,ytrain):
    '''
        Uniformisation of the number of sites per datasets
    '''
    print xtrain.shape
    print ytrain.shape
    print 'Uniformisation of the number of site per states (in training data ONLY)'
    #Create a new dataset sampled in the previous 
    idxSamples = np.empty(0,dtype=np.int32)
    NSite = int(np.max(ytrain.sum(axis=0))/2) #Num site per state in training data
    for i in range(ytrain.shape[1]):
        idxi = np.where(ytrain[:,i]>0)[0]
        idxi = idxi[np.random.randint(low=0,high=len(idxi),size=NSite)] #Random sampling of Nsites sites
        idxSamples = np.append(idxSamples,idxi)

    #idxSamples = idxSamples.astype(np.int32)
    ytrain = ytrain[idxSamples,:]
    xtrain = xtrain[idxSamples,:,:]
    
    return xtrain,ytrain


def get_dataset_all_states_SplitNum(States,sequence,split_train=0.8,uniformisation=False):
    
    Nstates = States.shape[0]
    
    idx1 = np.where(States.sum(axis=0)>0)[0]
    N1 = len(idx1)
    States = States[:,idx1]
    
    print 'Input Sequence'
    Seq = sequence[idx1]
    SeqOHE = np.zeros((4,len(Seq[0]),N1))
    for i in range(N1):
        for k in range(len(Seq[i])):
            if Seq[i][k]=='A':
                m=0
            if Seq[i][k]=='C':
                m=1
            if Seq[i][k]=='T':
                m=2
            if Seq[i][k]=='G':
                m=3

            SeqOHE[m,k,i] = 1
    
    print 'Split the data in order to generate batch of samples'
    
    print 'Transpose matrix in order to get the batch dimension first'
    States = np.transpose(States,(1,0)) #Sequence x State
    SeqOHE = np.transpose(SeqOHE,(2,1,0)) #Sequences x 200 bp x 4 nucl.
    
    
    print 'Split the data set in train & test set'
    idxR = np.random.permutation(States.shape[0])
    N_train = int(States.shape[0]*split_train)
    xtrain = SeqOHE[idxR[:N_train]]
    ytrain = States[idxR[:N_train]]
    
    xtest = SeqOHE[idxR[N_train:]]
    ytest = States[idxR[N_train:]]
    
    
    if uniformisation==True:
        xtrain,ytrain = Uniformisation(xtrain,ytrain)
    
    return xtrain,ytrain,xtest,ytest





#from keras.layers import LSTM, Convolution1D, Flatten, Dropout, Dense
#from keras.layers.embeddings import Embedding
#from keras.models import Sequential

#max_review_length = 1600
#embedding_vecor_length = 300
#model = Sequential()
#model.add(Convolution1D(128,1, border_mode='same'))
#model.add(Pooling)
#model.tanh

#model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
#model.add(Convolution1D(64, 3, border_mode='same'))
#model.add(Convolution1D(32, 3, border_mode='same'))
#model.add(Convolution1D(16, 3, border_mode='same'))
#model.add(Flatten())
#model.add(Dropout(0.2))
#model.add(Dense(180,activation='sigmoid'))
#model.add(Dropout(0.2))
#model.add(Dense(1,activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
