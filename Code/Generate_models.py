
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics, linear_model, neighbors, model_selection
from tensorflow.keras.callbacks import EarlyStopping
class Generate_model:
    def __init__(self,data,input_features,output,test_size,n_steps_in,n_steps_out,dType,categorical_features,normalize_method=preprocessing.MinMaxScaler()):
        self.data=pd.DataFrame(data)
        self.input_features=input_features
        self.output=output
        self.test_size=test_size
        self.normalize_method=normalize_method
        self.n_steps_in=n_steps_in
        self.n_steps_out=n_steps_out
        self.dType=dType
        self.categorical_features=categorical_features
    


    def split_sequence(self,sequence_x,sequence_y):
        X=[]
        y=[]
        for i in range(len(sequence_x)):
            # find the end of this pattern
            end_ix = i + self.n_steps_in
            out_end_ix = end_ix + self.n_steps_out
        # check if we are beyond the sequence
            if out_end_ix > len(sequence_x):
                    break
        # gather input and output parts of the pattern
            seq_x= sequence_x[i:end_ix] 
            seq_y=sequence_y[end_ix:out_end_ix] if(self.dType=="sequence") else sequence_y[(end_ix-1):out_end_ix-1]
            X.append(seq_x)
            y.append(seq_y) 
        return np.float32(X), np.float32(y)

    def build_model(self,model,epochs=10):
        self.data.replace('?', np.nan, inplace=True)
        self.data.dropna(inplace = True)
        l=[]
        for colums in  self.categorical_features:
            le = LabelEncoder()
            self.data[colums]=le.fit_transform(self.data[colums])
            l.append(le)
        x = self.data[self.input_features]
        y = self.data[self.output]
        x_normalize = self.normalize_method
        y_normalize = self.normalize_method
        x_norm = x_normalize.fit_transform(x)
        
        if(self.dType=="sequence"):
             y_norm=np.float32(y_normalize.fit_transform(np.array(y).reshape(-1,1)))
        else:
             y_norm=y
        x_norm, y_norm=self.split_sequence(x_norm, y_norm)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x_norm, y_norm, test_size = self.test_size, random_state = 42)
        #x_train, y_train=self.split_sequence(x_train, y_train)
        #x_test, y_test=self.split_sequence(x_test, y_test)
        if(self.dType=="sequence"):
            y_train=np.reshape(y_train,(len(y_train),self.n_steps_out))
            y_test=np.reshape(y_test,(len(y_test),self.n_steps_out))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(np.float32(x_train), np.float32(y_train),batch_size=64,epochs=epochs,validation_data=(x_test, y_test))

        return model,x_train,y_train,x_test,y_test,l,(x_normalize,y_normalize)
   



