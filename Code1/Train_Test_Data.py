
import numpy as np
from os.path import sep
import pandas as pd
import sys
sys.path.append('../')
from Code1.project_configuration import get_parameter


# Function for getting Training and Testing data from the data frame

def train_test(z,z2,Target):
        
    file_name= 'spectra.csv'  #spectral_data_new_z_z1_10_600010_6000.csv
    spectral_data_dir = get_parameter('spectral_data_dir')

    df = pd.read_csv(spectral_data_dir+sep+file_name, index_col=0)
    df = df.loc[(df['RPM'] == 4000)&(df['Mic ID'] == 1)]

    # Training data set Creation
    df_train=df.loc[(df['Tag'] == z) & (df['Status'] == 'OK') & (df['Machine ID'].isin([1, 2, 11, 12,13,14,15,16,17,18,19,20,21,24,25]))]

    # Testing data set Creation
    df_test=df.loc[(df['Tag'] == z2) & (df['Machine ID'].isin([6,7,8,9,10,23,26,27,28,29,30]))] #,26,27,28,29,30
    
    # Spectral Column Declearation
    first_spectra_column = df_train.columns.to_list().index('0') 


    # Training and Testing data creation

    X_train = np.array( df_train.iloc[:, first_spectra_column:].to_numpy(), dtype=float )
    X_train = np.log10(X_train)
    y_train = df_train[Target].to_numpy()
    machine_ids_train=df_train['Machine ID']


    X_test = np.array( df_test.iloc[:, first_spectra_column:].to_numpy(), dtype=float )
    X_test = np.log10(X_test)
    y_test = df_test[Target].to_numpy()
    machine_ids_test=df_test['Machine ID']
    y_test=np.where(y_test=='OK',0,1)
   

    return X_train, y_train,machine_ids_train,X_test,y_test,machine_ids_test