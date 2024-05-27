######################################################
import pandas  as pd
import numpy   as np

import subprocess
import math
import sys


from os               import listdir, path
from tensorflow       import keras
from tensorflow.keras import layers
######################################################

def cargar_archivo(nombre_archivo):

    # Obtiene la extensión del archivo
    extension = path.splitext(nombre_archivo)[1]

    # Carga el archivo dependiendo de su extensión
    if extension == '.csv':
        df = pd.read_csv(nombre_archivo, parse_dates=['Fecha'])
        df['id_Cliente'] = int(''.join(caracter for caracter in nombre_archivo if caracter.isdigit()))
    elif extension == '.xlsx':
        df = pd.read_excel(nombre_archivo, parse_dates=['Fecha'])
        df['id_Cliente'] = int(''.join(caracter for caracter in nombre_archivo if caracter.isdigit()))
    else:
        print("Formato de archivo no soportado")
        return None
    cols = ['Active_energy', 'Reactive_energy', 'Voltaje_FA', 'Voltaje_FC']
    groups = ['id_Cliente', df.Fecha.dt.date]
    df = df.groupby(groups)[cols].sum()
    return df.reset_index().copy()

######################################################

def redondear_al_multiplo_de_4_mas_cercano(numero):
    numero_redondeado = 4 * math.floor(numero / 4)
    return numero_redondeado
######################################################

def create_sequences(values, time_steps):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)
######################################################

def train_model(x_train):
    # Build the autoencoder model
    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    #model.summary()

    # Train the autoencoder model
    history = model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=4,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="min")
        ],
        verbose=0
    )
    
    return history, model
######################################################

def work_flow(df, variable, ventana=32, threshold='mean'):
    
    df_train = df[[variable]].copy()
    
    #TIME_STEPS
    TIME_STEPS = redondear_al_multiplo_de_4_mas_cercano(ventana)        
    
    #Normalized Train Data
    training_mean = df_train.mean()
    training_std  = df_train.std()
    
    df_training_value = (df_train - training_mean) / training_std
    
    #Create sequences for training data
    x_train = create_sequences(df_training_value.values, TIME_STEPS)    
    
    #Train Model
    history, model = train_model(x_train)

    # Get predicted data from the autoencoder
    x_train_pred = model.predict(x_train, verbose=0)
    
    # Calculate the Mean Absolute Error (MAE) loss for training samples
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
    train_mae_loss = train_mae_loss.reshape((-1))

    # Set the threshold for anomaly detection
    
    if threshold == 'mean':
        threshold = np.mean(train_mae_loss)
    else: threshold = threshold
    
    # Detect anomalies based on the threshold
    anomalies = (train_mae_loss > threshold) 

    df.loc[:, 'Anomaly'] = np.append(np.full(len(df_train) - len(anomalies), False), anomalies)
    df.loc[:, 'Variable'] = variable
    df.loc[:, 'Valor'] = df.loc[:, variable]
    df['Cliente:'] =  'Cliente ' + df['id_Cliente'].astype(str)
    
    df_return = df[['Cliente:','Fecha', 'Variable', 'Valor','Anomaly']].copy()
    
    file = './data/cliente_'+str(df.id_Cliente[0])+'_'+variable+'.csv'
    df_return.to_csv(file, index=False)
    
    #if ec2:
    #    send_file_to_ec2('ubuntu', 'api_bi.pem', '3.16.165.87', file, '/home/ubuntu/' + file[2:])
    
    return df_return
#############################################

def main():
    
    df       = cargar_archivo(sys.argv[1])
    variable = sys.argv[2]
    
    return work_flow(df, variable)

if __name__ == "__main__":
    main()


    
