from flask import Flask, jsonify

import pandas as pd
import glob

directorio = './data/*'
archivos_csv = glob.glob(directorio + '.csv')

app = Flask(__name__)

@app.route('/data', methods=['GET'])
def get_data():
    # Carga el dataframe aquí. Por ejemplo, estoy cargando un CSV.
    lista_df = []

    # Lee cada archivo CSV y lo agrega a la lista
    for filename in archivos_csv:
        df = pd.read_csv(filename)
        lista_df.append(df)

    # Concatena todos los dataframes en la lista
    df_concatenado = pd.concat(lista_df, ignore_index=True)

    # Convierte el dataframe a un diccionario para poder usar jsonify
    data = df_concatenado.to_dict('records')
    #print(type(data))
    
    return jsonify(data)

    
    #return data
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
