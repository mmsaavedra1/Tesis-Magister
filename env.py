FILENAME = "Constante"
SCALER = 1
MIP_GAP = 0.05
TIME_LIMIT = 99999
DELTA = 9
REPLICS = 1

PERIODS=30
TIMES=500

import json
import random
from M2 import *
#K, F, a = read_sheet( f"~/Desktop/Produccion-Tesis/Input/Constante.xlsx", "Patrones")
#diccionario = {}
## 100 Replicas
#for k in range(1, 101):
#    diccionario[k] = {}
#    # 500 Tiempos de Simulacion
#    for i in range(1, 501):
#        diccionario[k][i] = {}
#        # 14 Productos a vender
#        for f in F:
#            diccionario[k][i][f] = random.random()
#with open('random.json', 'w') as fp:
#    json.dump(diccionario, fp)

with open('random.json', 'r') as fp:
    ERROR_DDA = json.load(fp)

#ERROR_DDA = pd.read_excel('random.xlsx', engine='openpyxl').to_dict()