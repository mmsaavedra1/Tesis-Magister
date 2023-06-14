import os
import sys    
import time
import random
import warnings
import numpy as np
import pandas as pd


# Parameters of system
warnings.filterwarnings("ignore")
from M2 import *
from M1 import *
from nuevas_funciones import *

# Decorador para medir tiempo de ejecución
def timeit(func):
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

scaler = 1 # Factor de conversion para achicar modelo. (e.g. cajas de 200 pollos).
mip_gap = 0.05 # Gap de optimalidad.
time_limit = 99999 # Máximo tiempo de resolución [s].
periods = 30 # Horizonte temporal.  Si es None, por default es 90.
save = False # Decision si guardar output de modelo.
model_name = "M2"  # Nombre del modelo
model = model_2 # Mdelo obtenido del dict
case = 1 # Caso de estudio

# 2º Archivos de input del modelo
filename = "Constante"  

# 3º Experimentos
# IMPORTANTE: ¡Cuando delete=True, se borran las corridas anteriores de experimentos dentro de la carpeta!    
# experiment -> [Q, Alfa, Beta, Res, S0 o  Periods]
experiment_names = ["Q"]
# experiment_names = ["Res", "S0", "Periods", "Q", "Alfa", "Beta"]

# 4º Multiplicadores para las variables de analisis
a_mults = [30 + 10*i for i in range(13)] # Multiplicador que va variando para Periods
b_mults = [0.5 + i*0.1 for i in range(11)] # Multiplicador que va variando para Q, alfa, beta
c_mults = [i for i in range(5)] # Multiplicador que va variando para Res, S0
#q_mults = [0.5 + i*0.1 for i in range(15)] # Multiplicador que va variando para Q
q_mults = [1 + i*0 for i in range(15)] # Multiplicador que va variando para Q

# 5º Ejecutar todos los case de multiplicadores, segun experimento elegido en experiment_names
exp_mults = {"Q": q_mults, "Alfa": b_mults, "Beta": b_mults, "Res": c_mults, "S0": c_mults, "Periods": a_mults}


# Ejecutar modelo - Pruebas de funcionalidad del modelo
@timeit
def main():
    """
    return model_2(
        string_input=filename,
        mip_gap=mip_gap,
        time_limit=time_limit,
        periods=periods,
        scaler=scaler,
        case=case, 
        iterate=False,
        save=True, 
        loggin=1,
        delta_=9)
    """

    return model_1(
        string_input=filename,
        mip_gap=mip_gap,
        time_limit=time_limit,
        periods=periods,
        scaler=scaler,
        case=case,
        iterate=False,
        loggin=1,
        delta_=9
    )

main()
pickle_to_excel(filename, 0, model_name)

