import sys        
import random
import warnings
import time
import numpy as np
import pandas as pd
import collections, functools, operator

from env import *

from M2 import *
from simulacion import *

# Parameters of system
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Parametros de la simulacion
    filename = FILENAME
    scaler = SCALER
    mip_gap = MIP_GAP
    time_limit = TIME_LIMIT
    delta = DELTA


    ################# CREAR SIMULACIONES #################
    case_politica_1 = 1                 # P_t == P_t-1
    replics = REPLICS                         # r

    # Escenario_1
    remaining_days = 30         # t2
    periods = PERIODS           # t3
    times = TIMES               # t4
    simulation_1 = Simulation(
        model=model_2, filename=filename, mip_gap=mip_gap, time_limit=time_limit,
        scaler=scaler, periods=periods, delta=delta, times=times, replics=replics,
        case=case_politica_1, it_case=5, remaining_days=remaining_days, error_dda=ERROR_DDA, _print=True, n_escenario=1)

    # Correr todas las simulaciones
    simulation_1.run_replics()
    simulation_1.save_to_pickle(n_escenario=1)