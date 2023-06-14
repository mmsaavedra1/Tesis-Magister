import sys        
import random
import warnings
import time
import numpy as np
import pandas as pd
import collections, functools, operator

from env import *

from M2 import *
from simulacion_model_2 import *

# Parameters of system
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Parametros de la simulacion
    filename = FILENAME
    scaler = SCALER
    mip_gap = MIP_GAP
    time_limit = TIME_LIMIT
    delta = DELTA


    ################# CREAR SIMULACIONES ###########a######

    case_politica_1 = 1                 # P_t == P_t-1
    replics = REPLICS                         # r

    # Escenario_3
    remaining_days = 1  # t2
    periods = PERIODS        # t3
    times = TIMES         # t4
    simulation_3 = Simulation(
        model=MODEL, filename=filename, mip_gap=mip_gap, time_limit=time_limit,
        scaler=scaler, periods=periods, delta=delta, times=times, replics=replics,
        case=case_politica_1, it_case=5, remaining_days=remaining_days, error_dda=ERROR_DDA, _print=LOGGER, n_escenario=3)


    # Correr todas las simulaciones
    simulation_3.run_replics()
    simulation_3.save_to_pickle(n_escenario=3)