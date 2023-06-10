import sys        
import random
import warnings
import time
import numpy as np
import pandas as pd
import collections, functools, operator

import pickle

from M2 import *
from logger_simulacion import *

# Parameters of system
warnings.filterwarnings("ignore")


def refresh_S0(S0, f, delta, demand):
    i = 1
    while demand >= 0 and i <= delta:
        inv = S0[f][i]
        subtract = min([inv, demand])
        S0[f][i] -= subtract
        demand -= subtract
        i += 1
    merma = S0[f][1]
    for i2 in range(1, delta+1):
        if S0[f][i2+1] < 0:
            S0[f][i2+1] = 0
        S0[f][i2] = S0[f][i2+1]
    return demand, merma, S0


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


class Simulation:
    def __init__(self, model, filename, mip_gap, time_limit, scaler, periods, case, delta, times, it_case, replics, remaining_days, error_dda, _print=True, n_escenario=None):
        self.model = model
        self.filename = filename
        self.mip_gap = mip_gap
        self.time_limit = time_limit
        self.scaler = scaler
        self.periods = periods
        self.case = case
        self.delta = delta
        self.times = times
        self.it_case = it_case
        self.replics = replics
        self.remaining_days = remaining_days
        self.error_dda = error_dda    
        self._print = _print   
        self.n_escenario = n_escenario
        
        # Resumen de promedio de replicas
        self.X = {}
        self.S = {}
        self.S_inicial = {}
        self.D = {}
        self.P = {}
        self.L = {}
        self.D_real = {}
        self.sales = {}
        self.prod = {}
        self.prod_w = {}
        self.out_obj_val = {}

        # Metricas de interes
        self.objective_value = {}
        self.ingresos = {}
        self.costo_corte = {}
        self.costo_inventario = {}
        self.costo_merma = {}

    def get_parameters(self):
        return read_all_data(
            f"Input/{self.filename}.xlsx",
            periods=self.periods,
            scaler=self.scaler,
            print_out=False,
            fix_alfa=False,
            q_mult=1,
            alfa_mult=1,
            beta_mult=1,
            res_mult=1,
            S0_mult=1,
            delta_=self.delta
        )

    def run(self, r):
        # 1º Setear parametros para analisiss
        read_filename = f"~/Desktop/Produccion-Tesis/Input/{self.filename}.xlsx"
        aux, aux2, S0 = read_sheet(read_filename, "Inventarios Iniciales", num_columns=True)
        K, F, a = read_sheet(read_filename, "Patrones")
        F, T, T_delta, T_0, T_0_delta, K, alfa, beta, delta, S0, a, h, c, q, \
        c_merma, p_constante, p_compat, v_constante = self.get_parameters()

        pesos = {
        'Entero': 1.860,
        'Medio Pollo': 0.930,
        'Medio Pollo Superior': 1.080,
        'Medio Pollo Inferior': 0.780,
        'Cuarto de Pollo Superior': 0.540,
        'Muslo Entero': 0.390,
        'Pechuga Completa': 0.900,
        'Ala Completa': 0.090,
        'Media Pechuga': 0.450,
        'Blanqueta': 0.055,
        'Alón': 0.027,
        'Punta': 0.008,
        'Jamoncito': 0.140,
        'Medio Muslo': 0.250,
        }

        # 2º Seteo de parametros de simlacion
        n_opti = self.remaining_days

        # 3º Cargar logger
        if self._print:
            archi1=open(f"Logger/Simulacion {self.n_escenario}.txt","w")

        print("\nReplica:", r)
        # 4º Simulacion
        for t in range(1, self.times + 1):
            # 4.1º Ver decision de optimizacion
            if n_opti == self.remaining_days:
                n_opti = 0
                # 4.2º Optimizar
                demands, production, price, pattern_use, objective_function, runtime, inventario, demanda, w = self.model(
                    string_input=self.filename,
                    mip_gap=self.mip_gap,
                    time_limit=self.time_limit,
                    scaler=self.scaler,
                    periods=self.periods,
                    case=self.case, 
                    iterate=True, 
                    init_S0=S0,
                    save=False, 
                    delta_=self.delta,
                    loggin=0)
            
                print("Optimizacion", t)

                # 4.2 Ordenar decisiones en el tiempo
                for n in range(self.remaining_days):
                    for k in K:
                        self.X[k, t+n, r] = pattern_use[(k, n+1)]
                       
                    for f in F:
                        # Precios
                        #self.P[f, t+n, r] = price[(f, n+1)]                                            # OPCION 1: PRECIOS CONTINUOS
                        self.P[f, t+n, r] = round(price[(f, n+1)], 2)                                   # OPCION 2: PRECIOS CONTINUOS RENDONDEADOS
                        
                        # Produccion
                        self.prod[f, t+n, r] = sum(a[f][k]* self.X[k, t+n, r] for k in K)                                               #OPCION 1: Produccion
                        #self.prod[f, t+n, r] = w[w.f == f][w.s == n+1].loc[(w.t >= max(n+1-delta, 1)), ['value']].sum()['value']       #OPCION 2: Produccion Determinista

                        # Generar Demanda en funcion del Precio
                        self.D[f, t+n, r] = alfa[f][n+1]-(beta[f][n+1]*self.P[f, t+n, r])       # OPCION 2: Calculo Demanda según decisiones


            # 4.3º Actualizar estados
            for f in F:
                # 4.3.0º Respaldar inventarios inciales
                self.S_inicial[f, t, r] = sum(list(S0[f].values()))

                # 4.3.1º Generar realización de demanda para cambiar el sistema
                D_error = 0
                D_error = -0.6 + self.error_dda[str(r)][str(t)][f]*1.2                         # +- 60% de error         
                this_demand = min(alfa[f][1], max((1 + D_error)*self.D[f, t, r], 0))            # +- 60% de error 
                self.D_real[(f, t, r)] = this_demand

                # ************* LOG *************
                if f == "Entero" and self._print:
                    logger_inventario_inicial(archi1, f, t, r, self.delta, self.S_inicial, S0, self.prod)
                # ************* LOG *************
            
                # 4.3.2º Inventario inicial en funcion de la produccion
                S0[f][self.delta] += self.prod[f, t, r]
    
                # ************* LOG *************
                if f == "Entero" and self._print:
                    logger_inventario_actualizado(archi1, self.delta, f, t, S0)
                # ************* LOG *************

                # 4.3.3º Actualizacion de variables de estado del sistema 
                unserved_demand, this_L, S0 = refresh_S0(S0, f, self.delta, this_demand)
                self.sales[f, t, r] = self.D_real[(f, t, r)] - unserved_demand
                self.S[f, t, r] = sum(list(S0[f].values()).copy())
                self.L[f, t, r] = this_L

                 # ************* LOG *************
                if f == "Entero" and self._print:
                    logger_ventas(archi1, f, t, r, self.D, self.error_dda, self.D_real, self.sales, self.L, self.P, S0)
                # ************* LOG *************

                # 4.4º Almacenar metricas de producto           
                if t <= self.times:
                        self.ingresos[f, t, r] = self.P[f, t, r] * self.sales[(f, t, r)]
                        self.costo_inventario[f, t, r] = h[f][1] * self.S[f, t, r]               # 0.478 $/caja
                        self.costo_merma[f, t, r] = c_merma[f][1] * self.L[f, t, r]              # 0.956 $/caja
            
            # 4.5º Almacenar metricas de sistema           
            if t <= self.times:
                for k in K:
                    self.costo_corte[k, t, r] = c[k] * self.X[k, t, r]

                self.objective_value[t, r] =\
                    quicksum(self.ingresos[f, t, r] for f in F).getValue() \
                    - quicksum(self.costo_inventario[f, t, r] for f in F).getValue() \
                    - quicksum(self.costo_merma[f, t, r] for f in F).getValue() \
                    - quicksum(self.costo_corte[k, t, r] for k in K).getValue()

            # 4.6º Actualizar valores para terminar la semana
            n_opti += 1

        if self._print:
            archi1.close()
    
    @timeit
    def run_replics(self):
        for r in range(1, self.replics+1):
            self.run(r)

    def save_to_pickle(self, n_escenario):
        directorio = f"~/Desktop/Produccion-Tesis/Resultados/Escenario {n_escenario}/"
        pd.Series(self.X).rename_axis(['k', 't', 'r']).reset_index(name='value').to_excel(directorio+"X.xlsx", engine='openpyxl')
        pd.Series(self.S).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"S.xlsx", engine='openpyxl')
        pd.Series(self.S_inicial).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"S_inicial.xlsx", engine='openpyxl')
        pd.Series(self.D).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"D.xlsx", engine='openpyxl')
        pd.Series(self.P).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"P.xlsx", engine='openpyxl')
        pd.Series(self.L).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"L.xlsx", engine='openpyxl')
        pd.Series(self.D_real).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"D_real.xlsx", engine='openpyxl')
        pd.Series(self.sales).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"sales.xlsx", engine='openpyxl')
        pd.Series(self.prod).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"prod.xlsx", engine='openpyxl')
        pd.Series(self.objective_value).rename_axis(['t', 'r']).reset_index(name='value').to_excel(directorio+"objective_value.xlsx", engine='openpyxl')
        pd.Series(self.ingresos).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"ingresos.xlsx", engine='openpyxl')
        pd.Series(self.costo_inventario).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"costo_inventario.xlsx", engine='openpyxl')
        pd.Series(self.costo_merma).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"costo_merma.xlsx", engine='openpyxl')
        pd.Series(self.costo_corte).rename_axis(['k', 't', 'r']).reset_index(name='value').to_excel(directorio+"costo_corte.xlsx", engine='openpyxl')
