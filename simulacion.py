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
        self.simulacion_Periods = periods
        self.case = case
        self.delta = delta
        self.times = times
        self.it_case = it_case
        self.replics = replics
        self.remaining_days = remaining_days
        self.error_dda = error_dda    
        self._print = _print   
        self.n_escenario = n_escenario
        
        # Variables optimizacion
        self.opti_produccion = {}
        self.opti_precio = {}
        self.opti_patron = {}
        self.opti_inventario_final = {}
        self.opti_demanda = {}
        self.opti_merma = {}
        self.opti_inventario_inicial = {}

        # Resumen de promedio de replicas
        self.simulacion_X = {}
        self.simulacion_S = {}
        self.simulacion_S_inicial = {}
        self.simulacion_W0 = {}
        self.simulacion_D= {}
        self.simulacion_P = {}
        self.simulacion_L = {}
        self.simulacion_D_real = {}
        self.simulacion_sales = {}
        self.simulacion_Prod = {}
        self.simulacion_Prod_w = {}
        self.out_obj_val = {}
        self.S0 = {}

        # Metricas de interes
        self.objective_value = {}
        self.ingresos = {}
        self.costo_corte = {}
        self.costo_inventario = {}
        self.costo_merma = {}

    def get_parameters(self):
        return read_all_data(
            f"Input/{self.filename}.xlsx",
            periods=self.simulacion_Periods,
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

        # 2º Manejo de tiempo entre Simulacion|Optimizacion
        t_opti = {}
        t_valor = 0
        for t in range(1, 1000):
            t_opti[t] = t_valor+1
            t_valor += 1
            if t_valor == self.remaining_days:
                t_valor = 0

        # 3º Cargar logger
        if self._print:
            archi1=open(f"Logger/Simulacion {self.n_escenario}.txt","w")

        print("\nReplica:", r)
        # 4º Simulacion
        for t in range(1, self.times + 1):
            # 4.1º Ver decision de optimizacion
            if t_opti[t] == 1:
                # 4.2º Optimizar
                production, price, pattern_use, inv_final, demanda, merma, W0 = self.model(
                    string_input=self.filename,
                    mip_gap=self.mip_gap,
                    time_limit=self.time_limit,
                    scaler=self.scaler,
                    periods=self.simulacion_Periods,
                    case=self.case, 
                    iterate=True, 
                    init_S0=S0,
                    save=False, 
                    delta_=self.delta,
                    loggin=0)
            
                print("Optimizacion", t)

                for _ in range(self.remaining_days):       
                    # 4.2 Ordenar decisiones en el tiempo
                    for k in K:
                        self.simulacion_X[k, t, r] = pattern_use[(k, t_opti[t])]                                   # OPCION 1: DECISION CORTE
                    
                    for f in F:
                        # Produccion
                        self.simulacion_Prod[f, t, r] = sum(a[f][k]* self.simulacion_X[k, t, r] for k in K)                   #OPCION 1: VARIABLE SISTEMA PRODUCCION

                        # Precios
                        self.simulacion_P[f, t, r] = price[(f, t_opti[t])]                                         # OPCION 1: DECISION PRECIOS CONTINUOS
                        #self.simulacion_P[f, t, r] = round(price[(f, t_opti[t])], 2)                              # OPCION 2: DECISION PRECIOS CONTINUOS RENDONDEADOS

                        # Generar Demanda en funcion del Precio
                        self.simulacion_D[f, t, r] = alfa[f][t_opti[t]]-(beta[f][t_opti[t]]*self.simulacion_P[f, t, r])       # OPCION 1: VARIABLE SISTEMA DEMANDA

                        # Venta de producto en t que vences en un
                        for i in range(0, 30):
                            for u in range(0, delta-1):
                                self.W0[f, t+i, t+i+u, r] = W0[f, t_opti[t+i], t_opti[t+i]+u]                       # OPCION 1: VARIABLES SISTEMA INV INICIAL
            

                
            # TERMINAL LOGGER 
            print()
            print("*"*10)
            print(f"Tiempo {t}")
            print("*"*10)

            # 4.3º Actualizar sistema
            for f in F:
                #----------------------SIMULACION VS OPTIMIZACION-----------------------------#
                if f in ['Entero']:
                    print(f"(W0) Inv para vender en {t} que vence hasta {t+delta-1}")
                    print("   {:<10} {:<2} {:<2} {:<20} {:1} {:<20}".format("Producto", "t", "u", "Simulacion", "-", "Optimizacion"))
                    inv_inicial_opti = 0
                    for u in range(0, delta-1):
                        inv_inicial_opti += self.W0[f, t, t+u, r]
                        print("W0 {:<10} {:<2} {:<2} {:<20} {:1} {:<20}".format(f, t, t+u, S0[f][u+1], "-", self.W0[f, t, t+u, 1]))
                    print()
                #----------------------SIMULACION VS OPTIMIZACION-----------------------------#

                # 4.3.0º Respaldar inventarios inciales
                self.S_inicial[f, t, r] = sum(list(S0[f].values()))

                # 4.3.1º Generar realización de demanda para cambiar el sistema
                D_error = 0
                #D_error = -0.6 + self.error_dda[str(r)][str(t)][f]*1.2                         # +- 60% de error         
                this_demand = min(alfa[f][1], max((1 + D_error)*self.simulacion_D[f, t, r], 0))            # +- 60% de error 
                self.simulacion_D_real[(f, t, r)] = this_demand

                # ************* LOG *************
                if f == "Entero" and self._print:
                    logger_inventario_inicial(archi1, f, t, r, self.delta, self.S_inicial, S0, self.simulacion_Prod)
                # ************* LOG *************
            
                # 4.3.2º Inventario inicial en funcion de la produccion
                S0[f][self.delta] += self.simulacion_Prod[f, t, r]
    
                # ************* LOG *************
                if f == "Entero" and self._print:
                    logger_inventario_actualizado(archi1, self.delta, f, t, S0)
                # ************* LOG *************

                # 4.3.3º Actualizacion de variables de estado del sistema 
                unserved_demand, this_L, S0 = refresh_S0(S0, f, self.delta, this_demand)
                self.simulacion_sales[f, t, r] = self.simulacion_D_real[(f, t, r)] - unserved_demand
                self.S[f, t, r] = sum(list(S0[f].values()).copy())
                self.simulacion_L[f, t, r] = this_L

                 # ************* LOG *************
                if f == "Entero" and self._print:
                    logger_ventas(archi1, f, t, r, self.simulacion_D, self.error_dda, self.simulacion_D_real, self.simulacion_sales, self.simulacion_L, self.simulacion_P, S0)
                # ************* LOG *************


                #----------------------SIMULACION VS OPTIMIZACION-----------------------------#
                if f in ['Entero']:
                    print("-"*100)
                    print("                       {:<25} {:<20} {:1} {:<20}".format("Producto", "Simulacion", "-", "Optimizacion"))
                    print("Inv incial (t={:<2}) - {:<25}: {:<20} {:1} {:<20}".format(t, f, self.S_inicial[f, t, r], "-", inv_inicial_opti))     # S0 vs W0
                    print("Demanda    (t={:<2}) - {:<25}: {:<20} {:1} {:<20}".format(t, f, self.D[f, t, r], "-", demanda[f, t]))                # D_sim vs D_opti (alfa-beta*P)
                    print("Producc    (t={:<2}) - {:<25}: {:<20} {:1} {:<20}".format(t, f, self.simulacion_Prod[f, t, r], "-", production[f, t]))          # prod_sim vs prod_opti (∑ax)
                    print("Merma      (t={:<2}) - {:<25}: {:<20} {:1} {:<20}".format(t, f, self.simulacion_L[f, t, r], "-", merma[f, t]))                  # L_sim vs L_opti
                    print("-"*100)

                #if (f in ['Entero']) and (t < self.times):
                #    print(f"(S) Inv final para vender en {t} que vence hasta {t+delta-1}")
                #    print("   {:<10} {:<2} {:<2} {:<20} {:1} {:<20}".format("Producto", "t", "u", "Simulacion", "-", "Optimizacion"))
                #    for i in range(0, delta):
                #        print("S {:<10} {:<2} {:<2} {:<20} {:1} {:<20}".format(f, t, t+i, S0[f][i+1], "-", W0[f, t_opti[t]+1, t_opti[t]+1+i]))
                #    print()
                #----------------------SIMULACION VS OPTIMIZACION-----------------------------#

                # 4.4º Almacenar metricas de producto           
                if t <= self.times:
                        self.ingresos[f, t, r] = self.simulacion_P[f, t, r] * self.simulacion_sales[(f, t, r)]
                        self.costo_inventario[f, t, r] = h[f][1] * self.S[f, t, r]               # 0.478 $/caja
                        self.costo_merma[f, t, r] = c_merma[f][1] * self.simulacion_L[f, t, r]              # 0.956 $/caja
            
            # 4.5º Almacenar metricas de sistema           
            if t <= self.times:
                for k in K:
                    self.costo_corte[k, t, r] = c[k] * self.simulacion_X[k, t, r]

                self.objective_value[t, r] =\
                    quicksum(self.ingresos[f, t, r] for f in F).getValue() \
                    - quicksum(self.costo_inventario[f, t, r] for f in F).getValue() \
                    - quicksum(self.costo_merma[f, t, r] for f in F).getValue() \
                    - quicksum(self.costo_corte[k, t, r] for k in K).getValue()

        if self._print:
            archi1.close()

        print(f"\nUtilidad replica {r}: {sum(list(self.objective_value.values()))}\n")
    

    @timeit
    def run_replics(self):
        for r in range(1, self.replics+1):
            self.run(r)


    def save_to_pickle(self, n_escenario):
        directorio = f"~/Desktop/Produccion-Tesis/Resultados/Escenario {n_escenario}/"
        pd.Series(self.simulacion_X).rename_axis(['k', 't', 'r']).reset_index(name='value').to_excel(directorio+"X.xlsx", engine='openpyxl')
        pd.Series(self.S).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"S.xlsx", engine='openpyxl')
        pd.Series(self.S_inicial).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"S_inicial.xlsx", engine='openpyxl')
        pd.Series(self.W0).rename_axis(['f', 't', 'u', 'r']).reset_index(name='value').to_excel(directorio+"W0.xlsx", engine='openpyxl')
        pd.Series(self.D).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"D.xlsx", engine='openpyxl')
        pd.Series(self.simulacion_P).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"P.xlsx", engine='openpyxl')
        pd.Series(self.simulacion_L).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"L.xlsx", engine='openpyxl')
        pd.Series(self.simulacion_D_real).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"D_real.xlsx", engine='openpyxl')
        pd.Series(self.simulacion_sales).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"sales.xlsx", engine='openpyxl')
        pd.Series(self.simulacion_Prod).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"prod.xlsx", engine='openpyxl')
        pd.Series(self.objective_value).rename_axis(['t', 'r']).reset_index(name='value').to_excel(directorio+"objective_value.xlsx", engine='openpyxl')
        pd.Series(self.ingresos).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"ingresos.xlsx", engine='openpyxl')
        pd.Series(self.costo_inventario).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"costo_inventario.xlsx", engine='openpyxl')
        pd.Series(self.costo_merma).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"costo_merma.xlsx", engine='openpyxl')
        pd.Series(self.costo_corte).rename_axis(['k', 't', 'r']).reset_index(name='value').to_excel(directorio+"costo_corte.xlsx", engine='openpyxl')
