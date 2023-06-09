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
        self.opti_inventario_inicial = {}
        self.opti_inventario_final = {}
        self.opti_demanda = {}
        self.opti_merma = {}
        self.opti_demanda_perecible = {}

        # Resumen de promedio de replicas
        self.simulacion_X = {}
        self.simulacion_S_inicial = {}
        self.simulacion_S_final = {}
        self.simulacion_W0 = {}
        self.simulacion_D= {}
        self.simulacion_P = {}
        self.simulacion_L = {}
        self.simulacion_D_real = {}
        self.simulacion_Sales_perecible = {}
        self.simulacion_Sales = {}
        self.simulacion_Prod = {}
        self.simulacion_Prod_w = {}
        self.simulacion_S0 = {}

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


    def refresh_S0(self, S0, f, delta, demand, t, r):
        i = 1
        while demand >= 0 and i <= delta:
            inv = S0[f][i]
            subtract = min([inv, demand])
            S0[f][i] -= subtract
            demand -= subtract
            self.simulacion_Sales_perecible[f, t, t+i-1, r] = subtract  # Almacenar demanda
            i += 1
        merma = S0[f][1]

        for i2 in range(1, delta+1):
            if S0[f][i2+1] < 0:
                S0[f][i2+1] = 0
            S0[f][i2] = S0[f][i2+1]
        return demand, merma, S0


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
        
        # 4º Simulacion
        for t in range(1, self.times + 1):
            # 4.1º Ver decision de optimizacion
            if t_opti[t] == 1:
                # 4.2º Optimizar
                opti_produccion, opti_precio, opti_patron, opti_inv_final, opti_demanda, opti_merma, opti_inv_inicial, opti_demanda_perecible = self.model(
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
        
                print("Modelo 1-Optimizacion", t)

                for n in range(self.remaining_days):
                    # 4.3º Guardar los valores en la simulacion
                    for k in K:
                        self.simulacion_X[k, t+n, r] = opti_patron[(k, t_opti[t+n])]

                    for f in F:
                        self.simulacion_Prod[f, t+n, r] = sum(a[f][k]*self.simulacion_X[k, t+n, r] for k in K)
                        self.simulacion_P[f, t+n, r] = opti_precio[(f, t_opti[t+n])]
                        self.simulacion_D[f, t+n, r] = alfa[f][t_opti[t+n]]-(beta[f][t_opti[t+n]]*self.simulacion_P[f, t+n, r])

                    # 4.4º Guarda los valores de la optimizacion
                    for k in K:
                        self.opti_patron[k, t+n, r] = opti_patron[(k, t_opti[t+n])]
                    for f in F:
                        self.opti_produccion[f, t+n, r] = opti_produccion[(f, t_opti[t+n])]
                        self.opti_precio[f, t+n, r] = opti_precio[(f, t_opti[t+n])]
                        self.opti_demanda[f, t+n, r] = opti_demanda[(f, t_opti[t+n])]
                        self.opti_merma[f, t+n, r] = opti_merma[(f, t_opti[t+n])]
                        for u in range(delta-1):
                            self.opti_inventario_inicial[f, t+n, t+n+u, r] = opti_inv_inicial[(f, t_opti[t+n], t_opti[t+n]+u)]
                        for u in range(1, delta):
                            self.opti_inventario_final[f, t+n, t+n+u, r] = opti_inv_final[(f, t_opti[t+n], t_opti[t+n]+u)]
                        for u in range(0, delta):
                            self.opti_demanda_perecible[f, t+n, t+n+u, r] = opti_demanda_perecible[(f, t_opti[t+n], t_opti[t+n]+u)]

                
             # TERMINAL LOGGER 
            if self._print:
                print()
                print("*"*120)
                print(f"Tiempo {t}")
                print("*"*120)

            # 4.5º Actualizar sistema
            for f in F:
        
                #----------------------SIMULACION VS OPTIMIZACION-----------------------------#
                if f in ['Entero'] and self._print:
                    print(f"(S0) Inv para vender en {t} que vence hasta {t+delta-1}")
                    print("   {:<10} {:<2} {:<2} {:<20} {:1} {:<20}".format("Producto", "t", "u", "Simulacion", "-", "Optimizacion"))
                    inv_inicial_opti = 0
                    for u in range(0, delta-1):
                        inv_inicial_opti += self.opti_inventario_inicial[f, t, t+u, r]
                        print("S0 {:<10} {:<2} {:<2} {:<20} {:1} {:<20}".format(f, t, t+u, S0[f][u+1], "-", self.opti_inventario_inicial[f, t, t+u, r]))
                    print()
                #----------------------SIMULACION VS OPTIMIZACION-----------------------------#

                # 4.6º Inventarios inciales de simulacion
                self.simulacion_S_inicial[f, t, r] = sum(list(S0[f].values()))
                
                # 4.7º Guardar produccion de simulacion
                S0[f][self.delta] += self.simulacion_Prod[f, t, r]

                # 4.8º Generar demanda de simulacion
                D_error = 0
                #D_error = -0.6 + self.error_dda[str(r)][str(t)][f]*1.2
                self.simulacion_D_real[f, t, r] = min(alfa[f][1], max(0, (1+D_error)*self.simulacion_D[f, t, r]))
                
                # 4.9º Actualizar inventarios segun ventas
                demanda_insatisfecha, merma, S0 = self.refresh_S0(S0, f, self.delta, self.simulacion_D_real[f, t, r], t, r)
                self.simulacion_Sales[f, t, r] = self.simulacion_D_real[f, t, r] - demanda_insatisfecha
                self.simulacion_S_final[f, t, r] = sum(list(S0[f].values()))
                self.simulacion_L[f, t, r] = merma

                #----------------------SIMULACION VS OPTIMIZACION-----------------------------#
                if f in ['Entero'] and self._print:
                    print("-"*100)
                    print("                       {:<25} {:<20} {:1} {:<20}".format("Producto", "Simulacion", "-", "Optimizacion"))
                    print("Inv incial (t={:<2}) - {:<25}: {:<20} {:1} {:<20}".format(t, f, self.simulacion_S_inicial[f, t, r], "-", inv_inicial_opti))                  # S0 vs W0
                    print("Demanda    (t={:<2}) - {:<25}: {:<20} {:1} {:<20}".format(t, f, self.simulacion_D_real[f, t, r], "-", self.opti_demanda[f, t, r]))           # D_sim vs D_opti (alfa-beta*P)
                    print("Producc    (t={:<2}) - {:<25}: {:<20} {:1} {:<20}".format(t, f, self.simulacion_Prod[f, t, r], "-", self.opti_produccion[f, t, r]))          # prod_sim vs prod_opti (∑ax)
                    print("Merma      (t={:<2}) - {:<25}: {:<20} {:1} {:<20}".format(t, f, self.simulacion_L[f, t, r], "-", self.opti_merma[f, t, r]))                  # L_sim vs L_opti
                    print("-"*100)
                    print()

                    print(f"(D) Venta en {t} que vence hasta {t+delta-1}")
                    print("   {:<10} {:<2} {:<2} {:<20} {:1} {:<20}".format("Producto", "t", "u", "Simulacion", "-", "Optimizacion"))
                    for u in range(0, delta):
                        print("D {:<10} {:<2} {:<2} {:<20} {:1} {:<20}".format(f, t, t+u, self.simulacion_Sales_perecible[f, t, t+u, r], "-", self.opti_demanda_perecible[f, t, t+u, r]))
                    print()

                    print(f"(S) Inv final para vender en {t} que vence hasta {t+delta-1}")
                    print("   {:<10} {:<2} {:<2} {:<20} {:1} {:<20}".format("Producto", "t", "u", "Simulacion", "-", "Optimizacion"))
                    for u in range(1, delta):
                        print("S {:<10} {:<2} {:<2} {:<20} {:1} {:<20}".format(f, t, t+u, S0[f][u], "-",self.opti_inventario_final[f, t, t+u, r]))
                    print()
                #----------------------SIMULACION VS OPTIMIZACION-----------------------------#

                # 4.10º Almacenar metricas de producto (ingresos, costos inventario, costos merma)
                if t <= self.times:
                    # Ingresos
                    self.ingresos[f, t, r] = self.simulacion_P[f, t, r] * self.simulacion_Sales[f, t, r]
                    # Costo inventario
                    self.costo_inventario[f, t, r] = h[f][1] * self.simulacion_S_final[f, t, r]                 # 0.478 $/caja
                    # Costo merma
                    self.costo_merma[f, t, r] = c_merma[f][1] * self.simulacion_L[f, t, r]                      # 0.956 $/caja
            

            # 4.10º Almacenar metricas de periodo (costo corte, utilidad)
            if t <= self.times:
                for k in K:
                    self.costo_corte[k, t, r] = c[k] * self.simulacion_X[k, t, r]

                self.objective_value[t, r] =\
                    quicksum(self.ingresos[f, t, r] for f in F).getValue() \
                    - quicksum(self.costo_inventario[f, t, r] for f in F).getValue() \
                    - quicksum(self.costo_merma[f, t, r] for f in F).getValue() \
                    - quicksum(self.costo_corte[k, t, r] for k in K).getValue()


    @timeit
    def run_replics(self):
        for r in range(1, self.replics+1):
            print("\nReplica:", r)
            self.run(r)
            print(f"\nUtilidad replica {r}: {sum(list(self.objective_value.values()))}\n")


    def save_to_pickle(self, n_escenario):
        directorio = f"~/Desktop/Produccion-Tesis/Resultados/Escenario {n_escenario}/"
        # Guardar variables de simulacion
        pd.Series(self.simulacion_X).rename_axis(['k', 't', 'r']).reset_index(name='value').to_excel(directorio+"X.xlsx", engine='openpyxl')
        pd.Series(self.simulacion_S_final).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"S.xlsx", engine='openpyxl')
        pd.Series(self.simulacion_S_inicial).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"S_inicial.xlsx", engine='openpyxl')
        pd.Series(self.simulacion_D).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"D.xlsx", engine='openpyxl')
        pd.Series(self.simulacion_P).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"P.xlsx", engine='openpyxl')
        pd.Series(self.simulacion_L).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"L.xlsx", engine='openpyxl')
        pd.Series(self.simulacion_D_real).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"D_real.xlsx", engine='openpyxl')
        pd.Series(self.simulacion_Sales).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"Sales.xlsx", engine='openpyxl')
        pd.Series(self.simulacion_Prod).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"prod.xlsx", engine='openpyxl')
        
        # Guardar metricas de simulacion
        pd.Series(self.objective_value).rename_axis(['t', 'r']).reset_index(name='value').to_excel(directorio+"objective_value.xlsx", engine='openpyxl')
        pd.Series(self.ingresos).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"ingresos.xlsx", engine='openpyxl')
        pd.Series(self.costo_inventario).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"costo_inventario.xlsx", engine='openpyxl')
        pd.Series(self.costo_merma).rename_axis(['f', 't', 'r']).reset_index(name='value').to_excel(directorio+"costo_merma.xlsx", engine='openpyxl')
        pd.Series(self.costo_corte).rename_axis(['k', 't', 'r']).reset_index(name='value').to_excel(directorio+"costo_corte.xlsx", engine='openpyxl')
