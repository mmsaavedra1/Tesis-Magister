import sys        
import random
import warnings
import time
import numpy as np
import pandas as pd
import collections, functools, operator

import pickle

from M2 import *

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
    return demand, merma


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

        #_, _, a = read_sheet(read_filename, "Patrones - Piezas")

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

        # 3º Cargar soluciones de escenario 6
        #if self.n_escenario == 6:
        #    with open("prod.dat", "rb") as f:
        #        prod_escenario_1 = pickle.load(f)
        #    with open("P.dat", "rb") as f:
        #        P_escenario_1 = pickle.load(f)
        #    with open("X.dat", "rb") as f:
        #        X_escenario_1 = pickle.load(f)

        if self._print:
            archi1=open(f"Logger/Simulacion {self.n_escenario}.txt","w")

        print("\nReplica:", r)
        # 4º Simulacion
        for t in range(1, self.times + 1):
            # 4.1º Ver decision de optimizacion
            if n_opti == self.remaining_days:
                n_opti = 0
                # 4.2º Optimizar
                demands, production, price, pattern_use, objective_function, runtime, inventario, demanda, produccion_w = self.model(
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
                        # Copiar politicas de corte
                        #if (self.n_escenario == 6):  
                        #    self.X[k, t, r] = X_escenario_1[k, t, r]

                    for f in F:
                        self.P[f, t+n, r] = price[(f, n+1)]                                     # OPCION 1: PRECIOS CONTINUOS
                        #self.P[f, t+n, r] = round(price[(f, n+1)], 2)                           # OPCION 2: PRECIOS CONTINUOS RENDONDEADOS
                        self.prod[f, t+n, r] = sum(a[f][k]*pattern_use[(k, n+1)] for k in K)  
                        
                        # Copiar politicas de precios y produccion
                        #if (self.n_escenario == 6):  
                        #    self.prod[f,t,r] = prod_escenario_1[f, t, r]
                        #    self.P[f, t, r] = P_escenario_1[f, t, r]   
                        #self.D[f, t+n, r] = demanda[(f, n+1)]                                  # OPCION 1: Calculo Demanda (OBSOLETA)
                        self.D[f, t+n, r] = alfa[f][n+1]-(beta[f][n+1]*self.P[f, t+n, r])       # OPCION 2: Calculo Demanda según decisiones

            # 4.3º Actualizar estados
            for f in F:
                # 4.3.0º Respaldar inventarios inciales
                self.S_inicial[f, t, r] = sum(list(S0[f].values()))

                # 4.3.1º Generar realización de demanda para cambiar el sistema
                D_error = 0
                D_error = -0.4 + self.error_dda[str(r)][str(t)][f]*0.8              # +- 40% de error         
                this_demand = max((1 + D_error)*self.D[f, t, r], 0)                 # +- 40% de error 
                self.D_real[(f, t, r)] = this_demand

                # ************* LOG *************
                if f == "Entero" and self._print:
                    archi1.write(f"Inicio en {t}:\n")

                    archi1.write(f"(Inv inicial) Simulacion: {sum(list(S0[f].values()))}\n")
                    archi1.write("\n****** Detalle de inventario: *******\n")
                    for u in range(1, self.delta+1):
                        archi1.write(f"Inventario venta en {u}: {S0[f][u]}\n")
                    archi1.write("****** Fin Detalle de inventario: *******\n\n")

                    archi1.write("\n****** Detalle de Produccion: *******\n")
                    archi1.write(f"(Produccion) Simulacion: {self.prod[f,t,r]}\n")
                    archi1.write("****** Fin Detalle de produccion: *******\n\n")
                # ************* LOG *************
            
                # 4.3.2º Inventario inicial en funcion de la produccion
                S0[f][self.delta] += self.prod[f, t, r]
    
                # ************* LOG *************
                if f == "Entero" and self._print:
                    archi1.write("\n****** Detalle de inventario actualizado: *******\n")
                    for u in range(1, self.delta+1):
                        archi1.write(f"Inventario en {t} para que vence en {t+u-1}: {S0[f][u]}\n")
                    archi1.write("\n****** Fin de inventario actualizado: *******\n\n")
                # ************* LOG *************

                # 4.3.3º Actualizacion de variables de estado del sistema 
                unserved_demand, this_L = refresh_S0(S0, f, self.delta, this_demand)
                self.sales[f, t, r] = self.D_real[(f, t, r)] - unserved_demand
                self.S[f, t, r] = sum(list(S0[f].values()))
                self.L[f, t, r] = this_L

                 # ************* LOG *************
                if f == "Entero" and self._print:
                    archi1.write(f"\Ventas en {t}:\n")
                    archi1.write(f"(Demanda) [alfa-beta*p]) Opti: {self.D[f, t, r]}\n")
                    archi1.write(f"(Error) Simulacion: {self.error_dda[str(r)][str(t)][f]}\n")
                    archi1.write(f"(Factor Dcto) Simulacion: {-0.4 + self.error_dda[str(r)][str(t)][f]*0.8}\n")
                    archi1.write(f"(Demanda) Simulacion: {self.D_real[(f, t, r)]}\n")
                    archi1.write(f"(Venta) Simulacion: {self.sales[(f, t, r)]}\n")
                # ************* LOG *************

                # ************* LOG *************
                if f == "Entero" and self._print:
                    archi1.write(f"\nFinal en {t}:\n")
                    archi1.write(f"(Merma) Simulacion: {self.L[f, t, r]}\n")
                    archi1.write(f"(Precio de venta) Simulacion: {self.P[f, t, r]}\n")
                    archi1.write(f"(Inv final) Simulacion: {sum(list(S0[f].values()))}\n")
                    archi1.write("***************************\n\n")
                # ************* LOG *************
                
             # 4.4º Almacenar valor objetivo como metrica            
            if (t >= 251) and (t <= self.times):
                for f in F:
                    self.ingresos[f, t, r] = self.P[f, t, r] * self.sales[(f, t, r)]
                    self.costo_inventario[f, t, r] = h[f][1] * self.S[f, t, r]               # 0.478 $/caja
                    self.costo_merma[f, t, r] = c_merma[f][1] * self.L[f, t, r]              # 0.956 $/caja
                for k in K:
                    self.costo_corte[k, t, r] = c[k] * self.X[k, t, r]

                self.objective_value[t, r] =\
                    quicksum(self.ingresos[f, t, r] for f in F).getValue() \
                    - quicksum(self.costo_inventario[f, t, r] for f in F).getValue() \
                    - quicksum(self.costo_merma[f, t, r] for f in F).getValue() \
                    - quicksum(self.costo_corte[k, t, r] for k in K).getValue()
                    

            # 4.5º Actualizar valores para terminar la semana
            n_opti += 1

        if self._print:
            archi1.close()
            
        return True
               
    
    @timeit
    def run_replics(self):
        for r in range(1, self.replics+1):
            obj_val = self.run(r)
            self.out_obj_val[r] = obj_val


    def save_to_pickle(self, n_escenario):
        directorio = f"~/Desktop/Produccion-Tesis/Resultados/Escenario {n_escenario}/"
        pd.Series(self.X).rename_axis(['k', 't', 'r']).reset_index(name='value').to_pickle(directorio+"X.pkl")
        pd.Series(self.S).rename_axis(['f', 't', 'r']).reset_index(name='value').to_pickle(directorio+"S.pkl")
        pd.Series(self.S_inicial).rename_axis(['f', 't', 'r']).reset_index(name='value').to_pickle(directorio+"S_inicial.pkl")
        pd.Series(self.D).rename_axis(['f', 't', 'r']).reset_index(name='value').to_pickle(directorio+"D.pkl")
        pd.Series(self.P).rename_axis(['f', 't', 'r']).reset_index(name='value').to_pickle(directorio+"P.pkl")
        pd.Series(self.L).rename_axis(['f', 't', 'r']).reset_index(name='value').to_pickle(directorio+"L.pkl")
        pd.Series(self.D_real).rename_axis(['f', 't', 'r']).reset_index(name='value').to_pickle(directorio+"D_real.pkl")
        pd.Series(self.sales).rename_axis(['f', 't', 'r']).reset_index(name='value').to_pickle(directorio+"sales.pkl")
        pd.Series(self.prod).rename_axis(['f', 't', 'r']).reset_index(name='value').to_pickle(directorio+"prod.pkl")
        pd.Series(self.objective_value).rename_axis(['t', 'r']).reset_index(name='value').to_pickle(directorio+"objective_value.pkl")
        pd.Series(self.ingresos).rename_axis(['f', 't', 'r']).reset_index(name='value').to_pickle(directorio+"ingresos.pkl")
        pd.Series(self.costo_inventario).rename_axis(['f', 't', 'r']).reset_index(name='value').to_pickle(directorio+"costo_inventario.pkl")
        pd.Series(self.costo_merma).rename_axis(['f', 't', 'r']).reset_index(name='value').to_pickle(directorio+"costo_merma.pkl")
        pd.Series(self.costo_corte).rename_axis(['k', 't', 'r']).reset_index(name='value').to_pickle(directorio+"costo_corte.pkl")
