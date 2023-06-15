from gurobipy import *
from variable_sets import *
from interpreta_output import *
from read_data import *
import datetime as dt

#Modelo general
#String_input: nombre del archivo input
#scaler: En cuanto se achica el problema (cajas de esa cantidad)
#experiment: para hacer experimentos variando Q, Alfa, beta, valor residual, S, periods
#los _mult son para ponderar algunas variables
#rep: número de repetición del modelo (para algunos exps repito varias veces la misma instancia y después promedio)
#save: si es True, se guarda el output del modelo
def model_1(string_input, mip_gap, time_limit, scaler, periods=None, case=0, iterate=False,
          q_mult=1, alfa_mult=1, beta_mult=1, res_mult=1, S0_mult=1, loggin=0, delta_=9, save=True, init_S0=None):

    now = dt.datetime.now()

    #leer y calcular input necesarios para el modelo
    #F: lista de productos
    #T: lista de períodos (T_0 incluye el 0, T_delta termina en max(T) + delta - 1)
    #K: patrones de corte
    #alfa, beta: parámetros para cálculo de demanda en función del precio
    #delta: períodos hasta el vencimiento
    #S0: inventario inicial
    #a: cortes de producto f por patrón k
    #h: costo de inventario de t a t+1 por producto f
    #c: costo por patrón de corte k
    #q: insumo diario
    F, T, T_delta, T_0, T_0_delta, K, alfa, beta, delta, S0, a, h, c, q, \
    c_merma, p_constante, p_compat, v_constante = \
    read_all_data(
        f"~/Desktop/Produccion-Tesis/Input/{string_input}.xlsx",
        periods=periods,
        scaler=scaler,
        print_out=False,
        fix_alfa=False,
        q_mult=q_mult,
        alfa_mult=alfa_mult,
        beta_mult=beta_mult,
        res_mult=res_mult,
        S0_mult=S0_mult,
        delta_=delta_
    )
    delta = delta_
    if init_S0:
        S0 = init_S0                        # 0.1º Setear un inventario inicial fuera del Excel.


    model_name = "M1"
    MVTPI = Model("Modelo con variables de producción total e inventarios")
    MVTPI.Params.PSDTol = 0.0001
    MVTPI.Params.NonConvex = 2
    MVTPI.Params.MIPGap = mip_gap
    MVTPI.Params.TimeLimit = time_limit
    MVTPI.Params.OutputFlag = loggin # Para imprimir sin output


    #Calcular tuplas de indices factibles para las distintas variables (módulo variable_sets)
    inventario_set = S_sets(F, T, delta)
    demanda_set = D_sets(F, T, delta)
    produccion_set = X_sets(K, T)
    precios_set = P_sets(F, T)

    #Inicializar variables
    #Cantidad de cortes con patrón K en T (solo períodos en los que se produce/vende)
    X = MVTPI.addVars(produccion_set, vtype=GRB.INTEGER, lb=0,  name = 'x')
    #Inventario desde T a T+1 (con T = 0...last_t-1) de producto final F que vence en T+2..T+delta-1
    S = MVTPI.addVars(inventario_set, vtype=GRB.CONTINUOUS, lb=0,  name = 's')
    #Cantidad de la demanda de producto final tipo F en T que se satisface con productos que vencen en U = t+1..t+delta
    D = MVTPI.addVars(demanda_set,  vtype=GRB.CONTINUOUS, lb=0, name= 'd')
    #Precio del producto final F en período T
    P = MVTPI.addVars(precios_set, vtype=GRB.CONTINUOUS, lb=0,  name='p')
    #Cantidad de producto final F que se da por pérdido en período T
    L = MVTPI.addVars(precios_set, vtype=GRB.CONTINUOUS, lb=0,  name='l')


    #Objetivo: maximizar beneficio operacional (venta - costo de inventario - costo de producción)
    obj = quicksum(P[f, t]*D[f, t, u] for (f, t, u) in demanda_set) \
          - quicksum(h[f][t]*S[f, t, u] for (f, t, u) in inventario_set if t != 0) \
          - quicksum(c[k]*X[k, t] for (k, t) in produccion_set) \
          - quicksum(L[f, t]*c_merma[f][t] for (f, t) in precios_set)

    MVTPI.setObjective(obj, GRB.MAXIMIZE)

    #Todos los insumos de un día deben ser cortados
    MVTPI.addConstrs((quicksum(X[k, t] for k in K) == round(q[t]) for t in T))

    #Relación inventarios iniciales
    MVTPI.addConstrs((S[f, t, u] == S0[f][u] - D[f, t, u]
                      for f in F for t in [T[0]] for u in range(t + 1, t + delta - 1)))

    ##Relación inventarios NO iniciales
    MVTPI.addConstrs((S[f, t, u] == S[f, t - 1, u] - D[f, t, u]
                      for f in F for t in T[1:] for u in range(t + 1, t + delta - 1)))

    #Inicialización de inventarios recién producidos
    MVTPI.addConstrs((S[f, t, t + delta - 1] == quicksum(a[f][k]*X[k, t] for k in K) - D[f, t, t + delta - 1]
                      for f in F for t in T))

    #Definición de demandas en función del precio
    MVTPI.addConstrs((quicksum(D[f, t, u] for u in range(t, t + delta))
                      == alfa[f][t] - beta[f][t]*P[f, t]
                      for f in F for t in T))

    #Mermas en primer período
    MVTPI.addConstrs(S0[f][T[0]] == D[f, T[0], T[0]] + L[f, T[0]] for f in F)

    #Mermas en otros períodos
    MVTPI.addConstrs((S[f, t - 1, t] == D[f, t, t] + L[f, t] for f in F for t in T[1:]))

    #Limite de precio  (estas se agregan en definiciones de variables con lb y ub)
    MVTPI.addConstrs(P[f, t] <= alfa[f][t]/beta[f][t] for f in F for t in T if beta[f][t] != 0)

    # Compatibilidad de precio. Para cada período y relación de precio (f_quants), la suma producto es >= 0
    _, _, p_compat = read_sheet(f"~/Desktop/Produccion-Tesis/Input/{string_input}.xlsx", "Restricciones-Precios1v3")
    p_compat_lists = [list(zip(f_quants.keys(), f_quants.values())) for f_quants in list(p_compat.values())]
    MVTPI.addConstrs(quicksum(P[f, t]*quantity for f, quantity in p_compat_lists[i]) >= 0
                        for i in range(len(p_compat_lists)) for t in T)

    # Evitar canibalizacion de precios entre subproductos.
    _, _, p_compat_2 = read_sheet(f"~/Desktop/Produccion-Tesis/Input/{string_input}.xlsx", "Restricciones-Precios2v3")
    p_compat_lists_2 = [list(zip(f_quants.keys(), f_quants.values())) for f_quants in list(p_compat_2.values())]
    MVTPI.addConstrs(quicksum(P[f, t]*quantity for f, quantity in p_compat_lists_2[i]) <= 0
                        for i in range(len(p_compat_lists_2)) for t in T)

    # Demandas FIFO
    #Inventario inicial en t de producto final F que vence en u=t,t+1...t+delta-1
    inventario_incial_set = [(f, t, u) for f in F for t in T for u in range(t, t + delta)]
    W0 = MVTPI.addVars(inventario_incial_set, vtype=GRB.CONTINUOUS, lb=0,  name = 's')
    MVTPI.addConstrs(W0[f, 1, u] == S0[f][u] for u in range(1, delta) for f in F)
    MVTPI.addConstrs(W0[f, t+1, u] == S[f, t, u] for f in F for t in T[:-1] for u in range(t+1, t+delta) )

    # R11) Generar FIFO para asignacion de venta de inventario inicial
    B = MVTPI.addVars(inventario_incial_set, vtype=GRB.BINARY, name = 'BINARIA1')
    M = 1e9
    MVTPI.addConstrs(
        W0[f, t, u] - D[f, t, u] <= M*B[f, t, u] for f in F for t in T for u in range(t, t+delta)
    )
    MVTPI.addConstrs(
        quicksum(a[f][k]*X[k, t] for k in K) - D[f, t, t+delta-1] <= M*B[f, t, t+delta-1] for f in F for t in T 
    )
    MVTPI.addConstrs(
        M*(1-B[f, t, u]) <= D[f, t, u+1] for f in F for t in T for u in range(t, t+delta-1)
    )

    ##################################### CASE 1 ##############################################
    # Case 1: , pero sigue siendo una variable.
    if case == 1:
        MVTPI.addConstrs(P[f, t] == P[f, t + 1] for f in F for t in T[:-1])
    ##################################### CASE 1 ##############################################

    MVTPI.update()
    MVTPI.optimize()
    
    #status, opt_value = MVTPI.Status, MVTPI.objVal


    if iterate:
            X, P, L = get_all_vars_dict([X, P, L])
            demands = {(f,t): alfa[f][t] - beta[f][t]*P[f, t] for f in F for t in T}
            production = {(f,t): sum([a[f][k]*X[k, t] for k in K]) for f in F for t in T}
            pattern_use = {(k,t): X[k, t] for k in K for t in T}
            price = {(f,t): P[f, t] for f in F for t in T}
            
            # Obtencion de la demanda, el inventario y la produccion          
            inventario = {(f,t,u): S[f, t, u].X for (f,t,u) in inventario_set}
            inventario = pd.DataFrame([(*key, value) for (key, value) in inventario.items()]).rename(columns={0: 'f', 1: 't', 2:'u', 3: 'value'})
            inventario = inventario.set_index(['f', 't', 'u']).to_dict()['value']

            demanda = {(f,t,u): D[f, t, u].X for (f, t, u) in demanda_set}
            demanda = pd.DataFrame([(*key, value) for (key, value) in demanda.items()]).rename(columns={0: 'f', 1: 't', 2:'u', 3: 'value'})
            demanda = demanda.set_index(['f', 't', 'u']).to_dict()['value']

            merma = {(f,t): L[f, t] for f in F for t in T}
            merma = pd.DataFrame([(*key, value) for (key, value) in merma.items()]).rename(columns={0: 'f', 1: 't', 2: 'value'})

            produccion_w = None
            W0_ =  {(f,t,u): W0[f, t, u].X for (f,t,u) in inventario_incial_set}
            #W0_ = {(f,t,u): S[f, t, u].X for (f,t,u) in inventario_set}
            #W0_ = pd.DataFrame([(*key, value) for (key, value) in W0_.items()]).rename(columns={0: 'f', 1: 't', 2:'u', 3: 'value'})
            #print(W0_.head(10))

            return production, price, pattern_use, inventario, demands, L, W0_, demanda
            #return opti_produccion, opti_precio, opti_patrones, opti_inv_final, opti_demanda, opti_merma, opti_inv_inicial, opti_demanda_perecible


    return status, opt_value
