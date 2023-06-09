import sys
sys.path.insert(0, '/Library/gurobi952/mac64/')

from gurobipy import *
from variable_sets import *
from interpreta_output import *


#Modelo general
#String_input: nombre del archivo input
#scaler: En cuanto se achica el problema (cajas de esa cantidad)
#experiment: para hacer experimentos variando Q, Alfa, beta, valor residual, S, periods
#los _mult son para ponderar algunas variables
#rep: número de repetición del modelo (para algunos exps repito varias veces la misma instancia y después promedio)
#save: si es True, se guarda el output del modelo
def model_2(
    string_input: str,
    mip_gap: float,
    time_limit: int,
    scaler: int,
    periods: int=None,
    experiment: str=None,
    q_mult: int=1,
    alfa_mult: int=1,
    beta_mult:int =1,
    res_mult: int=1,
    S0_mult:int =1,
    rep: int=None,
    save: bool=True,
    case: int=0,
    iterate: bool=False,
    init_S0=None,
    weekly_price: bool=False,
    remaining_week_days: int=0,
    fixed_price_0=None,
    fixed_q1: int=False,
    loggin: bool=0,
    delta_: int=9):
    """
    Input:
    
    - string_input: Nombre del archivo input.
    - mip_gap: Gap deseado para el MIP.
    - time_limit: Limite en segundos de ejecucion.
    - scaler: Cantidad que se achica el problema (cajas de esa cantidad).
    - periods: Cantidad limite de periodos a analizar.
    - experiment: Para hacer experimentos variando: Q, Alfa, beta, valor residual, S, periods.
    - *_mult: Listas que contienen ponderadores de variables.
    - rep: Número de repetición del modelo (para algunos experimientos se repite varias veces la misma instancia y después promedio).
    - save: Variable para guardar informacion del modelo.
    - case: Indentificador que se le coloca a cada caso de estudio
    
    -iterate:
    -init_S0:
    -weekly_price:
    -reaining_week_days:
    -fixed_price_0:
    -fixed_q1: Parametro para ajustar via codigo el primer valor de q.
    """

    # Trigger temporal para medir tiempo de ejecucion
    now = dt.datetime.now()

    # 0º Generar todos los inputs del modelo
    # F: list() -> Productos finales 
    # T: list() -> Tiempo de rango []
    # T_delta: list() -> Tiempo de rango []
    # T_0: list() -> Tiempo de rango []
    # T_delta: list() -> Tiempo de rango [0, delta - 1]
    # K: list() -> Patrones de corte
    # Alfa, Beta: dict() -> Parametros regresion lineal de la demanda
    # Delta: Int() -> Tiempo hasta el vencimiento
    # S0: list() -> Inventario inicial de producto final
    # a_f_k: dict() -> Cantidad obtenida de producto final f por corte k
    # h: list() -> Costo de inventario final de t por producto f
    # c: list() -> Costo por patrón de corte k
    # q: list() -> Cantidad de producto inicial
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


    ################################## PARAMETROS DE MODELO ###################################
    if init_S0:
        S0 = init_S0                        # 0.1º Setear un inventario inicial fuera del Excel.

    # Parametros extra del modelo
    last_t = T[-1]                          # Último período en el que se puede producir/vender
    h_acum = calc_h_acum(h, delta, T, F)    # Costo inventario acumulado (g) entre t y s (venta o merma, lo que pase primero).
    ################################## PARAMETROS DE MODELO ###################################


    # 1º Setear todos los parametros de usuario para el solver
    model_name = "M2"
    MVPFCF = Model("Modelo con variables de producción en una fecha para consumo en otra fecha")
    MVPFCF.Params.OutputFlag = loggin # Para imprimir sin output
    MVPFCF.Params.MIPGap = mip_gap
    MVPFCF.Params.TimeLimit = time_limit


    # 2º Calcular indices de conjuntos - Modulo variable_sets
    produccion_set = X_sets(K, T)
    precios_set = P_sets(F, T)
    asignacion_sets = W_sets(F, T, delta)
    inv_inicial_sets = W0_sets(F, delta)
    inventario_set = P_sets(F, T)

    # 3º Creacion de variables del modelo
    #Cantidad de cortes con patrón K en T en piezas de pollo (NO cajas) (solo períodos en los que se produce/vende)
    #Y = MVPFCF.addVars(produccion_set, vtype=GRB.INTEGER, lb=0,  name = 'y')
    #Cantidad de cortes con patrón K en T en cajas de pollo (solo períodos en los que se produce/vende)
    X = MVPFCF.addVars(produccion_set, vtype=GRB.INTEGER, lb=0,  name = 'x')
    
    #Todoo el resto en cajas!!!
    # Cantidad de producto f, producido en t y que satisface la demanda en s
    W = MVPFCF.addVars(asignacion_sets, vtype=GRB.CONTINUOUS, lb=0,  name = 'w')
    #Cantidad de producto inicial f que satisface la demanda en s y vence en u
    W0 = MVPFCF.addVars(inv_inicial_sets,  vtype=GRB.CONTINUOUS, lb=0, name= 'w0')
    #Precio del producto final F en período T
    P = MVPFCF.addVars(precios_set, vtype=GRB.CONTINUOUS, lb=0,  name='p')
    #Cantidad de producto final F que se da por pérdido en período T
    L = MVPFCF.addVars(precios_set, vtype=GRB.CONTINUOUS, lb=0,  name='l')
    
    # 4º Restricciones del modelo
    # R1) Compatibilidad de precio. Para cada período y relación de precio (f_quants), la suma producto es >= 0
    _, _, p_compat = read_sheet(f"~/Desktop/Produccion-Tesis/Input/{string_input}.xlsx", "Restricciones-Precios1v3")
    p_compat_lists = [list(zip(f_quants.keys(), f_quants.values())) for f_quants in list(p_compat.values())]
    MVPFCF.addConstrs(quicksum(P[f, t]*quantity for f, quantity in p_compat_lists[i]) >= 0
                        for i in range(len(p_compat_lists)) for t in T)

    # R2) Evitar canibalizacion de precios entre subproductos.
    #_, _, p_compat_2 = read_sheet(f"~/Desktop/Produccion-Tesis/Input/{string_input}.xlsx", "Restricciones-Precios2v3")
    #p_compat_lists_2 = [list(zip(f_quants.keys(), f_quants.values())) for f_quants in list(p_compat_2.values())]
    #MVPFCF.addConstrs(quicksum(P[f, t]*quantity for f, quantity in p_compat_lists_2[i]) <= 0
    #                    for i in range(len(p_compat_lists_2)) for t in T)
    
    # R2) Evitar canibalizacion de precios entre subproductos.
    #_, _, p_compat_3 = read_sheet(f"~/Desktop/Produccion-Tesis/Input/{string_input}.xlsx", "Restricciones-Precios3v3")
    #p_compat_lists_3 = [list(zip(f_quants.keys(), f_quants.values())) for f_quants in list(p_compat_3.values())]
    #MVPFCF.addConstrs(quicksum(P[f, t]*quantity for f, quantity in p_compat_lists_3[i]) <= 0
    #                    for i in range(len(p_compat_lists_3)) for t in T)

    # R3) Todos los insumos deben ser cortados
    MVPFCF.addConstrs(quicksum(X[k, t] for k in K) == q[t] for t in T)

    # R4) Todo lo que se produce es asignado a algún período (puede ser venta o merma)
    MVPFCF.addConstrs(quicksum(a[f][k] * X[k, t] for k in K) == quicksum(W[f, t, s] for s in range(t, min(t + delta, T[-1] + 2))) for f in F for t in T)

    # R5.1) NUEVA: Satisfaccion de demanda
    MVPFCF.addConstrs(quicksum(W0[f, s, u] for u in range(s,  delta)) #inv inicial
                    + quicksum(W[f, t, s] for t in range(max(1, s - delta + 1), s + 1)) #produccion
                    ==  (alfa[f][s] - beta[f][s]*P[f, s]) + L[f, s] for f in F for s in range(1, delta)) #demanda + merma

    # R5.2) NUEVA: Distribucion inventario inicial FIFO --> Asignacion de inventario inicial por caducar debe ser FIFO
    MVPFCF.addConstrs(quicksum(W[f, t, s] for t in range(max(1, s - delta + 1), s + 1)) #produccion
                    ==  (alfa[f][s] - beta[f][s]*P[f, s]) + L[f, s] for f in F for s in range(delta, T[-1]+1)) #demanda + merma
    
    # R5.original
    #MVPFCF.addConstrs(quicksum(W0[f, s, u] for u in range(s,  delta)) #inv inicial
    #                    + quicksum(W[f, t, s] for t in range(max(1, s-delta+1), s + 1)) #produccion
    #                    == (alfa[f][s] - beta[f][s]*P[f, s]) + L[f, s] for f in F for s in T) #demanda + merma

    # R6) Relacion entre merma y asignacion, desde período delta
    MVPFCF.addConstrs(W[f, s - delta + 1, s] # produccion que vence en s y es asignada a s
                      >= L[f, s] for f in F for s in range(delta, last_t + 1))  # demanda + merma

    # R7) Relacion entre merma y asignacion, hasta período delta - 1
    MVPFCF.addConstrs(W0[f, s, s]  # inventario inicial que vence en s y es asignado a s
                      >= L[f, s] for f in F for s in range(1, delta))  # merma

    # R8) Todo el inventario inicial es asignado a alågún período
    MVPFCF.addConstrs(
        quicksum(W0[f, s, u] for s in range(1, u + 1)) == S0[f][u] for f in F for u in range(1, delta)
    )

    # R9) Relación de precio
    MVPFCF.addConstrs(P[f, s] <= alfa[f][s]/beta[f][s] for f in F for s in T if beta[f][s] != 0)

    # R10) No negatividad de asignacion de venta
    MVPFCF.addConstrs(W[f, t, s] >= 0 for f,t,s in asignacion_sets)
    MVPFCF.addConstrs(W0[f, t, s] >= 0 for f,t,s in inv_inicial_sets)

    # R11) Generar FIFO para asignacion de venta de inventario inicial
    #B = MVPFCF.addVars(inv_inicial_sets, vtype=GRB.BINARY, name = 'BINARIA1')
    #M = 1e9
    
    #MVPFCF.addConstrs(
    #    M*B[f, s, u] >= W0[f, s, u] for f, s, u in inv_inicial_sets
    #)

    #MVPFCF.addConstrs(
    #    B[f, s, u] + B[f, s_hat, u_hat] <= 1 for f, s, u in inv_inicial_sets for _, s_hat, u_hat in inv_inicial_sets if (u > u_hat) and (s < s_hat)
    #)
    
    
    ##################################### CASE 1 ##############################################
    # Case 1: , pero sigue siendo una variable.
    if case == 1:
        MVPFCF.addConstrs(P[f, t] == P[f, t + 1] for f in F for t in T[:-1])
    ##################################### CASE 1 ##############################################


    # 5º Definicion de funcion objetivo
    # Maximizar beneficio operacional (venta - costo de inventario - costo de producción)
    #ORIGINAL
    obj_v1 = quicksum(P[f, s]*(alfa[f][s] - beta[f][s] * P[f, s]) for f in F for s in T) - \
        quicksum(h_acum[f][t][s]*W[f, t, s] for f in F for t in T for s in range(t + 1, min(last_t + 2, t + delta))) - \
        quicksum(h_acum[f][1][s]*W0[f, s, u] for f in F for s in range(2, delta) for u in range(s, delta)) - \
        quicksum(c_merma[f][t] * L[f, t] for f in F for t in T) - \
        quicksum(c[k] * X[k, t] for k in K for t in T)
       
    MVPFCF.setObjective(obj_v1, GRB.MAXIMIZE)

    # 6º Optimizar el modelo
    MVPFCF.update()
    #MVPFCF.write("modelo.lp")
    MVPFCF.optimize()


    if save:
        print("Saved!")
        some_vars = [X, P, L]
        X, P, L = get_all_vars_dict(some_vars) #obtener dicts de alguns variables
        ingresos = sum([P[f, s] * (alfa[f][s] - beta[f][s] * P[f, s]) for f in F for s in T])
        c_inv_y_residual = sum([h_acum[f][t][s] * W[f, t, s].X for f in F for t in T for s in range(t + 1, min(last_t + 2, t + delta))])
        c_inv_no_inicial = sum([h_acum[f][t][s] * W[f, t, s].X for f in F for t in T for s in range(t + 1, min(last_t + 1, t + delta))])
        ingreso_residual = c_inv_no_inicial - c_inv_y_residual
        c_inv_inicial = sum([h_acum[f][1][s] * W0[f, s, u].X for f in F for s in range(2, delta) for u in range(s, delta)])
        c_produccion = sum([c[k] * X[k, t] for k in K for t in T])
        c_merma = sum([c_merma[f][t] * L[f, t] for f in F for t in T])

        indicadores = {
            "Ingresos": [ingresos],
            "Ingreso residual": [ingreso_residual],
            "C_inv_no_inicial": [c_inv_no_inicial],
            "C_inv_inicial": [c_inv_inicial],
            "C_produccion": [c_produccion],
            "C_merma": [c_merma]}

        alfa = change_dict_keys(alfa)
        beta = change_dict_keys(beta)
        S = w_to_s(W, W0, T, F, delta)  # obtener variables de inventario (como en M1) a partir de W y W0
        D = alpha_beta_to_d(W, W0, L, T, F, delta)  # obtener variables de demanda (como en M1) a partir de W, W0 y L
        out_vars = [X, S, D, P, L, alfa, beta]
        

        # Para conocer la produccion de pollos
        d = {(f,t): sum([a[f][k]*X[k, t] for k in K]) for f in F for t in T}
        pd.Series(d).rename_axis(['f', 't']).reset_index(name='value').to_excel("~/Desktop/produccion.xlsx")

        status, opt_value = save_model_data(
            MVPFCF,
            model_name,
            case,
            string_input,
            out_vars,
            experiment,
            now,
            q_mult, 
            alfa_mult,
            beta_mult,
            S0_mult,
            mip_gap,
            time_limit,
            res_mult,
            rep,
            scaler,
            q,
            alfa,
            beta,
            periods,
            indicadores
        ) # Guardar output y dump
    
    # 7º Manejo de resultados
    if not save:
        if not iterate:
            X, P, L, W_ = get_all_vars_dict([X, P, L, W])

            inventario =  pd.DataFrame([(*key, value) for (key, value) in w_to_s(W, W0, T, F, delta).items()])
            inventario = inventario.rename(columns={0: 'f', 1: 't', 2: 'delta', 3: 'value'})

            demanda = pd.DataFrame([(*key, value) for (key, value) in alpha_beta_to_d(W, W0, L, T, F, delta).items()]).rename(columns={0: 'f', 1: 't', 2: 'delta', 3: 'value'})
            demanda = demanda.rename(columns={0: 'f', 1: 't', 2: 'delta', 3: 'value'})

            demanda_alfa_beta = {(f,t): alfa[f][t] - beta[f][t]*P[f, t] for f in F for t in T}
            demanda_alfa_beta = pd.DataFrame([(*key, value) for (key, value) in demanda_alfa_beta.items()]).rename(columns={0: 'f', 1: 't', 2: 'value'})

            precio = {(f,t): P[f, t] for f in F for t in T}
            precio = pd.DataFrame([(*key, value) for (key, value) in precio.items()]).rename(columns={0: 'f', 1: 't', 2: 'value'})

            merma = pd.DataFrame([(*key, value) for (key, value) in L.items()]).rename(columns={0: 'f', 1: 't', 2: 'value'})
            merma = merma.rename(columns={0: 'f', 1: 't', 3: 'value'})

            produccion_w = pd.DataFrame([(*key, value) for (key, value) in W_.items()]).rename(columns={0: 'f', 1: 't', 2: 's', 3: 'value'})
            produccion_w = produccion_w.rename(columns={0: 'f', 1: 't', 2: 's', 3: 'value'})


            return inventario, demanda, demanda_alfa_beta, precio, merma, produccion_w, X


        if iterate:
            X, P, L, W_ = get_all_vars_dict([X, P, L, W])
            demands = {(f,t): alfa[f][t] - beta[f][t]*P[f, t] for f in F for t in T}
            production = {(f,t): sum([a[f][k]*X[k, t] for k in K]) for f in F for t in T}
            pattern_use = {(k,t): X[k, t] for k in K for t in T}
            price = {(f,t): P[f, t] for f in F for t in T}
            
            # Obtencion de la demanda, el inventario y la produccion
            demanda = pd.DataFrame([(*key, value) for (key, value) in alpha_beta_to_d(W, W0, L, T, F, delta).items()]).rename(columns={0: 'f', 1: 't', 2: 'delta', 3: 'value'})
            demanda = demanda.rename(columns={0: 'f', 1: 't', 2: 'delta', 3: 'value'})
            demanda = demanda.groupby(['f', 't']).sum().reset_index()[['f', 't', 'value']]
            demanda = demanda.set_index(['f', 't']).to_dict()['value']

            inventario =  pd.DataFrame([(*key, value) for (key, value) in w_to_s(W, W0, T, F, delta).items()])
            inventario = inventario.rename(columns={0: 'f', 1: 't', 2: 'delta', 3: 'value'})
            inventario = inventario.groupby(['f', 't']).sum().reset_index()[['f', 't', 'value']]
            inventario = inventario.set_index(['f', 't']).to_dict()['value']

            produccion_w = pd.DataFrame([(*key, value) for (key, value) in W_.items()]).rename(columns={0: 'f', 1: 't', 2: 's', 3: 'value'})
            produccion_w = produccion_w.rename(columns={0: 'f', 1: 't', 2: 's', 3: 'value'})

            produccion_w.to_excel('produccion_w.xlsx')

            #produccion_w = produccion_w.set_index(['f', 't', 's']).to_dict()['value']
            # Obtencion de la demanda, el inventario y la produccion

            return demands, production, price, pattern_use, MVPFCF.objVal, MVPFCF.Runtime, inventario, demanda, produccion_w


        else:

            status, opt_value = MVPFCF.Status, MVPFCF.objVal  # Retorno en caso de no guardar 

    return status, opt_value
