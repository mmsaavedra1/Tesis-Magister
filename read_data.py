import string
from typing import Any
import pandas as pd
from pprint import pprint


def read_sheet(
    filename: str,
    sheet: str,
    periods: int = None,
    num_columns: bool = False,
    delete_row_header: bool=False
    ):
    """
    Lee matrices ubicadas en una hoja de excel.
    Input:
    
    - filename: Nombre del archivo.
    - sheet: Nombre de la hoja a extraer informacion.
    - periods: Limitar en un cantidad de periodos temporales.
    - num_columns: Renombrar las columnas. Ej: "T10" -> "10"
    - delete_row_header: Si es True, elimina la primera columna del DataFrame
    
    
    Outpuyt:
    - column_names: Nombre de las columnas de la hoja. -> list
    - row_names: Nombre de las filas de la hoja. -> list
    - final_dict: Valores de la matriz de la hoja. -> dict
    """
    
    df = pd.read_excel(filename, sheet_name=sheet)
    values = df.drop(df.columns[0], 1).values.tolist() #eliminamos la primera columna y la pasamos a una lista
    column_names = df.columns.tolist()[1:] #guardamos nombres de las columnas (períodos)
    if num_columns: #de ser necesario, renombramos las columnas para que sean solo los números (sin el T)
        column_names = list(range(1, len(column_names) + 1))
    row_names = df[df.columns[0]].values.tolist() #nombres de las filas
    if sheet == "Patrones":
        del_row = row_names.pop(-1)

    #el input tiene 90 períodos.  Si queremos tener otro num de períodos, hay que hacer ajustes al input
    if periods and periods != len(column_names):
        prev_last_t = len(column_names) - 1 #último T (=90) antes de hacer cambio de num de períodos

        #si queremos agregar x períodos, copiamos los primeros x períodos "a la derecha" del df
        if periods >= len(column_names):
            cols_to_add = periods - len(column_names) #cantidad de períodos a agregar
            #pegado de los primeros x períodos "a la derecha" del df
            [values[i].append(values[i][j]) for i, row_name in enumerate(row_names) for j in range(cols_to_add)]
            column_names = list(range(1, periods + 1)) #actualizar nombres de las columnas

        #mover costos residuales desde prev_last_t al nuevo último período
        if sheet == "Costo Hold":
            for i, row_name in enumerate(row_names):
                values[i][periods - 1] = values[i][prev_last_t] #nueva posición de costos residuales
                values[i][prev_last_t] = values[i][prev_last_t - 1] #valor de prev_last_t se copia de columna anterior
        #actualizar columnas en caso de que queramos menos de 90 períodos
        if periods <= len(column_names):
            column_names = column_names[:periods]

    #pasamos de una matriz de values al diccionario final a entregar
    final_dict = {row_name: {column_name: values[i][j] for j, column_name in enumerate(column_names)}
                  for i, row_name in enumerate(row_names)}
    
    #si es True, eliminamos primera columna del df original
    if delete_row_header:
        final_dict = final_dict[list(final_dict.keys())[0]]

    return column_names, row_names, final_dict


def scale_matrix(
    data: dict,
    scaler: int,
    power: int) -> dict:
    """
    Amplifica datos de un diccionario, o bien, matriz de dos dimensiones.
    Input:
    - data: Matriz con la informacion.
    - scaler: Base de la potencia. 
    - power: Exponente de la potencia.
    
    Output:
    - data: Matriz con la informacion escalada -> dict.
    """

    for i1 in data.keys():
        for i2 in data[i1].keys():
            data[i1][i2] *= (scaler**power)


def scale_list(
    data: dict,
    scaler: int,
    power: int) -> dict:
    """
    Amplifica datos de un diccionario, o bien, matriz de 1 dimensiones.
    Input:
    - data: Matriz con la informacion.
    - scaler: Base de la potencia. 
    - power: Exponente de la potencia.
    
    Output:
    - data: Matriz con la informacion escalada -> dict.
    """
    for i1 in data.keys():
        data[i1] *= (scaler**power)


def read_all_data(
    filename: str,
    periods: int = None,
    print_out: bool = True,
    scaler: int = None,
    fix_alfa: bool = False,
    q_mult: int = 1,
    alfa_mult: int = 1,
    beta_mult: int = 1,
    res_mult: int = 1,
    S0_mult: int = 1,
    delta_: int = 9
    ) -> Any:
    
    """
    Sirve para leer toda la información de un excel, recorriendo todas sus hojas. Por defecto el Excel sigue un formato en específico para la lectura de los datos.
    
    Input:
    - filename: Nombre del archivo a leer.
    - periods: Limita la cantidad de periodos a leer.
    - print_out: Imprime el output de la lectura.
    - scaler: Amplifica/Simplifica las unidades de medida (cajas de "scaler" productos).
    - fix_alfa: Si es True, fija el parametro de alfa. Si es False, depende del tiempo.
    - q_mult: Valor que amplifica/simplifica los parametros del vector q.
    - alfa_mult: Valor que amplifica/simplifica los parametros del vector alfa.
    - beta_mult: Valor que amplifica/simplifica los parametros del vector beta.
    - res_mult: Valor que amplifica/simplifica los parametros del vector res.
    - S0_mult: Valor que amplifica/simplifica los parametros del vector S0 (inventario inicial).
    
    Output:
    - F: Listado de productos finales -> list.
    - T: Listado de periodos que se puede producir (sin contar el 0) -> list
    - T_0: Listado de periodos que se puede producir más el 0 -> list
    - T_delta: Listado de periodos hasta el ultimo vencimiento de productos (sin contar el 0) -> list
    - T_0_delta: Listado de periodos hasta el ultimo vencimiento de productos más el 0 -> list
    - K: Listado de número de patrón de corte -> list
    - alfa: Matriz de valores de alfa para cada producto final en el tiempo -> dict
    - beta: Matriz de valores de beta para cada producto final en el tiempo -> dict
    - delta: Dias de vencimiento de un producto final -> int
    - S0: Matriz de inventario inicial de cada producto, para cada tiempo -> dict
    - a: Matriz de cantidad de producto final resultante de cada patron de corte -> dict
    - h: Matriz de costos de inventario de productos final para cada periodo -> dict 
    - c: Matriz de costo de corte para cada patron -> dict
    - q: Matriz de total de producto antes de corte para cada periodo -> dict
    - c_merma: Matriz de costo de botar producto final para cada periodo -> dict 
    - p_compat: Matriz de compatibilidad de precio para cada tipo de corte entre productos finales -> dict
    - p_constante: Matriz de precio fijo de producto final para cada periodo -> dict
    - v_constante: Matriz de demanda fija de producto final para cada periodo -> dict
    """

    delta = delta_ #períodos hasta vencimiento
    K, F, a = read_sheet(filename, "Patrones")
    T_keys, _, h = read_sheet(filename, "Costo Hold", periods, num_columns=True)
    _, _, c_merma = read_sheet(filename, "Merma", periods, num_columns=True)
    _, _, p_constante = read_sheet(filename, "Precio constante", periods, num_columns=True)
    _, _, v_constante = read_sheet(filename, "Venta constante", periods, num_columns=True)
    _, _, p_compat = read_sheet(filename, "Restricciones")

    #_, _, a = read_sheet(filename, "Patrones - Piezas")

    #último período en el que se puede producir
    last_t = len(T_keys)
    T = [i+1 for i in range(last_t)] #períodos en los que se puede producir
    T_delta = T + [T[-1] + 1 + i for i in range(delta-1)] #períodos hasta útlima obsolesencia de productos (sin 0)
    T_0 = [0] + T #períodos en los que se puede producir + el inicio
    T_0_delta = [0] + T_delta

    _, _, c =  read_sheet(filename, "Costo Corte", delete_row_header=True)
    _, _, q =  read_sheet(filename, "Insumos", periods, num_columns=True, delete_row_header=True)
    _, _, S0 =  read_sheet(filename, "Inventarios Iniciales", num_columns=True)


    #fix_alfa: si True, toma input de alfa estático (no depende de t) a uno que depende de t.
    #          si es False, lee el alfa en función de t directamente desde el archivo
    if fix_alfa:
        _, _, alfa_y_beta = read_sheet(filename, "Demanda")
        alfa = {f: {i2: alfa_y_beta[f]["alpha"] for i2, t in enumerate(T, 1)} for i, f in enumerate(F, 1)}
        beta = {f: {i2: alfa_y_beta[f]["beta"] for i2, t in enumerate(T, 1)} for i, f in enumerate(F, 1)}
    else:
        _, _, alfa = read_sheet(filename, "Alfa", periods, num_columns=True)
        _, _, beta = read_sheet(filename, "Beta", periods, num_columns=True)

    if scaler:
        for data, power in zip([h, alfa, beta, S0], [1, -1, -2, -1]):
            # print(data)
            # print(scaler, power)
            scale_matrix(data, scaler, power)
        for data, power in zip([c, q], [1, -1]):
            scale_list(data, scaler, power)

    if q_mult != 1:
        scale_list(q, q_mult, 1)

    if alfa_mult != 1:
        scale_matrix(alfa, alfa_mult, 1)

    if beta_mult != 1:
        scale_matrix(beta, beta_mult, 1)

    if S0_mult != 1:
        scale_matrix(S0, S0_mult, 1)

    if res_mult != 1:
        for data in h.values():
            data[last_t] *= res_mult


    if print_out:
        pprint(f"Productos strings: {F}")
        pprint(f"last_t: {last_t}")
        pprint(f"T: {T}")
        pprint(f"T_delta: {T_delta}")
        pprint(f"T_0: {T_0}")
        pprint(f"T_0_delta: {T_0_delta}")
        pprint(f"Patrones K: {K}")
        pprint(f"Costos de hold: {h}")
        pprint(f"Cantidad de productos por patron: {a}")
        pprint(f"Costo corte: {c}")
        pprint(f"Alfa: {alfa}")
        pprint(f"Beta: {beta}")
        pprint(f"Inventarios iniciales: {S0}")
        pprint(f"Insumos: {q}")
        pprint(f"Costo_merma: {c_merma}")

    # if return_h_acum:
    #     h_acum = calc_h_acum(h, delta, T, F)
    #     return F, T, T_delta, T_0, T_0_delta, K, alfa, beta, delta, S0, a, h, c, q, h_acum

    return F, T, T_delta, T_0, T_0_delta, K, alfa, beta, delta, S0, a, h, c, q, c_merma, p_constante, p_compat, v_constante


def calc_h_acum(
    h: dict,
    delta: int,
    T: list,
    F: list) -> dict:
    """
    Sirve para sumar el costo de inventario acumulado desde la producción de producto en t hasta el periodo s, de venta o merma. Equivale al parametro "g" del modelo matemático.
    
    Input: 
    - h: Matriz de costos de inventario de productos final para cada periodo.
    - delta: Dias de vencimiento de un producto final.
    - T: Listado de periodos que se puede producir (sin contar el 0).
    - F: Listado de productos finales.
    
    Output:
    - h_acum: Matriz de costo de inventario acumulado, desde t hasta el tiempo s, para un producto final -> dict.
    """
    
    h_acum = {f:{t:{s:0 for s in range(t + 1, min(T[-1] + 2, t + delta))} for t in T} for f in F}

    for f in F:
        for t in T:
            for s in range(t + 1, min(T[-1] + 2, t + delta)):
                h_acum[f][t][s] = sum([h[f][i] for i in range(t, s)])
    return h_acum


def save_xls(
    sheets_dict: pd.DataFrame,
    out_string: str) -> None:
    """
    Sirve para guardar varias sheets en forma de dict, dentro de un excel.
    
    Input: 
    - sheets_dict: Dataframe con toda la informacion que se desea guardar en un excel.
    - out_string: Texto que contiene la ruta de donde se guardará el archivo.
        
    """
    with pd.ExcelWriter(out_string) as writer:
        for name, df in zip(sheets_dict.keys(), sheets_dict.values()):
            df.reset_index(drop=True, inplace=True)
            df.to_excel(writer, name, index=False)
        writer.save()


def alfa_beta_over_time(
    alfa_y_beta: dict,
    F: list,
    T: list,
    mults: list,
    string: string,
    base_alfa_beta: dict = None):
    """
    Funcion auxiliar que recibe vectores de alfa y beta constantes y los hace variar en el tiempo.
    Puede recibir una lista base para ser ponderada por los valores seteados en la lista "mults".
    
    Input:
    - alfa_y_beta: dict con valores de alfa y beta
    - mults: lista con multiplicadores que ponderando los valores en el tiempo especifico del indice.
    - string: admite valores "alpha" o "beta", e indica que lista se pondera
    - base_alfa_beta: base de valores de alfa y beta que se ponderan el tiempo.
    
    Output:
    - df: DataFrame que contiene la matriz de valores de alfa y beta, ya ponderados. -> DataFrame()
    """
    
    rows = []
    column_names = [""] + [f"T{i}" for i in T]
    if base_alfa_beta:
        for i, f in enumerate(F, 1):
            rows.append([f] + [mult * base_alfa_beta[f][string] for mult in mults])
    else:
        for i, f in enumerate(F, 1):
            rows.append([f] + [mult * alfa_y_beta[f][string] for mult in mults])
    df = pd.DataFrame(rows, columns=column_names)
    return df


def q_over_time(df, q_mult: dict):
    """
    Funcion auxiliar que recibe la matriz de "q" (cantidades de producto inicial) para cada periodo del tiempo,
    y pondera segun los valores de la lista q_mult.
    
    Input:
    - df: Matriz original de valores de "q"
    - q_mult: Lista de ponderadores de produccion para q.
    
    Output:
    - df: Matriz de valores de "q" ya ponderados por la lista "q_mult" -> DataFrame
    """
    df.set_index(df.columns[0], inplace=True)
    df = df.apply(lambda x: x*q_mult[int(x.name[1:]) - 1], axis=0)
    df.reset_index(inplace=True)
    return df


def set_costo_hold(df, costo_hold: dict):
    """
    Funcion auxiliar que recibe la matriz de costos de inventario y una propuesta de nuevos costos de inventario.
    
    Input:
    - df: Matriz de valores actuales de costos de inventario.
    - costo_hold: diccionario de nuevos valores de costo de inventario.
    
    Output:
    - df: Matriz de valores actualizados para los costos de inventario. -> DataFrame
    """
    df.set_index(df.columns[0], inplace=True)
    df = df.apply(lambda x: x*0 + costo_hold[x.name], axis=1) #ponemos el costo_hold que queremos
    df.T90 = -5*df.T89 #valor residual en la última columna
    df.reset_index(inplace=True)

    return df


def scale_inventario_inicial(df, scale_s0: dict):
    """
    Funcion auxiliar que pondera la matriz (diccionario) inventario inicial de cada producto final, según los valores de la lista "scale_s0".
    
    Input:
    - df: Matriz de valores de inventario inicial de producto final para cada periodo de tiempo.
    - scale_s0: diccionario con ponderadores para el inventario inicial de cada producto.
    
    Output:
    - df: Matriz de valores ponderados de inventario inicial de producto final para cada periodo de tiempo. 
    
    """
    df.set_index(df.columns[0], inplace=True)
    df = df.apply(lambda x: x /scale_s0[x.name], axis=1) #name es nombre de la fila (en esta caso es el producto f)
    df.reset_index(inplace=True)

    return df


def cambiar_costo_corte(patrones, costo_corte, corte_prop):
    """
    Funcion auxiliar que asigna nuevos costos (fijos y variables) para los patrones de corte.
   
    Input:
    - patrones: Matriz que contiene todos los cortes que se pueden realizar.
    - costo_corte: Matriz de costos originales para cada corte.
    - corte_prop: Matriz nuevos costos que se proponen (tanto fijos como variables).
    
    Output:
    - costo_corte: Matriz de costos actualizadas a cada corte neuvo.
    
    """
    costo_corte.set_index(costo_corte.columns[0], inplace=True)
    costo_corte = costo_corte.apply(lambda x: 0*x + corte_prop["Fijo"] +
                                              corte_prop["Variable"]*(patrones[x.name].sum() - 1), axis=0)
    costo_corte.reset_index(inplace=True)
    return costo_corte


def generate_excel(
    in_string,
    out_string,
    alfa_mult=None,
    beta_mult=None,
    base_alfa_beta=None,
    costo_hold=None,
    scale_s0=None,
    q_mult=None,
    corte_prop=None):
    """
    Funcion que genera un nuevo Excel, tomando un Excel base y generando las modificaciones de valores que desee el usuario.
    
    Input:
    - in_string: Ruta del Excel origial.
    - out_string: Ruta del nuevo Excel modificado.
    - alfa_mult: Listado para escalar los parametros de alfa.
    - beta_mult: Listado para escalar los parametros de beta.
    - base_alfa_beta: Listado original de valores de alfa_beta que se desean aplicar en el nuevo Excel.
    - costo_hold: Listado de nuevo costos de inventario que se desea otorgar al Excel.
    - scale_s0: Ponderador de valores para el inventario incial.
    - q_mult: Listado de ponderadores para el listado de produccion "q".
    - corte_prop: Listado de nuevos costos (fijos y variables) que se desean otorgar a los patrones de corte.
    """

    sheets_dict = pd.read_excel(in_string, sheet_name=None, decimal=",") #dict {sheet_name: df}

    #Escalar alfa a través del tiempo
    if alfa_mult:
        T = list(range(1, len(alfa_mult) + 1))
        aux, F, alfa_y_beta = read_sheet(in_string, "Demanda")
        sheets_dict["Alfa"] = alfa_beta_over_time(alfa_y_beta, F, T, alfa_mult, "alpha", base_alfa_beta)
        sheets_dict["Beta"] = alfa_beta_over_time(alfa_y_beta, F, T, beta_mult, "beta", base_alfa_beta)

    #cambiar costos de hold (estático)
    if costo_hold:
        df = sheets_dict["Costo Hold"]
        sheets_dict["Costo Hold"] = set_costo_hold(df, costo_hold)

    #escalar inventario inicial
    if scale_s0:
        df = sheets_dict["Inventarios Iniciales"]
        sheets_dict["Inventarios Iniciales"] = scale_inventario_inicial(df, scale_s0)

    #variar insumo q en el tiempo
    if q_mult:
        df = sheets_dict["Insumos"]
        sheets_dict["Insumos"] = q_over_time(df, q_mult)

    #cambiar costos de corte (estáticos)
    if corte_prop:
        patrones = sheets_dict["Patrones"]
        costo_corte = sheets_dict["Costo Corte"]
        sheets_dict["Costo Corte"] = cambiar_costo_corte(patrones, costo_corte,corte_prop)

    save_xls(sheets_dict, out_string) #guardar todos los sheets en un nuevo excel


#solo descomentar bloques de más abajo para generar nuevos excel (cuidado que puede borrar excels anteriores)
if __name__ == "__main__":
    pass
    # ponderadores = [0.8, 1.1, 1.4, 1.3, 1, 0.7, 0.7, 1, 1] #sube y despues baja
    # ponderadores = [2 - i for i in ponderadores] #baja y despues sube (espejo)
    # q_pond = [1.2, 1.2, 0.9, 0.7, 0.7, 0.8, 1.1, 1.2, 1.2]
    # print(sum(q_pond))
    # print(sum(ponderadores))
    # # beta_mult = [1/ponderadores[i] for i in range(math.ceil(len(T)/10))]
    # alfa_mult = []
    # beta_mult = []
    # base_alfa_beta = { "Entero": {"alpha": 0, "beta": 0}, "Medio": {"alpha": 21000, "beta": 20},
    #                    "Cuarto": {"alpha": 23500, "beta": 25}, "Octavo": {"alpha": 0, "beta": 0}}
    #
    # # base_alfa_beta = {"Entero": {"alpha": 20000, "beta": 15}, "Medio": {"alpha": 21000, "beta": 20},
    # #                   "Cuarto": {"alpha": 23500, "beta": 25}, "Octavo": {"alpha": 25000, "beta": 35}}
    # #
    # q_mult = []
    # for i in range(9):
    #     alfa_mult += [ponderadores[i] for j in range(10)]
    #     beta_mult += [1/ponderadores[i] for j in range(10)]
    #     q_mult += [q_pond[i] for j in range(10)]
    #
    #
    #
    # costo_hold = {"Entero": 40, "Medio": 20, "Cuarto": 10, "Octavo": 5} #Pasa a ser esto
    # scale_s0 = {"Entero": 24, "Medio": 12, "Cuarto": 6, "Octavo": 3} #se divide por esto
    # corte_prop= {"Fijo": 240, "Variable": 17}
    #
    #
    # alfa_mult = [1 for i in range(90)]
    # beta_mult = [1 for i in range(90)]
    # q_mult = None
    # # base_alfa_beta = {"Entero": {"alpha": 0, "beta": 0}, "Medio": {"alpha": 21000, "beta": 20},
    # #                   "Cuarto": {"alpha": 0, "beta": 0}, "Octavo": {"alpha": 0, "beta": 0}}
    #
    # string = "Alfabs_betasb"
    # # generate_excel("Input/Ejemplo Realista.xlsx", f"Input/{string}.xlsx", alfa_mult=alfa_mult, beta_mult=beta_mult,
    # #                base_alfa_beta=base_alfa_beta, costo_hold=costo_hold, scale_s0=scale_s0, q_mult=q_mult, corte_prop=corte_prop)


    # scaler = 200
    # F_subset = ["Medio", "Cuarto"]
    # periods = 130
    # F, T, T_delta, T_0, T_0_delta, K, alfa, beta, delta, S0, a, h, c, q = \
    #     read_all_data("Input/Constante.xlsx", periods=periods, scaler=scaler, print_out=True, fix_alfa=False)
    # h_acum = calc_h_acum(h, delta, T, F)
