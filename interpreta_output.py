import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from read_data import *
import os, shutil
import statistics
import datetime as dt
from cycler import cycler
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1200)
pd.options.mode.chained_assignment = None  # default='warn'

#retorna un df con los valores de x_vars que no son 0
def get_vars_df(model_name, case, vars, column_names, string, save=True):

    df = pd.DataFrame.from_dict(vars, orient="index",
                                    columns = ["value"])
    df.index = pd.MultiIndex.from_tuples(df.index,
                                   names=column_names)
    df.reset_index(inplace=True)
    if save:
        # print(f"string: {string}")
        # print(df)
        df.to_pickle(f"{model_name}/Output/Variables/{case}/{string}.pkl")
    return df


#genera un diccionario a partir de las variables de input
def get_vars_dict(vars):
    if not vars:
        return None
    else:
        return {x: vars[x].X for x in vars}


#toma un dict y cambia el orden de las keys (usado con alfa y beta en el modelo)
def change_dict_keys(data):

    new_data = {}
    for f, val1 in zip(data.keys(), data.values()):
        for t, val2 in zip(val1.keys(), val1.values()):
            new_data[(f, t)] = val2
    return new_data


#applica get_vars_dict on las vars de la lista all_vars
def get_all_vars_dict(all_vars):

    return [get_vars_dict(vars) for vars in all_vars]


#entrega el df respectivo.  Si agg=True, se agrega sobre las dimensiones f y t (se elimina la dimension u)
def fetch_df(model_name, case, string, agg=False, end_t=False):

    df = pd.read_pickle(f"{model_name}/Output/Variables/{case}/{string}.pkl")
    if end_t:
        df = df[df.t <= end_t]
    if agg:
        df = df.groupby(["f", "t"])["value"].agg("sum").reset_index()
    return df


#aplica fecth_df a cada string de la lista string
def fetch_all_dfs(model_name, case, strings, aggs, name, end_t=False, prefix=""):

    return [fetch_df(model_name, case, f"{prefix}{name}_{string}", agg, end_t) for string, agg in zip(strings, aggs)]


#lee un excel y retorna un diccionario con la cantidad de productos f producidos en un patrón k
def cortes_por_patron():

    df = pd.read_excel(f"Input/{name}.xlsx", sheet_name="Patrones")
    df.rename(columns={'Unnamed: 0': 'Patron'}, inplace=True)
    df = df.to_dict()
    a = {}
    # print(df)
    for i, product in zip(df["Patron"].keys(), df["Patron"].values()):
        a[product] = {}
        for key, val in zip(df.keys(), df.values()):
            if key != "Patron":
                a[product][key] = val[i]

    return a


#recibe los cortes por patrón (cortes), producción X y productos F y retorna la producción de cada producto por día
def produccion_por_dia(cortes, X, F, end_t=None):

    for f in F:
        cortes_f = cortes[f]
        X[f] = X.apply(lambda row: row.value*cortes_f[row.k], axis=1)
    prod = X[["t"] + F].groupby("t").sum().reset_index()
    #pivotear la matriz (wide to tall matrix)
    prod = pd.melt(prod, id_vars=['t'], value_vars= F, var_name='f')
    if end_t:
        prod = prod[prod.t <= end_t]
    # prod = pd.melt(prod, id_vars=['t'], value_vars=["Entero", "Medio", "Cuarto", "Octavo"], var_name='f')
    return prod


#toma un producto f y una variable (S o D) y retorna las variables agregadas por tiempo
def inventario_por_dia(f, S):

    gb = S[S.f == f].groupby("t")["value"].agg("sum")
    return gb


#toma un producto f y una variable (P, alfa o beta) y retorna el df correspondiente
def precio_por_dia(f, P):

    df = P[P.f == f]
    df = df.set_index("t")
    return df


#plot principal
def plot_f_t(model_name, F_subset, v1s, v2s, strings1, strings2, name, case, save=False, title=None, file_out=None):

    #definir algunos colores, linestyles y markers
    colors = ["r", "b", "g", "c", "y"]
    linestyles = ["solid", 'dashed', 'dotted', 'dashdot', ":"]
    markers = [".", "+", "v", 3, "x", "|"]

    for f, color, color2 in zip(F_subset, colors, colors[1:]):
        for v1, string1, linestyle in zip(v1s, strings1, linestyles):
            if f in F_subset:
            # if f in F_subset or string1 in ["Inventario"]:

                gb = v1[v1.f == f]["value"]
                gb.index = np.arange(1, len(gb) + 1)

                if string1 in ["Ventas", "Merma"]: #a merma le ponemos un estilo distinto, porque me quedé sin opciones de style
                    ax1 = gb.plot(style=f'{color}{markers[-2]}', label=f"{string1}")
                    # ax1 = gb.plot(style=f'{color}{markers[-2]}', label=f"{string1} {f}")
                else:
                    ax1 = gb.plot(color=color, linestyle=linestyle, label=f"{string1}")
                    # ax1 = gb.plot(color=color, linestyle=linestyle, label=f"{string1} {f}")

        for v2, string2, marker in zip(v2s, strings2, markers):
            if f in F_subset:
                df = precio_por_dia(f, v2)
                df.index = np.arange(1, len(df) + 1)
                ax2 = df.value.plot(style=f'{color}{marker}', label=f"{string2}", secondary_y=True)
                # ax2 = df.value.plot(style=f'{color}{marker}', label=f"{string2} {f}", secondary_y=True)

    scale_y = 1
    scale_x = 1
    if not save: #dejar más ancho el exe x para que quepa la leyenda (solo si es que se plottea)
        scale_x = 1.15
    ax1.set_ylabel("Cajas")
    ax2.set_ylabel("Dinero")
    if title:
        plt.title(title)
    else:
        plt.title(f"Modelo {model_name} escenario {name} caso {case}")

    ax1.set_ylim(ymin=0, ymax=ax1.get_ylim()[1]*scale_y)
    ax1.set_xlim(xmin=0, xmax=ax1.get_xlim()[1]*scale_x)
    ax2.set_ylim(ymin=0, ymax=ax2.get_ylim()[1] * scale_y)
    ax2.set_xlim(xmin=0, xmax=ax2.get_xlim()[1] * scale_x)

    ax1.legend(loc=1)
    ax2.legend(loc=4)
    if save:
        plt.savefig(f"{model_name}/Figures/{case}/png/{file_out}.png")
        plt.savefig(f"{model_name}/Figures/{case}/pdf/{file_out}.pdf")
        # plt.savefig(f"{model_name}/Figures/{case}/png/{name}.png")
        # plt.savefig(f"{model_name}/Figures/{case}/pdf/{name}.pdf")
    else:
        plt.show()
    plt.close()


def plot_dif_cases(model_name, cases, strings, aggs, name, f, end_t=None):

    merged_dfs = {s: [] for s in strings}
    for case in cases:
        out_dfs = fetch_all_dfs(model_name, case, strings, aggs, name, end_t)
        for i, s in enumerate(strings):
            df = out_dfs[i]
            df = df[df.f == f]
            df["case"] = case
            if len(merged_dfs[s]) > 0:
                merged_dfs[s] = pd.concat([merged_dfs[s], df],ignore_index=True)
            else:
                merged_dfs[s] = df

    for s in strings:
        merged_dfs[s] = merged_dfs[s].pivot(index='t', columns='case', values='value')
        merged_dfs[s].plot()
        plt.title(f"{s} producto {f} para distintos casos")

    plt.show()


def plot_prod(string, df, case, F=None, out_file=None, save=False, add_legend=False):

    # colormaps: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2,
    # Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired,
    # Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd,
    # PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds,
    # Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r,
    # YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r,
    # brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix,
    # cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar,
    # gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2,
    # gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r,
    # nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r,
    # seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r,
    # terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, winter, winter_r

    if not F:
        F = ["Medio Pollo", "Entero", "Medio Pollo Superior", "Cuarto de Pollo Superior", "Muslo Entero",
             "Media Pechuga", "Ala Completa", "Blanqueta", "Pechuga Completa", "Alón", "Punta", "Jamoncito", "Medio Muslo"]

    #generamos diccionario de venta total por producto, para ordenar los labels
    gb = df.groupby("f")["value"].sum().reset_index()
    total = {row.f: row.value for i, row in gb.iterrows()}

    #pivotear df y plot
    df = df.pivot(index='t', columns='f', values='value')
    df = df[F]
    styles = ["1", "4", "|", "d", "3", "*", "_", "o", "+", ".", "x", "2",  "p"]
    # styles = ["1", "<", ">", "^", ".", "3", "_", "v", "+", "*", "x", "2",  "p"]
    ax = df.plot(colormap='prism_r', grid=False, style=styles, legend=add_legend)
    # ax = df.plot(kind="scatter", x="f", y="value", colormap='Paired')
    # plt.title(f"{string} (cajas) para caso {case}")
    plt.ylabel(f"{string}")
    plt.xlabel(f"Periods since start")

    if add_legend:
        #ordener labels del legend
        handles, labels = list(zip(ax.get_legend_handles_labels()))
        handles_and_labels = list(zip(handles[0], labels[0]))
        # handles_and_labels.sort(key=lambda x: x[0], reverse=True)
        handles_and_labels.sort(key=lambda x: total[x[1]], reverse=True)
        handles, labels = list(zip(*handles_and_labels))
        ax.legend(handles, labels, loc='best', ncol=3)
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
        # ax.legend(handles, labels, bbox_to_anchor=(0.8, 0.8))

    if save:
        plt.savefig(f"{model_name}/Figures/{case}/png/{out_file}.png")
        plt.savefig(f"{model_name}/Figures/{case}/pdf/{out_file}.pdf")
    else:
        plt.show()


#a partir de alfa y beta, calcula y retorna el precio maximo por día
def fetch_precio_maximo(alfa, beta):

    precio_maximo = alfa.copy()
    precio_maximo["value"] = alfa.value/beta.value
    return precio_maximo


#lee el output de string_input y calcula el profit que se obtendría si fueramos un intermediario que
#solo compra y vende
#retorna el string_input, el valor dado por el programa, el valor en caso intermediario y la proporción entre ambos
def calc_max_profit(model_name, string_input, ps, q_mult=1, alfa_mult=1, beta_mult=1):

    F, T, T_delta, T_0, T_0_delta, K, alfa, beta, delta, S0, a, h, c, q, c_merma, p_constante, p_compat, v_constante = \
        read_all_data(f"Input/{string_input}", scaler=None, print_out=False, fix_alfa=False, q_mult=q_mult,
                      alfa_mult=alfa_mult, beta_mult=beta_mult)
    costos = {}
    profits = {}

    for i, f in enumerate(F):
        # print(f"f: {f}")
        if alfa[f][1] > 1:
            k = ps[i] #corte en el que se produce solo del producto f
            costos[f] = c[k]/a[f][k] #costo por pieza

            #precio de venta, demanda y profit obtenido de derivar f obj en función del precio e igualar a 0
            price = [(alfa[f][t] + costos[f]*beta[f][t])/(2*beta[f][t]) for t in T]
            demand = [alfa[f][t] - (alfa[f][t] + costos[f]*beta[f][t])/2 for t in T]
            profits[f] = sum([demand[t - 1]*(price[t - 1]- costos[f]) for t in T])

    total = sum(list(profits.values())) #sumamos sobre todos los T
    df = pd.read_pickle(f"{model_name}/Output/Results/{string_input[:-5]}_results.pkl")
    return string_input, df.value[0], total, df.value[0]/total


#genera una tabla donde para cada fila sale el nombre del modelo, la rentabilidad del modelo, rentabilidad siendo
#intermediario y la proporción entre ambos
def profit_df(model_name, ps):

    columns = ["Modelo", "Cutting_stock", "Compra-venta", "Proporcion"]
    rows = []
    for filename in os.listdir("Input/"):
        if filename[-5:] == ".xlsx" and filename != "Ejemplo Realista.xlsx":
            rows.append(calc_max_profit(model_name, filename, ps))

    df = pd.DataFrame(rows, columns=columns)
    print(df)


def summary_table(model_name, filename, F_subset, strings, aggs, cases, out_suffix="", save=True, start_t=None, end_t=False,
                  periods_exec=None, div=10**6):

    # cases = [5]
    # cases = [0, 1, 4, 5, 6, 7, 8]
    case_names = {0: "Base case", 1: "Constant price", 4: "No price compatibility", 5: "Dynamic demand", 6: "Weekly price",
                  7: "Dynamic quantity", 8: "Dynamic demand and quantity"}
    data = {"Caso": [], "Rentabilidad": [], "Ingresos": [], "Ingreso residual": [], "C_inv_no_inicial": [], "C_inv_inicial": [], "C_produccion": [],
            "C_merma": []}
    # to_calc = ["Venta", "Producción"]
    to_calc = ["Venta", "Producción", "Merma", "Precio", "Inventario"]
    indicadores = ["Ingresos", "Ingreso residual", "C_inv_no_inicial", "C_inv_inicial", "C_produccion", "C_merma"]
    kpis = ["Profit", "Revenue", "Inventory costs", "Production costs", "Waste costs"]
    # kpis = ["Rentabilidad", "Ingresos", "Costo_inv", "Costo_prod", "Costo_merma"]
    data = {key: [] for key in ["Case"] + kpis}
    # for f in F_subset:
    #     for item in to_calc:
    #         data[f"{item} {f}"] = []
    for case in cases:
        data["Case"].append(case_names[case])
        # df = pd.read_pickle(f"{model_name}/Output/Results/{case}/{filename}_results.pkl")
        # print(f"Results: {df}")

        # data["Rentabilidad"].append(df.value[0])
        # if periods_exec:
        #     data["Rentabilidad"][-1] /= periods_exec
        # for indicador in indicadores:
        #     num = df[indicador][0]
        #     if periods_exec:
        #         num /= periods_exec
        #     data[indicador].append(num)

        path = f"Input/{name}.xlsx"
        aux, aux2, c = read_sheet(path, "Costo Corte", delete_row_header=True)
        T_keys, aux, h = read_sheet(path, "Costo Hold", periods_exec, num_columns=True)
        aux, aux2, c_merma = read_sheet(path, "Merma", periods_exec, num_columns=True)

        if case < 5:
            X, S, D, P, L, beta, alfa = fetch_all_dfs(model_name, case, strings, aggs, name, end_t=end_t)
            out = calc_profit(X, c, S, h, D, P, L, c_merma, start_t, end_t)
            prod = produccion_por_dia(cortes, X, F_subset)
            all_dfs = [D, prod, L, P, S]
        else:
            strings_iterator = ["X", "S", "D", "P", "L", "D_real", "sales", "prod"]
            iter_aggs = [False for i in range(len(strings_iterator))]
            X, S, D, P, L, D_real, sales, prod = fetch_all_dfs(model_name, 0, strings_iterator, iter_aggs, name,
                                                               end_t=end_t, prefix=f"Iteration/{case}/")
            out = calc_profit(X, c, S, h, sales, P, L, c_merma, start_t, end_t)

            all_dfs = [sales, prod, L, P, S]

        for key, val in zip(kpis, out):
            data[key].append(val/div)

        # for f in F_subset:
        #     for item, item_df in zip(to_calc, all_dfs):
        #         data[f"{item} {f}"].append(item_df[item_df.f == f].value.mean())
    final_df = pd.DataFrame.from_dict(data)
    final_df = final_df.round(decimals=1)
    if save:
        final_df.to_excel(f"{model_name}/Output/Summary/Resumen_{out_suffix}_{filename}.xlsx")
    else:
        print(final_df)


def summary_chart(model_name, filename, cases, strings, F_subset, aggs_dict, xlabel=None, ylabel=None, title=None, out_file=None, save=True, end_t=False, threshold=None):


    to_calc = ["Venta", "Producción", "Merma", "Precio", "Inventario"]
    indicadores = ["Ingresos", "Ingreso residual", "C_inv_no_inicial", "C_inv_inicial", "C_produccion", "C_merma"]
    data = {"case": [], "f": [], "value": []}
    for i, case in enumerate(cases):
        item_df = fetch_df(model_name, case, f"{filename}_{strings[0]}", aggs_dict[strings[0]], end_t)
        if len(strings) > 1:
            df2 = fetch_df(model_name, case, f"{filename}_{strings[1]}", aggs_dict[strings[1]], end_t)
            item_df.value = item_df.value*df2.value
        # prod = produccion_por_dia(cortes, X, F_subset)
        # all_dfs = [D, prod, L, P, S]
        for f in F_subset:
            data["case"].append(case)
            data["f"].append(f)
            data["value"].append(item_df[item_df.f == f].value.mean())
        if i == 0:
            order_df = pd.unique(pd.DataFrame.from_dict(data).sort_values(by="value", ascending=False).f)
    df = pd.DataFrame.from_dict(data)
    # df.groupby(["case", "f"]).value.unstack().plot(kind="bar", stacked=True)
    pivot = df.pivot_table(index="case", columns="f", values="value")
    pivot.columns = pd.CategoricalIndex(pivot.columns.values, ordered=True, categories=order_df)
    pivot = pivot.sort_index(axis=1)
    # ax = df.plot.scatter(x="case", y="value", c="f")

    # colormaps: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2,
    # Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired,
    # Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd,
    # PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds,
    # Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r,
    # YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r,
    # brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix,
    # cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar,
    # gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2,
    # gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r,
    # nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r,
    # seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r,
    # terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, winter, winter_r

    ax = pivot.plot(kind="bar", stacked=True, colormap='terrain')
    # ax = df.plot(kind="scatter", x="case", y="value", colormap='Paired')

    for lbl in ax.patches:
        if lbl.get_height() > threshold:
            ax.annotate(f"{int(lbl.get_height())}", (lbl.get_x() + .2, lbl.get_y() - 1), fontsize=10, color='black')

    handles, labels = list(zip(ax.get_legend_handles_labels()))
    handles_and_labels = list(zip(handles[0], labels[0]))
    handles_and_labels.reverse()
    handles, labels = list(zip(*handles_and_labels))
    ax.legend(handles, labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save:
        plt.savefig(f"{model_name}/Figures/{0}/png/{out_file}.png")
        plt.savefig(f"{model_name}/Figures/{0}/pdf/{out_file}.pdf")
    else:
        plt.show()

    # groups = df.groupby("f")
    # for name, group in groups:
    #     ax = plt.plot(group["case"], group["value"], marker="o", linestyle="", label=name)
    # if save:
    #     final_df.to_excel(f"{model_name}/Output/Summary/Resumen_{filename}.xlsx")
    # else:
    #     print(final_df)



#plottea los experimentos Q, Alfa, Beta y Periods
def experiment_plotter(model_name, case, experiment, name, ps, base_index=-1, save=False, out_file=None, xlabel=None):

    indexes = []
    values = []
    rep_values = {}
    this_index = None
    base_value = 0
    # print(f"{model_name}/Output/Experiments/{case}/{experiment}/")
    # print(os.listdir(f"{model_name}/Output/Experiments/{case}/{experiment}/"))
    for filename in os.listdir(f"{model_name}/Output/Experiments/{case}/{experiment}/"):
        if filename[-1] == "l":
            print(filename)
            df = pd.read_pickle(f"{model_name}/Output/Experiments/{case}/{experiment}/{filename}")
            # print(df)
            if experiment == "Q":
                this_index = df.q[0][1]
                indexes.append(this_index)
            elif experiment == "Alfa":
                this_index = df.alfa_mult[0]
                indexes.append(this_index)
            elif experiment == "Beta":
                this_index = df.beta_mult[0]
                indexes.append(this_index)

            if experiment == "Periods":
                value = df.exec_time[0]
                index = df.periods[0]
                try:
                    rep_values[index].append(value)
                except KeyError:
                    rep_values[index] = [value]
            else:
                values.append(df.value[0])
                out_string = "Profit/(Base case profit)"
                if this_index == base_index:
                    base_value = df.value[0]

    #para Q, calculamos el profit max (con modelo compra-venta)
    if experiment == "Q":
        pass
        # best_results = calc_max_profit(model_name, f"{name}.xlsx", ps)[2]
        # plt.hlines(best_results, min(indexes), max(indexes), colors="r", label="Rentabilidad compra-venta")
        # plt.legend(loc=4)

    #si es Periods, tenemos que agrupar por número de periods, para despúes promediar
    elif experiment == "Periods":
        out_string = "Total execution time (sec.)"
        values = [statistics.mean(val) for val in rep_values.values()]
        indexes = list(rep_values.keys())

    if experiment in ["Beta", "Q", "Alfa"]:
        for i in range(len(values)):
            values[i] /= base_value

    df = pd.DataFrame.from_dict({"indexes": indexes, "values": values})
    df = df.sort_values(by=["indexes"])
    ax = df.plot(x="indexes", y="values", grid=False, marker=".", markersize=12)
    ax.get_legend().remove()
    plt.ylim(bottom=0, top=max(values)*1.1)
    plt.xlim(left=0, right=max(indexes)*1.1)
    # plt.title(f"Modelo {model_name}, {out_string} vs {experiment}")
    plt.xlabel(xlabel)
    plt.ylabel(f"{out_string}")
    if save:
        plt.savefig(f"{model_name}/Figures/{case}/png/{out_file}.png")
        plt.savefig(f"{model_name}/Figures/{case}/pdf/{out_file}.pdf")
    else:
        plt.show()


#plottea q en el tiempo
def plot_q(model_name, filename, scaler=1, save=True):

    aux, aux2, q = read_sheet(filename, "Insumos", num_columns=True, delete_row_header=True)

    if scaler != 1:
        scale_list(q, scaler, -1)

    plt.plot(list(q.keys()), list(q.values()))
    plt.ylim(bottom=0, top=max(list(q.values()))*1.1)
    plt.xlim(left=0, right=max(list(q.keys())))
    plt.title("Insumo (cajas) para input Varia_insumo")

    if save:
        plt.savefig(f"{model_name}/Figures/q_varia.png")
        plt.savefig(f"{model_name}/Figures/q_varia.pdf")
    else:
        plt.show()
    plt.close()


#plottea los experimentos S0 y Res
#vars_strings: nombres de las variables
#aggs: lista de bools, que si son True, siginifica que hay que agregar dicha variable (eliminar dimensión u)
#experiment: nombre del experimento
#min_t max_t: rango de períodos entre los cuales plottear
#save: si es True, se guarda.  Sino, se plottea
def plot_s_res_experiment(model_name, case, vars_strings, aggs, experiment, f, min_t, max_t, save=False, out_file=None):

    final_data = {}
    colors = list(mcolors.TABLEAU_COLORS.values()) #lista de colores
    fig, ax = plt.subplots(1)

    for string, agg in zip(vars_strings, aggs):
        final_data[string] = {"x": [], "y": []}
        x_data = final_data[string]["x"]
        y_data = final_data[string]["y"]
        for i, filename in enumerate(os.listdir(f"{model_name}/Output/Variables/{case}/{experiment}/{string}/")):
            if filename[0] != ".":
                df = fetch_df(model_name, case, f"{experiment}/{string}/{filename[:-4]}", agg=agg)
                df = df[df.f == f]
                x_df = pd.read_pickle(f"{model_name}/Output/Experiments/{case}/{experiment}/{filename}")
                x = x_df[f"{experiment}_mult"][0]
                x_data.append(x)
                y_data.append(df)
                if i <= 9:
                    plt.scatter(df.t[min_t:max_t], df.value[min_t:max_t], color=colors[i], label=x, s=5)

        plt.grid()
        #ordenar los labels para legend
        handles, labels = list(zip(ax.get_legend_handles_labels()))
        handles_and_labels = list(zip(handles[0], labels[0]))
        handles_and_labels.sort(key=lambda x: float(x[1]))
        handles, labels = list(zip(*handles_and_labels))

        ax.legend(handles, labels)
        # plt.ylim(bottom=0)
        plt.xlabel(f"Período")
        plt.ylabel(f"{string} producto {f}")
        plt.title(f"Modelo {model_name} variando multiplicador de {experiment}")
        # plt.ylim(ymin=0, ymax=1.2*max(df.value))
        plt.xlim(xmin=0)

        if save:
            plt.savefig(f"{model_name}/Figures/{case}/png/{out_file}.png")
            plt.savefig(f"{model_name}/Figures/{case}/pdf/{out_file}.pdf")
        else:
            plt.show()


def plot_patrones(X, out_file=None, cortes=None, f=None, case=None, save=False, xlabel=None, ylabel=None):

    if cortes:
        cortes = cortes[f]
        # print(cortes)
        X["value"] = X.apply(lambda  row: row.value*cortes[row.k], axis=1)
        # print(X)

    prod_by_t = X.groupby("t")["value"].agg("sum").reset_index().to_dict()
    prod_by_t = {prod_by_t["t"][key]: prod_by_t["value"][key] for key in prod_by_t["t"].keys()}

    prod_by_k = X.groupby("k")["value"].agg("sum").reset_index()
    prod_by_k = prod_by_k[prod_by_k.value > 0].to_dict()["k"].values()
    # print(prod_by_k)

    X = X[X.k.isin(prod_by_k)]

    # X["perc"] = X.apply(lambda row: row.value/prod_by_t[row.t] if prod_by_t[row.t] != 0  else 0, axis=1)
    # print(X)
    df = X.pivot(index='t', columns='k', values='value')
    style = ["1", "2", "3", "4", ".", "|", "_", "+", "o", "d", "D", "x", "X", "p"]
    ax = df.plot(grid=False, style=style)
    # ax = df.plot(style=".", grid=True)
    # ordener labels del legend
    handles, labels = list(zip(ax.get_legend_handles_labels()))
    for i in range(len(labels[0])):
        print(labels[0][i])
        labels[0][i] = labels[0][i][1:]
    print(handles)
    print(labels)
    handles_and_labels = list(zip(handles[0], labels[0]))
    handles_and_labels.sort(key=lambda x: int(x[1]))
    handles, labels = list(zip(*handles_and_labels))

    # ax.legend(title='Pattern used')
    ax.legend(handles, labels, title='Pattern used', ncol=3)
    ax.set_xlim(xmin=0)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    string = f"Uso de patrones"
    if cortes:
        string += f" producto {f}"
    if case:
        string += f" caso {case}"

    # plt.title(string)
    if save:
        plt.savefig(f"{model_name}/Figures/{case}/png/{out_file}.png")
        plt.savefig(f"{model_name}/Figures/{case}/pdf/{out_file}.pdf")
    else:
        plt.show()


def sim_plots(dfs, strings, f, axes, j, ymin=None, ymax=None, xlabel=None, ylabel=None, add_legend=False):

    for i, df in enumerate(dfs):
        dfs[i] = df[df.f == f]
        dfs[i][""] = strings[i]

    df = pd.concat(dfs)
    df = df.pivot(index="t", columns="", values="value")
    df = df[strings]
    # styles = ["-", "-.", "--", ":", "cd", "mo"]
    styles = ["-", "-.", "--", ":", ".", "x", "cd", "mo"]
    colors = list(mcolors.TABLEAU_COLORS.values())
    # colors = ['r', 'g', 'b', 'c']
    # styles = ["1", "2", "3", "4", ".", "|", "_", "+"]
    if j == 0:
        df.plot(ax=axes[j], grid=False, style=styles, sort_columns=False, legend=add_legend)
        if add_legend:
            handles, labels = list(zip(axes[j].get_legend_handles_labels()))
            axes[j].legend(handles[0], labels[0], loc='best', ncol=1)
    else:
        df.plot(ax=axes[j], grid=False, style=styles[-2:], sort_columns=False,  legend=add_legend)
    # df.plot(ax=ax, grid=True, style=styles)
    
    ############################### CONFIGURACION MOISES ##############################
    if j == 0:
        axes[j].legend(loc='center right', ncol=1, bbox_to_anchor=(1.45, 0.5))
    else:
        axes[j].legend(loc='center right', ncol=1, bbox_to_anchor=(1.29, 0.5))
    axes[j].set_title(f'Simulación producto final: {f}')
    ############################### CONFIGURACION MOISES ##############################
    
    
    axes[j].set_xlabel('')
    axes[j].set_xlim(xmin=0, xmax=max(df.index))
    if ymax:
        axes[j].set_ylim(ymin=0, ymax=ymax)
    if ylabel:
        axes[j].set_ylabel(ylabel)
    #plt.show()


def price_and_q_plots(multi_dfs, multi_strings, multi_ymax, multi_ylabels, f, out_file=None, save=False,
                      model_name="M2", case=0, add_legend=False):

    fig, axes = plt.subplots(2, 1, sharex=False, sharey=False)

    for i, dfs, strings, ymax, ylabel in zip(range(len(multi_dfs)),multi_dfs, multi_strings, multi_ymax, multi_ylabels):
        sim_plots(dfs, strings, f, axes, i, ymax=ymax, xlabel=None, ylabel=ylabel, add_legend=add_legend)
        
    ############################### CONFIGURACION MOISES ##############################
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.7,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
                        
    fig.set_figheight(7)
    fig.set_figwidth(11)
    ############################### CONFIGURACION MOISES ##############################


    plt.xlabel("Periods since start")
    if save:
        plt.savefig(f"{model_name}/Figures/{case}/png/{out_file}.png")
        plt.savefig(f"{model_name}/Figures/{case}/pdf/{out_file}.pdf")
    else:
        plt.show()


def all_price_q_plots(key_fs, end_t, fetch_strings, aggs, name, case=0, model_name="M2", it_case=5, dynamic=False,
                      save=False, add_legend=False):

    for f in key_fs:

        fig, axes = plt.subplots(2, 1)
        X, S, D, P, L, beta, alfa = fetch_all_dfs(model_name, case, fetch_strings, aggs, name, end_t=end_t)
        precio_maximo = fetch_precio_maximo(alfa, beta)
        cortes = cortes_por_patron()
        prod = produccion_por_dia(cortes, X, [f], end_t=end_t)
        strings_0 = ["Max estimated demand (sales)", "Production", "Estimated demand (sales)", "Inventory"]
        dfs_0 = [alfa, prod, D, S]
        out_file = f"Estático_{f}"

        if dynamic:
            strings_iterator = ["X", "S", "D", "P", "L", "D_real", "sales", "prod"]

            iter_aggs = [False for i in range(len(strings_iterator))]
            X, S, D, P, L, D_real, sales, prod = fetch_all_dfs(model_name, case, strings_iterator, iter_aggs, name,
                                                               end_t=end_t,
                                                               prefix=f"Iteration/{it_case}/")
            strings_0 = ["Max estimated demand (sales)", "Production", "Estimated demand (sales)", "Inventory", "Actual demand",
                         "Sales"]
            dfs_0 = [alfa, prod, D, S, D_real, sales]
            out_file = f"Dinámico_{it_case}_{f}"

        ymax_0 = 11000
        ylab_0 = "Boxes"

        strings_1 = ["Maximum price", "Price"]
        dfs_1 = [precio_maximo, P]
        ymax_1 = 85000
        ylab_1 = "Price (CLP)"

        # multi_dfs = [dfs_0, dfs_1]
        # multi_strings = [strings_0, strings_1]
        # multi_ymax = [ymax_0, ymax_1]
        # multi_ylabels = [ylab_0, ylab_1]

        multi_dfs = [dfs_0 + dfs_1, dfs_1]
        multi_strings = [strings_0 + strings_1, strings_1]
        multi_ymax = [ymax_0, ymax_1]
        multi_ylabels = [ylab_0, ylab_1]

        price_and_q_plots(multi_dfs, multi_strings, multi_ymax, multi_ylabels, f, out_file=out_file, save=save,
                          add_legend=add_legend)

#borra todoo el contenido de folder, así que usar con cuidado!
def delete_content(folder):

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                delete_content(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


#calcular inventario (S) a partir de W y W0 (en modelo 2)
def w_to_s(W, W0, T, F, delta):

    S = {(f, t, u): 0 for f in F for t in T for u in range(t + 1, t + delta)}
    for f in F:

        #producidos durante el paríodo de planificación
        for t in T:
            #asignación posible hasta que vence, limitado por el período max(T) + 1 (para valores residuales)
            for s in range(t + 1, min(t + delta, T[-1] + 2)):
                val = W[f, t, s].X
                #sumamos h a todos los períodos entre t y s (como son producidos en t, todos vences en t+delta-1
                for n in range(t, s):
                    S[f, n, t + delta - 1] += val

        #inventario inicial
        for s in range(1, delta):
            for u in range(s, delta):
                val = W0[f, s, u].X
                for n in range(1, s):
                    S[f, n, u] += val

    return S



#calcular demanda a partir de W, W0 y L
def alpha_beta_to_d(W, W0, L, T, F, delta):

    D = {(f, t, u): 0 for f in F for t in T for u in range(t, t + delta)}
    for f in F:
        # producidos durante el paríodo de planificación
        for t in T:
            #if (f == 'Entero' and t == 1):
            #    print(f"Merma (L) en t={t}",L[f, t])
            D[f, t, t + delta - 1] -= L[f, t] #restamos merma de la demanda
            #asignación a períodos desde que es producido o hasta que vence (y limitado por max(T))
            for s in range(t, min(t + delta, T[-1] + 1)):
                #if (f == 'Entero' and t == 1):
                #    print(f"(W) produce en t={t} para vender en s={s}",W[f, t, s].X)
                val = W[f, t, s].X
                D[f, s, t + delta - 1] += val

        # inventario inicial
        for s in range(1, delta):
            for u in range(s, delta):
                #if (f == 'Entero' and s == 1):
                #    print(f"(W0) vende en t={s} lo que vence en s={u}",W0[f, s, u].X)
                val = W0[f, s, u].X
                D[f, s, u] += val

    return D


def calc_profit(X, c_X, S, c_S, V, P, L, c_L, start_t, end_t):

    F = pd.unique(V.f)
    dfs = [X, S, V]
    len_t = end_t - start_t
    X = X[(start_t < X.t)&(X.t <= end_t)]
    S = S[(start_t < S.t)&(S.t <= end_t)]
    L = L[(start_t < L.t)&(L.t <= end_t)]

    X["cost"] = X.apply(lambda row: row.value*c_X[row.k], axis=1)
    S["cost"] = S.apply(lambda row: row.value*c_S[row.f][row.t], axis=1)
    L["cost"] = L.apply(lambda row: row.value*c_L[row.f][row.t], axis=1)
    # print(pd.unique(S.t))
    V = V.set_index(["f", "t"]).join(P.set_index(["f", "t"]), on=["f", "t"], rsuffix="2")
    ingresos = sum(V.value*V.value2)/len_t
    costos_inv = sum(S.cost)/len_t
    costo_prod = sum(X.cost)/len_t
    costo_merma = sum(L.cost)/len_t

    profit = ingresos - costos_inv - costo_prod - costo_merma
    # print(f"profit: {profit}")
    # print(f"ingresos: {ingresos}")
    # print(f"costo_prod: {costo_prod}")
    # print(f"costos_inv: {costos_inv}")
    return profit, ingresos, costos_inv, costo_prod, costo_merma



#Para correr los experimentos
#filename: nombre del input
#scaler: En cuanto se achica el problema (cajas de esa cantidad)
#experiment: para hacer experimentos variando Q, Alfa, o beta
#mults: lista de multiplicadores para ir variando la variable respectiva
#si delete=True, se borran los output de ejecuciones anteriores del experimento
def experiment_executions(model, model_name, case, filename, mip_gap, time_limit, scaler, experiment, mults, delete=False, reps=None, periods=90):

    if delete:
        if experiment in ["Res", "S0"]:
            folder = f"{model_name}/Output/Variables/{case}/{experiment}/"
            delete_content(folder)
        folder = f"{model_name}/Output/Experiments/{case}/{experiment}/"
        delete_content(folder)
    for mult in mults:
        print(f"Case {case}, experiment: {experiment}, mult: {mult}")
        if experiment == "Q":
            print(f"mult: {mult}")
            model(filename, mip_gap, time_limit, scaler, experiment=experiment, q_mult=mult, case=case, periods=periods)
        elif experiment == "Alfa":
            model(filename, mip_gap, time_limit, scaler, experiment=experiment,
                  alfa_mult=mult, case=case, periods=periods)
        elif experiment == "Beta":
            model(filename, mip_gap, time_limit, scaler, experiment=experiment,
                  beta_mult=mult, case=case, periods=periods)
        elif experiment == "Res":
            model(filename, mip_gap, time_limit, scaler,
                  experiment=experiment, res_mult=mult, case=case, periods=periods)
        elif experiment == "S0":
            model(filename, mip_gap, time_limit, scaler,
                  experiment=experiment, S0_mult=mult, case=case, periods=periods)
        elif experiment == "Periods":
            [model(filename, mip_gap, time_limit, scaler, experiment=experiment, periods=mult, case=case,
                   rep=rep) for rep in range(reps)]


#guardar output del modelo
def save_model_data(model, model_name, case, string_input, out_vars,  experiment, now, q_mult, alfa_mult, beta_mult,
                    S0_mult, mip_gap, time_limit, res_mult, rep, scaler, q, alfa, beta, periods, indicadores):

    status, opt_value = model.Status, model.objVal
    total_time = (dt.datetime.now() - now).total_seconds()
    
    results = {"status": [status], "value": [opt_value], "scaler": [scaler], "q_mult": [q_mult],
               "alfa_mult": [alfa_mult], "beta_mult": [beta_mult], "q": [q], "alfa": [alfa], "beta": [beta],
               "Res_mult": [res_mult], "S0_mult": [S0_mult], "exec_time": [total_time], "periods": [len(q.values())],
               "mip_gap": [mip_gap], "time_limit": [time_limit]}
    # print(f"indicadores: {indicadores}")
    
    results = {**results, **indicadores}
   
    # print(results)
    results = pd.DataFrame.from_dict(results)
    
    #nombres de las variables
    strings = ["X", "S", "D", "P", "L", "alpha", "beta"]
    #nombres de las columnas para cada variable
    column_names = [["k", "t"], ["f", "t", "u"], ["f", "t", "u"], ["f", "t"], ["f", "t"], ["f", "t"], ["f", "t"]]

    ##################### TESTEO PARA PROBAR MODELO ######################################
    results.to_pickle(
        f"{model_name}/Output/Results/{case}/{string_input}_results.pkl")
    [get_vars_df(model_name, case, var, columns, string_input + "_" + string) for var, columns, string in
     zip(out_vars, column_names, strings)]
    ##################### TESTEO PARA PROBAR MODELO ######################################


    #si no es un experimento, guardamos los valores de las variables de output y results del model
    if not experiment:
        results.to_pickle(f"{model_name}/Output/Results/{case}/{string_input}_results.pkl")
        [get_vars_df(model_name, case, var, columns, string_input + "_" + string) for var, columns, string in
         zip(out_vars, column_names, strings)]

    #para algunos experimentos guardamos results y variables, para otros solo results
    else:
        if experiment in ["Q", "Alfa", "Beta"]:
            results.to_pickle(
                f"{model_name}/Output/Experiments/{case}/{experiment}/{string_input}_q{q_mult}_alpha{alfa_mult}_beta{beta_mult}.pkl")
        elif experiment in ["Periods"]:
            results.to_pickle(f"{model_name}/Output/Experiments/{case}/{experiment}/{string_input}_p{periods}_r{rep}.pkl")
        elif experiment in ["Res", "S0"]:
            results.to_pickle(f"{model_name}/Output/Experiments/{case}/{experiment}/{string_input}_res{res_mult}_S{S0_mult}.pkl")
            [get_vars_df(model_name, case, var, columns, f"{experiment}/{string}/{string_input}_res{res_mult}_S{S0_mult}")
             for var, columns, string in zip(out_vars, column_names, strings)]

    return status, opt_value


if __name__ == "__main__":


######## Datos para todas las funcionalidades ############

    model_name = "M2"
    name = "Constante" #archivo de input
    case = 0
    end_t = 90
    periods_exec = 90

    all_F =  [
        "Entero",
        "Medio Pollo",
        "Medio Pollo Superior",
        "Cuarto de Pollo Superior",
        "Muslo Entero",
        "Pechuga Completa",
        "Ala Completa",
        "Media Pechuga",
        "Blanqueta",
        "Alón",
        "Punta",
        "Jamoncito",
        "Medio Muslo"
    ]
    
    F = ["Entero", "Media Pechuga"] #todos los cortes
    F_subset = F
    # F_subset = ["Entero", "Medio Pollo"] #cortes a considerar para la venta

    ps = [None, "p9", "p5", None] #patrones en los que se produce solo el producto que se quiere (ej: p9 produce 2 medios)
    strings = ["X", "S", "D", "P", "L", "beta", "alpha"] #strings asociados a las variables
    cortes = cortes_por_patron()
    aggs = [False, True, True, False, False, False, False]
    aggs_dict = {s: a for s, a in zip(strings, aggs)}
    scaler = 1
    X, S, D, P, L, beta, alfa = fetch_all_dfs(model_name, case, strings, aggs, name, end_t=end_t)
    precio_maximo = fetch_precio_maximo(alfa, beta)
    cortes = cortes_por_patron()
    prod = produccion_por_dia(cortes, X, all_F, end_t=end_t)


    # #resumen de principales indicadores
    # cases = [0, 1, 4]
    # summary_table(model_name, name, all_F, strings, aggs, cases=cases, out_suffix="static", save=True, start_t=20, end_t=35,
    #               periods_exec=periods_exec)
    # cases = [5, 6]
    # summary_table(model_name, name, all_F, strings, aggs, cases=cases, out_suffix="dynamic", save=True, start_t=20, end_t=35,
    #               periods_exec=periods_exec)

    # cases = [5, 7, 8]
    # summary_table(model_name, name, all_F, strings, aggs, cases=cases, out_suffix="dynamic_all", save=True, start_t=20, end_t=35,
    #               periods_exec=periods_exec)



    # summary_chart(model_name, name, cases, ["D"], all_F, aggs_dict, save=to_save, end_t=False, xlabel="Caso",
    #               ylabel="Venta (#)",title="Venta total por producto", out_file="Venta_total", threshold=50)
    # summary_chart(model_name, name, cases, ["D", "P"], all_F, aggs_dict, save=to_save, end_t=False, xlabel="Caso",
    #               ylabel="Ingresos ($)",title="Ingresos totales por producto", out_file="Ingreso_total", threshold=1000000)


    prod = produccion_por_dia(cortes, X, all_F)
    
    path = f"Input/{name}.xlsx"
    aux, aux2, c =  read_sheet(path, "Costo Corte", delete_row_header=True)
    T_keys, aux, h = read_sheet(path, "Costo Hold", periods_exec, num_columns=True)
    aux, aux2, c_merma = read_sheet(path, "Merma", periods_exec, num_columns=True)

    # print(f"X: {X}")
    # print(f"S: {S}")
    # print(f"D: {D}")
    # print(f"P: {P}")
    # print(f"L: {L}")
    # print(f"beta: {beta}")
    # print(f"alfa: {alfa}")


################ Plot importante! ###########################
    f = 'Entero'
    key_fs =["Entero", "Medio Pollo", "Muslo Entero", "Pechuga Completa", 'Media Pechuga'] # Que productos quiero graficar
    dynamic = False  # Para la simulación
    it_case = 5 # Politica a revisar
    to_save = False # if True, save.  Else, plot
    add_legend = True # Agregar leyenda a los graficos 

    
    all_price_q_plots(
        key_fs,
        end_t,
        strings,
        aggs,
        name,
        case=0,
        model_name="M2",
        dynamic=dynamic,
        it_case=it_case,
        save=to_save, 
        add_legend=add_legend
    )

    # strings_iterator = ["X", "S", "D", "P", "L", "D_real", "sales", "prod"]
    # iter_aggs = [False for i in range(len(strings_iterator))]
    # _, _, _, _, _, D_real, sales, _ = fetch_all_dfs(model_name, 0, strings_iterator, iter_aggs, name,
    #                                                end_t=end_t, prefix=f"Iteration/{5}/")
    # calc_profit(X, c, S, h, sales, P, L, c_merma, 0, end_t)
    
    # for f2 in key_fs:
    #     plot_f_t(model_name, [f2], [prod, D, S, D_real, sales], [P, precio_maximo], ["Produccion", "Demanda estimada", "Inventario", "Demanda real", "Ventas"],
    #              ["Precio por producto", "Precio máximo"], f"{name}-iterator", case, save=to_save, title=f"Evolución variables escenario dinámico producto {f2}",
    #                  file_out=f"vars_dinámico_{f2}")

################ Plot importante! ###########################



##### Plots de patrones y evolución de variables para casos simultáneos
    # plot_patrones(X, out_file="Patrones", case=case, save=to_save, xlabel="Periods since start", ylabel="Boxes") #patrones utilizados para todos los productos
    # plot_patrones(X, out_file=f"Patrones_{f}", cortes=cortes, f=f, case=case, save=to_save, xlabel="Periods since start", ylabel="Chicken boxes (#)") #patrones utilizados un subproducto

    # cases = [0]
    # plot_dif_cases(model_name, cases, ["P", "D"], aggs, name, f, end_t=end_t) #evolución de variables para varios casos
    # prod = produccion_por_dia(cortes, X, all_F)
    # plot_prod("Price (CLP)", P, case, out_file="Precio_por_dia", save=to_save, add_legend=add_legend) #Precio todos los productos
    # plot_prod("Sales (# of boxes)", D, case, out_file="Venta_por_dia", save=to_save, add_legend=add_legend) #venta todos los productos
    # plot_prod("Inventory (# of boxes)", S, case, out_file="Inv_por_dia", save=to_save, add_legend=add_legend) #inventario todos los productos
    # plot_prod("Production (# of boxes)", prod, case, out_file="Prod_por_dia", save=to_save, add_legend=add_legend) #producción todos los productos
    # plot_prod("Waste (# of boxes)", L, case, out_file="Merma_por_dia", save=to_save, add_legend=add_legend) #merma todos los productos

############ Gráfico de varias variables en simultáneo para un solo caso y varios productos
    # precio_maximo = fetch_precio_maximo(alfa, beta)
    # prod = produccion_por_dia(cortes, X, F, end_t=end_t)
    # dfs = [prod, D, S, alfa]
    # f = "Entero"
    # fig, axes= plt.subplots(2, 1)
    # strings = ["Production", "Estimated demand", "Inventory", "Max estimated demand"]
    # sim_plots(dfs, strings, f, axes[0], out_file=None, save=False, xlabel=None, ylabel=None)

    # print(f"prod: {prod}")
    # for f2 in key_fs:
    #     plot_f_t(model_name, [f2], [prod, D, S, alfa, L], [P, precio_maximo], ["Produccion", "Venta", "Inventario", "Demanda máxima", "Merma"],
    #              ["Precio por producto", "Precio máximo"], name, case, save=to_save, title=f"Evolución variables escenario estático producto {f2}",
    #              file_out=f"vars_estatico_{f2}")

    # # plottear/guardar q
    # aux, aux2, q = read_sheet(f"Input/{name}.xlsx", "Insumos", num_columns=True, delete_row_header=True)
    # plot_q(model_name, "Input/Varia_insumo.xlsx", scaler=scaler, save=to_save)


############# Funcion para guardar todos los plots #############

    # names = ["Constante", "Varia_insumo", "Alfa_sube_baja", "Alfa_baja_sube", "Beta_baja_sube", "Beta_sube_baja",
    #          "Solo_medio", "Solo_cuarto"]
    # for name in names:
    #     X, S, D, P, L, beta, alfa = fetch_all_dfs(model_name, strings, aggs, name)
    #     aux, aux2, q = read_sheet(f"Input/{name}.xlsx", "Insumos", num_columns=True, delete_row_header=True)
    #     precio_maximo = fetch_precio_maximo(alfa, beta)
    #     prod = produccion_por_dia(cortes, X, F)
    #     plot_f_t(model_name, F, F_subset, [prod, D, S, alfa, L], [P, precio_maximo], ["Produccion", "Venta", "Inventario", "Alfa", "Merma"],
    #              ["Precio por producto", "Precio máximo"], name, save=False)

#############   Experimentos Q, Alfa, Beta o Periods   #############
    # case=0
    # Leer experimento (save = False para plottear, True para guardar)
    # base_indexes = {"Q": 40000, "Beta": 1, "Alfa": 0, "Periods": -1}
    # xlabels = {"Q": "Daily input boxes", "Beta": "Price sensitivity multiplier", "Alfa": "Maximum price multiplier",
    #            "Periods": "Planning horizon (periods)"}
    # experiment = "Q"
    # experiment_plotter(model_name, case, experiment, name, ps, xlabel=xlabels[experiment],
    #                    base_index=base_indexes[experiment], save=to_save, out_file=experiment)

# #############   Experimentos S0 o Res   #############
#     # F: ["Entero", "Medio Pollo", "Medio Pollo Superior", "Cuarto de Pollo Superior", "Muslo Entero", "Pechuga Completa",
#     #     "Ala Completa", "Media Pechuga", "Blanqueta", "Alón", "Punta", "Jamoncito", "Medio Muslo"]
#     case=0
#     experiment = "S0"
#     start = {"Res": -21, "S0": 0}
#     end = {"Res": -1, "S0": 30}
#     f = "Entero"
#     # f = "Medio Pollo"
#     plot_s_res_experiment(model_name, case, ["P"], [True], experiment, f, start[experiment], end[experiment], save=to_save, out_file="S0")


############ max profit #################

    # profit_df(model_name, ps)




