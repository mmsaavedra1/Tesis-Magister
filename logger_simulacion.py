def logger_inventario_inicial(archi1, f, t, r, delta, S_inicial, S0, prod):
    archi1.write(f"Inicio en {t}:\n")
    archi1.write(f"(Inv inicial) Simulacion: {S_inicial[f, t, r]}\n")
    archi1.write("\n****** Detalle de inventario: *******\n")
    for u in range(1, delta+1):
        archi1.write(f"Inventario en {t} que vence en {t+u-1}: {S0[f][u]}\n")
    archi1.write("****** Fin Detalle de inventario: *******\n\n")
    archi1.write("\n****** Detalle de Produccion: *******\n")
    archi1.write(f"(Produccion) Simulacion: {prod[f,t,r]}\n")
    archi1.write("****** Fin Detalle de produccion: *******\n\n")


def logger_inventario_actualizado(archi1, delta, f, t, S0):
    archi1.write("\n****** Detalle de inventario actualizado: *******\n")
    for u in range(1, delta+1):
        archi1.write(f"Inventario en {t} para que vence en {t+u-1}: {S0[f][u]}\n")
    archi1.write("\n****** Fin de inventario actualizado: *******\n\n")


def logger_ventas(archi1, f, t, r, D, error_dda, D_real, sales, L, P, S0):
    archi1.write(f"\Ventas en {t}:\n")
    archi1.write(f"(Demanda) [alfa-beta*p]) Opti: {D[f, t, r]}\n")
    archi1.write(f"(Error) Simulacion: {error_dda[str(r)][str(t)][f]}\n")
    archi1.write(f"(Factor Dcto) Simulacion: {-0.4 + error_dda[str(r)][str(t)][f]*0.8}\n")
    archi1.write(f"(Demanda) Simulacion: {D_real[(f, t, r)]}\n")
    archi1.write(f"(Venta) Simulacion: {sales[(f, t, r)]}\n")
    archi1.write(f"\nFinal en {t}:\n")
    archi1.write(f"(Merma) Simulacion: {L[f, t, r]}\n")
    archi1.write(f"(Precio de venta) Simulacion: {P[f, t, r]}\n")
    archi1.write(f"(Inv final) Simulacion: {sum(list(S0[f].values()))}\n")
    archi1.write("***************************\n\n")