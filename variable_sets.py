"""
Set de funciones que generan los conjuntos del problema.
"""

#Inventario desde 1 a |T| (con T = 1...last_t) de producto final f que vence en t+1..t+delta-1
def S_sets(F, T, delta):

    return [(f, t, u) for f in F for t in T for u in range(t + 1, t + delta)]


#Cantidad de la demanda de producto final tipo f en t que se satisface con productos que vencen en u = t..t+delta-1
def D_sets(F, T, delta):

    return [(f, t, u) for f in F for t in T for u in range(t, t + delta)]


#Cantidad de cortes con patrón k en t (solo períodos en los que se produce/vende)
def X_sets(K, T):

    return [(k, t) for k in K for t in T]


#Precio para el producto f en t (solo períodos en los que se produce/vende)
def P_sets(F, T):

    return [(f, t) for f in F for t in T]


#Producto f producido en t guardado hasta S
def W_sets(F, T, delta):

    return [(f, t, s) for f in F for t in T for s in range(t, min(t + delta, T[-1] + 2))]


#Producto inicial f guardado hasta s que vence en u
def W0_sets(F, delta):

    return [(f, s, u) for f in F for s in range(1, delta) for u in range(s, delta)]

def W0_sets_v2(F, T, delta):

    return [(f, s, u) for f in F for s in T for u in range(s, s+delta-1)]

#Inventarios en t que vence en u de producto f
def I_sets(F, T, delta):
    return [(t,u,f) for f in F for t in T for u in range(1, delta+1)]

