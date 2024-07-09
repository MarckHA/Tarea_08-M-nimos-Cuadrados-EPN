# -*- coding: utf-8 -*-
"""
Python 3
08 / 07 / 2024
@author: Mark_H

"""

import numpy as np
        
#----------------Ajuste de mínimos cuadrados---------------

def minimosCuadrados(n, grado, xi, yi):
    A = np.zeros((grado+1,grado+1))
    b = np.zeros((grado+1,1))
    for i in range (0,grado+1):
        for j in range(0,grado+1):
            k=0
            while (k<n):
                A[i,j] += xi[k]**((i)+(j))
                k += 1
        for l in range(n):
            b[i] += (xi[l]**(i))*yi[l]
            
    print_matrix(A, "Matriz A")
    print_matrix(b, "Vector b")
    return A,b

def hallarCoef(a,b):
    A=np.linalg.inv(a)
    x = np.dot(A,b)
    print_matrix(x, "Coeficientes del polinomio")
    return x

#--------------------------Graficar-------------------------

import matplotlib.pyplot as plt
import sympy as sym

def graficar(xi,yi,c,colorcurva,rango_x,rango_y,x_pol,y_pol,lim_inf):
    x = sym.Symbol('x')
    f_x = sum(round(coef[0],4) *x**i for i, coef in enumerate(c))
    # Generar valores de x
    x_val = np.linspace(min(xi)-lim_inf, max(xi)+lim_inf, 100)
    f = sym.lambdify(x, f_x, modules=['numpy'])
    # Calcular los  valores de y 
    y = f(x_val)
    
    xi = np.array(xi)
    yi = np.array(yi)
    # Calcular los residuos (errores)
    residuos = yi - f(xi)
    imprimirErrores(residuos,xi)
    # Calcular el error cuadrático medio (MSE)
    mse = np.mean(residuos**2)
    print("El error cuadrático medio para este ajuste es de:", round(mse,6))
    imprimirPolinomio(f_x)
    # Graficar
    plt.figure(figsize=(10, 8))
    # Graficar los puntos originales con barras de error
    plt.errorbar(xi, yi, yerr=abs(residuos), fmt=' ', color='red', label='Datos originales con error')
    # Graficar la ecuación
    plt.plot(x_val, y,color = colorcurva, label = 'Polinomio aproximado')
    #Graficar los datos originales
    plt.scatter(xi, yi, color='black', label='Datos originales', s = 20)
    #Graficar los nombres de los ejes
    plt.xlabel('x')
    plt.ylabel('y')
    #Limites para x e y
    ax = plt.gca()
    ax.set_ylim(rango_y)
    ax.set_xlim(rango_x)
    #Agregar cuadrícula
    plt.grid(True)
    #Agregar la leyenda
    plt.legend()
    #Agregar la ecuación de la curva a la gráfica
    plt.text(x_pol, y_pol, f'$P(x) = {sym.latex(f_x)}$', fontsize=12, color=colorcurva, verticalalignment='bottom')
    # Marca los ejes coordenados
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.show()

#------------------------------Graficar no lineales-------------------------------

def graficarNoLineales(xi,yi,f_x,colorcurva,rango_x,rango_y,x_pol,y_pol,lim_inf):   
    x = sym.Symbol('x')
    # Generar valores de x
    x_val = np.linspace(min(xi)-lim_inf, max(xi)+1, 100)
    f = sym.lambdify(x, f_x, modules=['numpy'])
    # Calcular los  valores de y 
    y = f(x_val)
    
    xi = np.array(xi)
    yi = np.array(yi)
    # Calcular los residuos (errores)
    residuos = yi - f(xi)
    imprimirErrores(residuos,xi)
    # Calcular el error cuadrático medio (MSE)
    mse = np.mean(residuos**2)
    print("El error cuadrático medio para este ajuste es de:", round(mse,2))
    imprimirPolinomio(f_x)
    # Graficar
    plt.figure(figsize=(10, 8))
    # Graficar los puntos originales con barras de error
    plt.errorbar(xi, yi, yerr=abs(residuos), fmt=' ', color='red', label='Datos originales con error')
    # Graficar la ecuación
    plt.plot(x_val, y,color = colorcurva, label = 'Polinomio aproximado')
    #Graficar los datos originales
    plt.scatter(xi, yi, color='black', label='Datos originales', s = 20)
    #Graficar los nombres de los ejes
    plt.xlabel('x')
    plt.ylabel('y')
    #Limites para x e y
    ax = plt.gca()
    ax.set_ylim(rango_y)
    ax.set_xlim(rango_x)
    #Agregar cuadrícula
    plt.grid(True)
    #Agregar la leyenda
    plt.legend()
    #Agregar la ecuación de la curva a la gráfica
    plt.text(x_pol, y_pol, f'$P(x) = {sym.latex(f_x)}$', fontsize=12, color=colorcurva, verticalalignment='bottom')
    # Marca los ejes coordenados
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.show()

#--------------------------Regreso a la expresión original-----------------------
from IPython.display import display, Math

def expOriginal(c,exp):
    print("Con los coeficientes asociados al polinomio linealizado hallamos los coeficientes de nuestra\n expresión:\n")  
    #Hallar los coeficientes adecuados
    b_exp = np.e**(c[0,0])
    a_exp = c[1,0]
    print("a =",a_exp," y b =", b_exp,"\n")
        
    x = sym.Symbol('x')
    if exp:
        #Generar la ecuación en la forma be^{ax}
        f_x = round(b_exp,4)*sym.exp(round(a_exp,4)*x)
    else:
        #Generar la ecuación en la forma bx^{a}
        f_x = round(b_exp,4)*x**(round(a_exp,4))
    return f_x

#-----------------------Impresión--------------------------

def print_matrix(matrix, name):
    print(name + ":")
    for row in matrix:
        print(" [ ", "   ".join(f"{elem:12.4f}" for elem in row), "]")
        
#---------------------------Imprimir errores puntuales----------------------------
def imprimirErrores(residuos,xi):
    i=1
    print(" ")
    for res in residuos:
        print("El error absoluto de f(x_"+str(i)+") al punto x_"+str(i)+" es de",round(abs(res),6))
        i += 1
        
#------------------------------Impresión polinomios-----------------------------    
def imprimirPolinomio(f_x):    
    # Generar la representación LaTeX de la expresión
    latex_expr = sym.latex(f_x)
    print("Por tanto, el polinomio aproximado en la forma solicitada es:\n")
    # Mostrar la expresión LaTeX
    display(Math(latex_expr))