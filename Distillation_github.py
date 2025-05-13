# -*- coding: utf-8 -*-
"""
Created in 2025

@author: Vicetrion

All uses allowed.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.optimize import fsolve
from typing import Callable

#McCabe-Thiele
###############################################################################
def antoine_temperature(A, B, C, P):
    """Calculate temperature from Antoine equation given A, B, C coefficients and pressure."""
    return (B / (A - np.log10(P))) - C

def equilibrium_data(Aa, Ba, Ca, Ab, Bb, Cb, P, points):
    """Generate equilibrium data (xA vs yA) for a binary mixture."""
    # Calculate boiling temperatures for pure A and B
    temp_A = antoine_temperature(Aa, Ba, Ca, P)
    temp_B = antoine_temperature(Ab, Bb, Cb, P)
    
    # Generate temperature range
    temps = np.linspace(temp_A, temp_B, points)
 
    # Calculate vapor pressures for each component
    pA = 10**(Aa - Ba / (temps + Ca))
    pB = 10**(Ab - Bb / (temps + Cb))
    
    # Calculate equilibrium compositions
    xA = (P - pB) / (pA - pB)
    yA = (pA * xA) / P
    
    return xA, yA, temps

def plot_equilibrium(xA, yA,points):
    """Plot equilibrium diagram with data points and fitted polynomial."""
    plt.figure(figsize=(8,8))
    plt.plot(xA, yA, 'r')
    
    # Reference line y=x
    plt.plot(xA, xA, 'k--',lw = 1)
    
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))
    
    plt.xlabel('xA (liquid mole fraction)')
    plt.ylabel('yA (vapor mole fraction)')
    plt.title('Equilibrium Curve (xA vs yA)')
    plt.legend(['Equilibrium Data', "y=x reference"], fontsize = 8)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.show()
    
def plot_polyfit(poly_coeffs,points):
    """Plot fitted polynomial."""
    plt.figure(figsize=(8,8))
    #Plot polynomial fit
    x_fit = np.linspace(0, 1, points)
    y_fit = np.polyval(poly_coeffs, x_fit)
    plt.plot(x_fit, y_fit, 'purple')
    
    # Reference line y=x
    plt.plot(x_fit, x_fit, "k--", lw = 1) 
    
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))
    
    plt.xlabel('xA (liquid mole fraction)')
    plt.ylabel('yA (vapor mole fraction)')
    plt.title('Equilibrium Curve (xA vs yA)')
    plt.legend([f'Polynomial fit (degree {len(poly_coeffs)-1})', "y=x reference"], fontsize = 8)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.show()

def plot_txy(poly_TxA, poly_TyA,titre,W,F,D,points,P):
    """Plot Txy diagram."""
    plt.figure(figsize=(8,8))
    #on évalue avec les polynomes
    x_fit = np.linspace(0, 1, points)
    x_ = np.polyval(poly_TxA, x_fit)
    
    y_ = np.polyval(poly_TyA, x_fit)
    
    plt.plot(x_fit, x_-273.15, 'b', label='T vs xA (liquid)')
    plt.plot(x_fit, y_-273.15, 'r', label='T vs yA (vapor)')
    
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(x_[0]-273.150, x_[points-1]-273.150, 30))############### le 30 a modifier si besoin
    
    plt.plot([W, W], [y_[points-1]-273.150, np.polyval(poly_TyA, W)-273.150], "g")
    plt.plot([0, W], [np.polyval(poly_TyA, W)-273.150, np.polyval(poly_TyA, W)-273.150], "g")
    
    plt.plot([W, W], [x_[points-1]-273.150, np.polyval(poly_TxA, W)-273.150], "g")
    plt.plot([0, W], [np.polyval(poly_TxA, W)-273.150, np.polyval(poly_TxA, W)-273.150], "g")
    
    
    plt.plot([F, F], [y_[points-1]-273.150, np.polyval(poly_TyA, F)-273.150], "g")
    plt.plot([0, F], [np.polyval(poly_TyA, F)-273.150, np.polyval(poly_TyA, F)-273.150], "g")
    
    plt.plot([F, F], [x_[points-1]-273.150, np.polyval(poly_TxA, F)-273.150], "g")
    plt.plot([0, F], [np.polyval(poly_TxA, F)-273.150, np.polyval(poly_TxA, F)-273.150], "g")
    
    
    plt.plot([D, D], [y_[points-1]-273.150, np.polyval(poly_TyA, D)-273.150], "g")
    plt.plot([0, D], [np.polyval(poly_TyA, D)-273.150, np.polyval(poly_TyA, D)-273.150], "g")
    
    plt.plot([D, D], [x_[points-1]-273.150, np.polyval(poly_TxA, D)-273.150], "g")
    plt.plot([0, D], [np.polyval(poly_TxA, D)-273.150, np.polyval(poly_TxA, D)-273.150], "g")
    
    affiche = 0.88*(x_[0]-x_[points-1])+x_[points-1]-273.15
    ecart = 1
    
    plt.text(0.7, affiche, f"Pression = {P:.{decimal}f} bar", fontsize=8, color='g')
    affiche = affiche-ecart*2
    
    plt.text(0.7, affiche, f"Ty_W = {(np.polyval(poly_TyA, W)-273.150):.{decimal}f} °C", fontsize=8, color='r')
    affiche = affiche-ecart
    plt.text(0.7, affiche, f"Tx_W = {(np.polyval(poly_TxA, W)-273.150):.{decimal}f} °C", fontsize=8, color='b')
    affiche = affiche-ecart*2
    
    plt.text(0.7, affiche, f"Ty_F = {(np.polyval(poly_TyA, F)-273.150):.{decimal}f} °C", fontsize=8, color='r')
    affiche = affiche-ecart
    plt.text(0.7, affiche, f"Tx_F = {(np.polyval(poly_TxA, F)-273.150):.{decimal}f} °C", fontsize=8, color='b')
    affiche = affiche-ecart*2
    
    plt.text(0.7, affiche, f"Ty_D = {(np.polyval(poly_TyA, D)-273.150):.{decimal}f} °C", fontsize=8, color='r')
    affiche = affiche-ecart
    plt.text(0.7, affiche, f"Tx_D = {(np.polyval(poly_TxA, D)-273.150):.{decimal}f} °C", fontsize=8, color='b')
    affiche = affiche-ecart
    
    plt.xlabel('Benzène mole fraction')
    plt.ylabel('Temperature [°C]')
    plt.title(titre)
    plt.legend(["x","y"], fontsize=8)
    plt.xlim(0, 1)
    plt.ylim(x_[points-1]-273.15, x_[0]-273.15)
    plt.grid()
    plt.show()

class distillColumn:
    def __init__(self, feed: float, xb: float, xf: float, xd: float, q: float,EfficacitePlateaux:float,
                 fxy : Callable, r: float, name: str = ""):
        """__init__ _summary_

        Args:
            feed (float): flowrate of the feed stream
            xb (float): molar fraction of the liquid at the buttoms
            xf (float): molar fraction of the liquid in the feed stream
            xd (float): molar fraction of the liquid at the distilate
            q (float): thermodynamical state of the feed
            EfficacitePlateaux : efficacité des plateaux
            fxy : xy equilibrium
            r (float): reflux ratio
            name (str, optional): name of the object/tower. Defaults to "".
        """
        self.system_name = name
        #
        self.feed = feed
        self.q = q
        self.f = fxy
        self.x_B, self.x_F, self.x_D = xb, xf, xd
        self.R = r
        #
        self.efficacite = EfficacitePlateaux
        #
        self.D = (xf - xb) / (xd - xb) * feed
        self.B = feed - self.D
        self.x_mid = q!=1 and (xd/(r+1)+xf/(q-1))/((q/(q-1))-(r/(r+1))) or xf
        self.y_mid = self.upper_line(self.x_mid)
        #
        self.L_upper = self.D * self.R
        self.V_upper = (1 + self.R) * self.D
        self._a = (self.y_mid - self.x_B) / (self.x_mid - self.x_B)
        self._b = (1 - self._a) * self.x_B
        self.L_lower = self.B / (self._a - 1)
        self.V_lower = self.L_lower - self.B
        #
        self.R_prime_W = self.V_lower/self.B
        #
        self.n_trays = 0    # initializing the tray numbers
        self.Rmin = 0       # initializing the reflux ratio
        # almost private methods
        self._rmin()        # compute the minimum reflux ratio
        self._azeocheck()   # check if there are any azeotropes
        
    def _azeocheck(self):
        """_azeocheck check if there are any azeotropes in the solution.
        distillation process stops at the azeotrope point, if exists.
        """
        self.x_azeo = -1    # initial value
        fun = lambda x: self.f(x) - x
        x = fsolve(fun, 0.5)[0] # solve: f(x) = x where f(x) is the equilibrim
        if (0 < x < 1) and (x > 1e-2 or x < 1-1e-2):
            self.x_azeo = x
        return None
    
    def upper_line(self, x: float):
        """upper_line the function for operating line at the upper section of
        the column.
        R/(R+1)*x + x_d/(R+1)
        latex: y = \frac{R}{R+1}x + \frac{x_{d}}{R+1}

        Args:
            x (float): input parameter, liquid mole fraction

        Returns:
            _type_: vapour mole fraction (y)
        """
        return self.R / (self.R+1) * x + self.x_D / (self.R+1)
    
    def lower_line(self, x: float):
        """lower_line the function for operating line at the lower section of
        the column.

        Args:
            x (float): input parameter, liquid mole fraction

        Returns:
            _type_: vapour mole fraction (y)
        """
        return self._a * x + self._b
    
    def plot(self,points, figure):
        
        x_lower = np.linspace(self.x_B, self.x_mid, points)
        x_upper = np.linspace(self.x_mid, self.x_D, points)
        x = np.linspace(0, 1, points)
        _ = plt.figure(figure, figsize = (8, 8))
        # plot title
        plt.title(self.system_name)
        # plotting
        plt.plot(x_lower, self.lower_line(x_lower), "r")    # stripping section
        plt.plot(x_upper, self.upper_line(x_upper), "m")    # rectifying section
        plt.plot([self.x_F, self.x_mid],[self.x_F, self.upper_line(self.x_mid)],"green")                                       # q-line
       
        plt.legend(["Stripping", "Rectifying", "q-line"], fontsize=8)
        
        plt.plot(x, self.f(x), "k", lw = 1)                 # equilibrium
        plt.plot(x, x, "k--", lw = 1)                       # y = x
        plt.plot([self.x_B, self.x_B], [0, self.x_B], "b--")
        plt.plot([self.x_D, self.x_D], [0, self.x_D], "b--")
        plt.plot([self.x_F, self.x_F], [0, self.x_F], "b--")
        
        for i in range(len(self.x_vals)-1):
            plt.plot([self.x_vals[i], self.x_vals[i]], [self.y_vals[i], self.y_vals[i+1]], 'k', lw=0.5)
            plt.annotate(f"{len(self.x_vals)-2-i}", ((self.x_vals[i]+ self.x_vals[i+1])/2, self.y_vals[i+1]+0.01))
            plt.plot([self.x_vals[i], self.x_vals[i+1]], [self.y_vals[i+1], self.y_vals[i+1]], 'k', lw=0.5)
        
        plt.plot([self.x_B, self.x_B],[self.x_B,self.y_vals[i+1]],'k',lw=0.5)
       
        #affichage de la légende
        affiche = 0.88
        ecart = 0.025
        
        plt.text(0.01, affiche, f"Rmin = {self.Rmin:.{decimal}f}", fontsize=8, color='black')
        affiche = affiche-ecart
        plt.text(0.01, affiche, f"R = {self.R:.{decimal}f}", fontsize=8, color='black')
        affiche = affiche-ecart
        plt.text(0.01, affiche, f"Fensk = {self.fensk(points):.{decimal}f}", fontsize=8, color='black')
        
        affiche = affiche-ecart*2   
        if self.q!=1:
            plt.text(0.01, affiche, f"Eq. F = {(self.q / (self.q - 1)):.{decimal}f} x - {(self.x_F / (self.q - 1)):.{decimal}f}", fontsize=8, color='black')
        affiche = affiche-ecart
        plt.text(0.01, affiche, f"Eq. bas = {self._a:.{decimal}f} x + {self._b:.{decimal}f}", fontsize=8, color='black')
        affiche = affiche-ecart
        plt.text(0.01, affiche, f"Eq. haut = {self.R / (self.R+1):.{decimal}f} x + {self.x_D / (self.R+1):.{decimal}f}", fontsize=8, color='black')
        
        affiche = affiche-ecart*2
        Plateaux = len(self.x_vals)-2 + (self.x_B - self.x_vals[i])/(self.x_vals[i+1]- self.x_vals[i])
        plt.text(0.01, affiche, f"Plateaux McCabe-Thiele théoriques = {Plateaux:.{decimal}f}, soit {int(Plateaux)} + rebouilleur", fontsize=8, color='black')
        
        affiche = affiche-ecart
        plt.text(0.01, affiche, f"Plateaux McCabe-Thiele réels = {Plateaux/self.efficacite:.{decimal}f}, soit {int(Plateaux/self.efficacite)} + rebouilleur", fontsize=8, color='black')
        
        affiche = affiche-ecart*2
        plt.text(0.01, affiche, f"Alimentation (kmol/h) = {self.feed:.{decimal}f}", fontsize=8, color='black')
        
        affiche = affiche-ecart
        plt.text(0.01, affiche, f"Residut (kmol/h) = {self.B:.{decimal}f}", fontsize=8, color='black')
        
        affiche = affiche-ecart
        plt.text(0.01, affiche, f"Distillat (kmol/h) = {self.D:.{decimal}f}", fontsize=8, color='black')
       
        affiche = affiche-ecart*2
        plt.text(0.01, affiche, f"V_haut (kmol/h) = {self.V_upper:.{decimal}f}", fontsize=8, color='black')
        
        affiche = affiche-ecart
        plt.text(0.01, affiche, f"L_haut (kmol/h) = {self.L_upper:.{decimal}f}", fontsize=8, color='black')
        
        affiche = affiche-ecart
        plt.text(0.01, affiche, f"V_bas (kmol/h) = {self.V_lower:.{decimal}f}", fontsize=8, color='black')
        
        affiche = affiche-ecart
        plt.text(0.01, affiche, f"L_bas (kmol/h) = {self.L_lower:.{decimal}f}", fontsize=8, color='black')
       
        
        
        # axis ticks and labels
        plt.xticks(np.linspace(0, 1, 11))
        plt.yticks(np.linspace(0, 1, 11))
        plt.xlabel("Benzène liquid mole-fraction")
        plt.ylabel("Benzène vapour mole-fraction")
        # range of the x, y parameters; between 0 and 1
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid();
        plt.show()
    
    def q_line(self,x):
        if self.q==1:
            qmod = 0.9999999999999999
        else:
            qmod = self.q
        return (qmod / (qmod - 1)) * x - (self.x_F / (qmod - 1))

    
    def find_intersection(self):
        return fsolve(lambda x: self.upper_line(x) - self.q_line(x), self.x_F)[0]
    
    def run(self,points,MAX_etages,poly_coeffs):
                
        x_intersect = self.find_intersection()
        y_intersect = self.upper_line(x_intersect)

        # Stripping line slope
        strip_slope = (y_intersect - self.x_B) / (x_intersect - self.x_B)
        strip_intercept = self.x_B * (1 - strip_slope)

        x_esc = self.x_D
        stages = 0

        x_vals = [x_esc]
        y_vals = [self.x_D]

        while x_esc > self.x_B and stages < MAX_etages:
            if x_esc > x_intersect:
                y_esc = self.upper_line(x_esc)
            else:
                y_esc = strip_slope * x_esc + strip_intercept
                
            x_next = fsolve(lambda x_: np.polyval(poly_coeffs, x_) - y_esc, x_esc)[0]

            x_vals.append(x_next)
            y_vals.append(y_esc)

            stages += 1
            x_esc = x_next

        self.x_vals  = x_vals
        self.y_vals =y_vals
            
        self.n_trays = stages
        # control if there are infinite number of trays calculated which means
        # an azeotrope exists.
        if self.n_trays < MAX_etages:
            return 0
        else:
            return -1
    
    def fensk(self, points):
        """fensk Fensk equation: an estimate of the minimum required trays for
        a given system; A rule of thumb.

        Args:
            N (int, optional): number of points in the calculation. 

        Returns:
            _type_: minimum number of the trays.
        """
        alphaFun = lambda x: (self.f(x)/x)/((1-self.f(x))/(1-x))
        x = np.linspace(1e-6, 1, points, endpoint = False)
        y = alphaFun(x)
        a = trapezoid(y, x)
        self.average_alpha = a  # average volatility of the solution
        n = np.log(self.x_D/(1 - self.x_D) * (1 - self.x_B)/self.x_B) / np.log(a)
        return n
    
    def _rmin(self):
        """_rmin compute the minimum reflux ratio
        """
        if self.q == 1:
            x = self.x_F
        else:
            self.eq = lambda x: self.f(x)-self.q/(self.q-1)*x+self.x_F/(self.q-1)
            x = fsolve(self.eq, self.x_F)[0]
        y = self.f(x)
        k = (y - self.x_D) / (x - self.x_D)
        self.Rmin = k / (1 - k)
    
        return None
    
    def L_etages(self,points):
        _ = plt.figure("McCabe-Thiele xa", figsize = (8, 8))
        # plot title
        plt.title("Titre liquide par étage (théorique)")
        
        plt.plot(self.x_vals[::-1], np.linspace(0, len(self.x_vals)-1, len(self.x_vals)),"o-")
        plt.plot(np.ones([len(self.x_vals)])-self.x_vals[::-1], np.linspace(0, len(self.x_vals)-1, len(self.x_vals)),"o-")
       
        plt.legend(["xA McCabe-Thiele","xB McCabe-Thiele"], fontsize=8)
        
        # axis ticks and labels
        plt.xticks(np.linspace(0, 1, 11))
       
        plt.xlabel("Benzène (A), Toluène (B) liquid mole-fraction")
        plt.ylabel("Étages")
        # range of the x, y parameters; between 0 and 1
        
        plt.grid();
        plt.show()
    
    def V_etages(self,points):
        _ = plt.figure("McCabe-Thiele ya", figsize = (8, 8))
        # plot title
        plt.title("Titre vapeur par étage (théorique)")
        
        plt.plot(self.y_vals[::-1], np.linspace(0, len(self.y_vals)-1, len(self.y_vals)),"o-")
        plt.plot(np.ones([len(self.y_vals)])-self.y_vals[::-1], np.linspace(0, len(self.y_vals)-1, len(self.y_vals)),"o-")
       
        plt.legend(["yA McCabe-Thiele","yB McCabe-Thiele"], fontsize=8)
        
        # axis ticks and labels
        plt.xticks(np.linspace(0, 1, 11))
       
        plt.xlabel("Benzène (A), Toluène (B) vapor mole-fraction")
        plt.ylabel("Étages")
        # range of the x, y parameters; between 0 and 1
        
        plt.grid();
        plt.show()
###############################################################################

class Condenseur:
    def hL_D(self):
        return self.col.x_D * self.h_l_D_A + (1 - self.col.x_D) * self.h_l_D_B

    def HV_D(self):
        return self.col.x_D * self.H_v_D_A + (1 - self.col.x_D) * self.H_v_D_B
    
    def run_cond(self): 
        return -(self.col.R+1)*self.col.D*(self.HV_D-self.hl_D)*1000/3600 #passage en J/s = W
    
    def T_sortie_caloporteur(self):  #température de sortie du fluide caloporteur °C
        return abs(self.Qc_cond)/(self.debit_D_caloporteur*self.Cp_D_caloporteur)+self.TE_D_caloporteur
    
    def __init__(self, col : distillColumn, hl_D_A : float,hl_D_B : float,Hv_D_A:float,Hv_D_B:float,Cp_D_cal:float,debit_cal_D:float,TE_D_cal:float):
        self.col = col
        #
        self.h_l_D_A = hl_D_A               #kJ/kmol
        self.h_l_D_B = hl_D_B
        #
        self.H_v_D_A = Hv_D_A
        self.H_v_D_B = Hv_D_B
        #
        self.Cp_D_caloporteur = Cp_D_cal*1000           #de kJ/kmol/K en J/kmol/K
        self.debit_D_caloporteur = debit_cal_D/3600     #de kmol/h en kmol/s
        
        self.TE_D_caloporteur = TE_D_cal                #°C
        #
        self.HV_D = self.HV_D()
        self.hl_D = self.hL_D()    
        #
        self.Qc_cond = self.run_cond()
        self.Tf_caloporteur = self.T_sortie_caloporteur()
        
    
    def Affiche(self):
        print(f"Qc = {self.Qc_cond/1000000:.{decimal}f} MW")    #!!! on fait un passage en MW pour l'affichage
        print(f"TF_caloporteur = {self.Tf_caloporteur:.{decimal}f} °C")    
         
    
    
class Bouilleur:
    def hL_W(self):
        return self.col.x_B * self.h_l_B_A + (1 - self.col.x_B) * self.h_l_B_B

    def HV_W(self):
        return self.col.x_B * self.H_v_B_A + (1 - self.col.x_B) * self.H_v_B_B
    
    def run_bouil(self): 
        return self.col.V_lower*(self.HV_B-self.hl_B)*1000/3600   #passage en J/s = W
    
    def T_sortie_caloporteur(self):  #température de sortie du fluide caloporteur
        return -abs(self.Qc_bouil)/(self.debit_B_caloporteur*self.Cp_B_caloporteur)+self.TE_B_caloporteur
    
    def __init__(self, col : distillColumn, hl_W_A : float,hl_W_B : float,Hv_W_A:float,Hv_W_B:float,Cp_W_cal:float,debit_cal_W:float,TE_W_cal:float):
        self.col = col
        #
        self.h_l_B_A = hl_W_A
        self.h_l_B_B = hl_W_B
        #
        self.H_v_B_A = Hv_W_A
        self.H_v_B_B = Hv_W_B
        #
        self.Cp_B_caloporteur = Cp_W_cal
        self.debit_B_caloporteur = debit_cal_W
        self.TE_B_caloporteur = TE_W_cal                #°C
        #
        self.HV_B = self.HV_W()
        self.hl_B = self.hL_W()    
        #
        self.Qc_bouil = self.run_bouil()
        self.Tf_caloporteur = self.T_sortie_caloporteur()
    
    def Affiche(self):
        print(f"Qc = {self.Qc_bouil/1000000:.{decimal}f} MW")    #!!! on fait un passage en MW pour l'affichage
        print(f"TF_caloporteur = {self.Tf_caloporteur:.{decimal}f} °C")
        
   
    
    

###############################################################################
# Données
# Antoine coefficients for Component A and B (P en bar et T en Kelvin obligatoirement)
Aa, Ba, Ca = 4.72583, 1660.652, -1.461
Ab, Bb, Cb = 4.07827, 1343.943, -53.773

P_colonne_haut = 1.5 # Pression (bar)
P_colonne_bas = 2 

#Colonne à distiller
MAX_etages =150

F = 108.2542842     #flux d'alimentation (kmol/h)
xf = 0.747          #fraction molaire de l'alimentation
xd = 0.995758       #fraction molaire du ditillat
xr = 0.03         #fraction molaire du résidut

q = 1   #fraction de liquide à l'alimentation (L=1, V=0)

R = 1.5    #reflux (R = L/D)
EfficacitePlateaux = 0.95 #efficacité des plateaux réels, nous aurions pu prendre l'efficacité de Murphree et changer le reste du code en fonction pour être précis

resolution = 1E-5      #précision souhaitée

degree = 6 #degrée des polynomes pour les régressions (3+ préférable)

decimal = 8 #strictement visuel pour les résulats, aucun impact sur les calculs

###############################################################################
Pmoyen=(P_colonne_haut+P_colonne_bas)/2         #valeur moyenne de P dans la colonne

points = round(1.0/resolution)

# Generate equilibrium data
xA, yA, temps = equilibrium_data(Aa, Ba, Ca, Ab, Bb, Cb, Pmoyen, points)

xA_haut, yA_haut, temps_haut = equilibrium_data(Aa, Ba, Ca, Ab, Bb, Cb, P_colonne_haut, points)
xA_bas, yA_bas, temps_bas = equilibrium_data(Aa, Ba, Ca, Ab, Bb, Cb, P_colonne_bas, points)

# Fit a polynomial for equilibrium (degree 3+ for example)
poly_coeffs_haut = np.polyfit(xA_haut, yA_haut, degree)
poly_coeffs = np.polyfit(xA, yA, degree)
poly_coeffs_bas = np.polyfit(xA_bas, yA_bas, degree)

poly_coeffs_Tx_haut = np.polyfit(xA_haut, temps_haut, degree)
poly_coeffs_Tx = np.polyfit(xA, temps, degree)
poly_coeffs_Tx_bas = np.polyfit(xA_bas, temps_bas, degree)

poly_coeffs_Ty_haut = np.polyfit(yA_haut, temps_haut, degree)
poly_coeffs_Ty = np.polyfit(yA, temps, degree)
poly_coeffs_Ty_bas = np.polyfit(yA_bas, temps_bas, degree)

###############################################################################
# Plot
plot_equilibrium(xA, yA,points)

plot_polyfit(poly_coeffs,points) #courbe approximée de l'équilibre

plot_txy(poly_coeffs_Tx,poly_coeffs_Ty,'Txy Diagram, fraction Benzène',xr,xf, xd,points,Pmoyen)

###############################################################################
#Colonne à distiller, McCabe-Thiele

b = distillColumn(F,xr,xf, xd, q,EfficacitePlateaux, lambda x, coeffs=poly_coeffs: np.polyval(coeffs, x), R,"Distillation benzène-toluène")
b.run(points,MAX_etages,poly_coeffs)


###############################################################################
#Plot
b.plot(points,"McCabe-Thiele")

b.L_etages(points)

b.V_etages(points)
###############################################################################
#Une fois la colonne obtenue par une première exécution du programme, on calcule l'énergie au condenseur et au rebouilleur en prenant en compte les températures de la colonne
# à P haut
Hv_A_D, Hv_B_D = 31991	  ,  30385.11        #Hv au distillat, kJ/kmol 
hl_A_D, hl_B_D = -8485.3  ,  -15349         #hl au distillat, kJ/kmol

cp_cal_D = 75.377              #kJ/kmol/K     eau
debit_D_calop = 1500            #kmol/h arbitraire
Te_cal_D = 20


Cond_haut = Condenseur(b, hl_A_D, hl_B_D, Hv_A_D, Hv_B_D,cp_cal_D,debit_D_calop,Te_cal_D)
Cond_haut.Affiche()
###############################################################################
#On fait de même pour le rebouilleur, à P bas

Hv_A_W, Hv_B_W = 36606	,  36380        #Hv au distillat, kJ/kmol
hl_A_W, hl_B_W = 8280   ,  4594.6         #hl au distillat, kJ/kmol

cp_cal_W = 75.377             #kJ/kmol eau ici en tant que fluide caloporteur
debit_W_calop =  850      #kmol/h arbitraire
Te_cal_W = 160            #°C

Bouil_bas = Bouilleur(b, hl_A_W, hl_B_W, Hv_A_W, Hv_B_W,cp_cal_W,debit_W_calop,Te_cal_W)
Bouil_bas.Affiche()

