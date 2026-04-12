import numpy as np

class SIR:

    def __init__(self,params:dict[str,float]):
        self.params=params

    def rk4(x, dx, y, deriv,params):
        """
        /*-----------------------------------------
        sous programme de resolution d'equations
        differentielles du premier ordre par
        la methode de Runge-Kutta d'ordre 4
        x = abscisse, une valeur scalaire, par exemple le temps
        dx = pas, par exemple le pas de temps
        y = valeurs des fonctions au temps t(i), c'est un tableau numpy de taille n
        avec n le nombre d'équations différentielles du 1er ordre
        
        rk4 renvoie les nouvelles valeurs de y pour t(i+1)
        
        deriv = variable contenant le nom du
        sous-programme qui calcule les derivees
        deriv doit avoir trois arguments: deriv(x,y,params) et renvoyer 
        un tableau numpy dy de taille n 
        ----------------------------------------*/
        """
        #  /* d1, d2, d3, d4 = estimations des derivees
        #    yp = estimations intermediaires des fonctions */  
        ddx = dx/2.       #         /* demi-pas */
        d1 = deriv(x,y,params)   #       /* 1ere estimation */          
        yp = y + d1*ddx
        #    for  i in range(n):
        #        yp[i] = y[i] + d1[i]*ddx
        d2 = deriv(x+ddx,yp,params)     #/* 2eme estimat. (1/2 pas) */
        yp = y + d2*ddx    
        d3 = deriv(x+ddx,yp,params)  #/* 3eme estimat. (1/2 pas) */
        yp = y + d3*dx    
        d4 = deriv(x+dx,yp,params)     #  /* 4eme estimat. (1 pas) */
        #/* estimation de y pour le pas suivant en utilisant
        #  une moyenne ponderee des derivees en remarquant
        #  que : 1/6 + 1/3 + 1/3 + 1/6 = 1 */
        return y + dx*( d1 + 2*d2 + 2*d3 + d4 )/6  

    def deriv_SIR(self,t,y,params):
        """
        Derivees pour SIR
        """
        beta = params["beta"]
        gamma= params["gamma"]
        dy = np.zeros(3)
        dy[0] = -beta*y[0]*y[1] # dérivée de S
        dy[1] = beta*y[0]*y[1]-gamma*y[1] # dérivée de I
        dy[2] = gamma*y[1] # dérivée de R

        return dy
    
    def deriv_SIRD_2groupes_beta_matrix(self, t, y, params):
        """
            Derivees pour SIRD
        """
        s, i, r, d, s_v, i_v, r_v, d_v = y

        beta = np.array(params["beta"])  
        # beta = [[beta_JJ, beta_JV],
        #         [beta_VJ, beta_VV]]

        gamma = params["gamma"]
        gamma_V = params["gamma_v"]
        mu = params["mu"]
        mu_V = params["mu_v"]

        dy = np.zeros(8)

        # Forces d'infection
        lambda_S = beta[0, 0] * i + beta[0, 1] * i_v
        lambda_V = beta[1, 0] * i + beta[1, 1] * i_v

        # Groupe des non vulnérables
        dy[0] = -s * lambda_S
        dy[1] = s * lambda_S - (gamma + mu) * i
        dy[2] = gamma * i
        dy[3] = mu * i

        # Groupe vulnérables
        dy[4] = -s_v * lambda_V
        dy[5] = s_v * lambda_V - (gamma_V + mu_V) * i_v
        dy[6] = gamma_V * i_v
        dy[7] = mu_V * i_v

        return dy
    
    def deriv_SIRD(self, t, y, params):
        """
        Derivees pour SIRD
        """
        beta = params["beta"]
        gamma= params["gamma"]
        mu= params["mu"]
        dy = np.zeros(4)
        dy[0] = -beta*y[0]*y[1] # dérivée de S
        dy[1] = beta*y[0]*y[1]-gamma*y[1] - mu*y[1] # dérivée de I
        dy[2] = gamma*y[1] # dérivée de R
        dy[3] = mu*y[1] # Dérivée de D

        return dy
    
    def deriv_SIRD_dev(self, t, y, params):
        """
        Derivees pour SIR
        """
        beta = params["beta"]
        gamma= params["gamma"]
        mu= params["mu"]
        rho= params["rho"]
        dy = np.zeros(4)
        dy[0] = -beta*y[0]*y[1] + rho*y[2] # dérivée de S
        dy[1] = beta*y[0]*y[1]-gamma*y[1] - mu*y[1] # dérivée de I
        dy[2] = gamma*y[1] - rho*y[2] # dérivée de R
        dy[3] = mu*y[1] # Dérivée de D

        return dy

    def deriv_SIRDV(self, t, y, params):
        """
        Derivees pour SIR
        """
        beta = params["beta"]
        gamma= params["gamma"]
        mu = params["mu"]
        omega = params["omega"]
        dy = np.zeros(5)
        dy[0] = -beta*y[0]*y[1] - omega*y[0] # dérivée de S
        dy[1] = beta*y[0]*y[1] -gamma*y[1] - mu*y[1] # dérivée de I
        dy[2] = gamma*y[1] # dérivée de R
        dy[3] = mu*y[1] # Dérivée de D
        dy[4] = omega*y[0] # Dérivée de V

        return dy

    def euler(self,t, dt, y, deriv):
        """
        Un pas de Euler : y(t+dt) = y(t) + dt*y'(t)
        """
        params=self.params
        dy = deriv(t, y, params)
        y[:] = y + dt*dy
        return y
    
    def laplacien_neumann(self, I, h):
        """
        Calcule le laplacien discret 2D de la matrice I
        avec condition au bord sans flux (Neumann)
        """

        Ipad = np.pad(I, ((1, 1), (1, 1)), mode="edge")

        L = (
            Ipad[2:, 1:-1] +
            Ipad[:-2, 1:-1] +
            Ipad[1:-1, 2:] +
            Ipad[1:-1, :-2] -
            4 * Ipad[1:-1, 1:-1]
        ) / h**2

        return L
    
    def deriv_SIRD_euler_explicite_spatiale(self, t, y, params):
        """
            Derivees pour SIRD
        """
        s, i, r, d = y

        beta = params["beta"] 
        gamma = params["gamma"]
        mu = params["mu"]
        D = params["D"]
        h = params["h"]
        dy = np.zeros_like(y)

        # Groupe des non vulnérables
        l=self.laplacien_neumann(i,h)
        dy[0] = -s * beta
        dy[1] = s * beta - (gamma + mu) * i +D*l
        dy[2] = gamma * i
        dy[3] = mu * i

        return dy

    def euler(self,t, dt, y, deriv):
        """
        Un pas de Euler : y(t+dt) = y(t) + dt*y'(t)
        """
        params=self.params
        dy = deriv(t, y, params)
        y[:] = y + dt*dy
        return y
    
class COVID19:

    def __init__(self,params:dict[str,float]):
        self.params=params

    def rk4(x, dx, y, deriv,params):
        """
        /*-----------------------------------------
        sous programme de resolution d'equations
        differentielles du premier ordre par
        la methode de Runge-Kutta d'ordre 4
        x = abscisse, une valeur scalaire, par exemple le temps
        dx = pas, par exemple le pas de temps
        y = valeurs des fonctions au temps t(i), c'est un tableau numpy de taille n
        avec n le nombre d'équations différentielles du 1er ordre
        
        rk4 renvoie les nouvelles valeurs de y pour t(i+1)
        
        deriv = variable contenant le nom du
        sous-programme qui calcule les derivees
        deriv doit avoir trois arguments: deriv(x,y,params) et renvoyer 
        un tableau numpy dy de taille n 
        ----------------------------------------*/
        """
        #  /* d1, d2, d3, d4 = estimations des derivees
        #    yp = estimations intermediaires des fonctions */  
        ddx = dx/2.       #         /* demi-pas */
        d1 = deriv(x,y,params)   #       /* 1ere estimation */          
        yp = y + d1*ddx
        #    for  i in range(n):
        #        yp[i] = y[i] + d1[i]*ddx
        d2 = deriv(x+ddx,yp,params)     #/* 2eme estimat. (1/2 pas) */
        yp = y + d2*ddx    
        d3 = deriv(x+ddx,yp,params)  #/* 3eme estimat. (1/2 pas) */
        yp = y + d3*dx    
        d4 = deriv(x+dx,yp,params)     #  /* 4eme estimat. (1 pas) */
        #/* estimation de y pour le pas suivant en utilisant
        #  une moyenne ponderee des derivees en remarquant
        #  que : 1/6 + 1/3 + 1/3 + 1/6 = 1 */
        return y + dx*( d1 + 2*d2 + 2*d3 + d4 )/6 

    def model(y, t, beta, gamma, mu, nu_max, t_start):
        S, I, R, D, V = y
        N = S + I + R + D + V
    
        # Condition de déclenchement
        if t < t_start:
            nu = 0
        else:
            nu = nu_max
    
        dSdt = -beta * S * I / N - nu * S
        dIdt = beta * S * I / N - (gamma + mu) * I
        dRdt = gamma * I
        dDdt = mu * I
        dVdt = nu * S
    
        return [dSdt, dIdt, dRdt, dDdt, dVdt]
    
    def deriv(self, t, y, params):
        """
        Derivees pour SIR
        """
        beta = params["beta"]
        gamma = params["gamma"]
        mu = params["mu"]
        omega_max = params["omega"]
        rho = params["rho"]
        t_start = params["t_start"]

        # CONDITION DE DÉBUT DE VACCINATION
        if t < t_start :
            omega = 0
        else:
            omega = omega_max

        dy = np.zeros(5)
        dy[0] = -beta*y[0]*y[1] - omega*y[0] + rho*y[4] + rho*y[2] # dérivée de S
        dy[1] = beta*y[0]*y[1] -gamma*y[1] - mu*y[1] # dérivée de I
        dy[2] = gamma*y[1] - rho*y[2] # dérivée de R
        dy[3] = mu*y[1] # Dérivée de D
        dy[4] = omega*y[0] - rho*y[4] # Dérivée de V

        return dy
    
    def euler(self,t, dt, y, deriv):
        """
        Un pas de Euler : y(t+dt) = y(t) + dt*y'(t)
        """
        params=self.params
        dy = deriv(t, y, params)
        y[:] = y + dt*dy
        return y
    
class ZOMBIE:
    
    def __init__(self,params:dict[str,float]):
        self.params=params

    def rk4(x, dx, y, deriv,params):
        """
        /*-----------------------------------------
        sous programme de resolution d'equations
        differentielles du premier ordre par
        la methode de Runge-Kutta d'ordre 4
        x = abscisse, une valeur scalaire, par exemple le temps
        dx = pas, par exemple le pas de temps
        y = valeurs des fonctions au temps t(i), c'est un tableau numpy de taille n
        avec n le nombre d'équations différentielles du 1er ordre
        
        rk4 renvoie les nouvelles valeurs de y pour t(i+1)
        
        deriv = variable contenant le nom du
        sous-programme qui calcule les derivees
        deriv doit avoir trois arguments: deriv(x,y,params) et renvoyer 
        un tableau numpy dy de taille n 
        ----------------------------------------*/
        """
        #  /* d1, d2, d3, d4 = estimations des derivees
        #    yp = estimations intermediaires des fonctions */  
        ddx = dx/2.       #         /* demi-pas */
        d1 = deriv(x,y,params)   #       /* 1ere estimation */          
        yp = y + d1*ddx
        #    for  i in range(n):
        #        yp[i] = y[i] + d1[i]*ddx
        d2 = deriv(x+ddx,yp,params)     #/* 2eme estimat. (1/2 pas) */
        yp = y + d2*ddx    
        d3 = deriv(x+ddx,yp,params)  #/* 3eme estimat. (1/2 pas) */
        yp = y + d3*dx    
        d4 = deriv(x+dx,yp,params)     #  /* 4eme estimat. (1 pas) */
        #/* estimation de y pour le pas suivant en utilisant
        #  une moyenne ponderee des derivees en remarquant
        #  que : 1/6 + 1/3 + 1/3 + 1/6 = 1 */
        return y + dx*( d1 + 2*d2 + 2*d3 + d4 )/6 
    
    def deriv(self, t, y, params):
        """
        Derivees pour SIR
        """
        beta = params["beta"]
        gamma = params["gamma"]
        mu = params["mu"]
        rho = params["rho"]

        dy = np.zeros(5)
        dy[0] = -beta*y[0]*y[1] - omega*y[0] + rho*y[4] + rho*y[2] # dérivée de S
        dy[1] = beta*y[0]*y[1] -gamma*y[1] - mu*y[1] # dérivée de I
        dy[2] = gamma*y[1] - rho*y[2] # dérivée de R
        dy[3] = mu*y[1] # Dérivée de D
        dy[4] = omega*y[0] - rho*y[4] # Dérivée de V

        return dy
    
    def deriv_ZOMBIE(self, t, y, params):
        """
        Derivees pour SIRD
        """
        beta = params["beta"]
        gamma= params["gamma"]
        mu= params["mu"]
        alpha= params["alpha"]
        dy = np.zeros(4)
        dy[0] = - mu*y[0]*y[1] - beta*y[0]*y[1] # dérivée de S
        dy[1] = gamma*y[2] - alpha*y[0]*y[1] # dérivée de I
        dy[2] = -gamma*y[2] + beta*y[0]*y[1] # dérivée de R
        dy[3] = alpha*y[0]*y[1] + mu*y[0]*y[1] # Dérivée de D

        return dy
    
    def euler(self,t, dt, y, deriv):
        """
        Un pas de Euler : y(t+dt) = y(t) + dt*y'(t)
        """
        params=self.params
        dy = deriv(t, y, params)
        y[:] = y + dt*dy
        return y