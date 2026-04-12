import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nosPackages.mathUtils as mathUtils

# population initial
S_init=0.999 # personnes suscétibles (départ) en fraction -> 1=100%
I_init=0.001 # personnes infectés
R_init=0 # personnes guérit

# paramètre de contagion
beta=0.558 # taux de contagion journalier de base pour le covid = 0.6 personnes/jour
gamma=0.2 # 1/gamma = nb de jour contagieux = 5 jour pour le covid
R0=beta/gamma*S_init # nombre de personnes contaminés par un malade = 3 pour le covid

dt = 0.1
tmax = 100.0
temps = np.arange(0, tmax + dt, dt)
n = temps.size

S = np.empty(n)
I = np.empty(n)
R = np.empty(n)
Rt = np.empty(n)

y = np.empty(3)
y[0] = S_init
y[1] = I_init
y[2] = R_init

params={"beta":beta,"gamma":gamma}
model=mathUtils.SIR(params)
deriv=model.deriv_SIR
euler=model.euler

for i in range(n):
    S[i] = y[0]
    I[i] = y[1]
    R[i] = y[2]
    t = temps[i]
    y = euler(t, dt, y, deriv)
Rt = beta/gamma*S # evolution du R0 (contagion pour une personne)


#graphes

plt.figure()
plt.title("Evolution de S(t), I(t), R(t) et Rt")

ax1 = plt.gca()
ax1.plot(temps, S, label="S(t)")
ax1.plot(temps, I, label="I(t)")
ax1.plot(temps, R, label="R(t)")
ax1.set_xlabel("temps")
ax1.set_ylabel("S(t), I(t), R(t)")
ax1.legend(loc="upper right")

ax2 = ax1.twinx()
ax2.plot(temps, Rt, "--", label="Rt")
ax2.axhline(1, linestyle=":", label="Rt = 1")
ax2.set_ylabel("Rt")
ax2.legend(loc="center right")

plt.savefig("sir_SIR_Rt.pdf", bbox_inches='tight')
plt.show()



gamma = 0.2
dt = 0.1
tmax = 100.0
temps = np.arange(0, tmax + dt, dt)
n = temps.size

# boucle sur beta : de 0.2 à 3.0 , donc de R0=1 à 15
betaTab = np.arange(0.2, 3.0 + 0.2, 0.2)

# vecteurs résultats
R0List = np.empty(betaTab.size)
RtFinal = np.empty(betaTab.size)
RFinal = np.empty(betaTab.size)
immunisesFinal = np.empty(betaTab.size)
epidemie_endigue_list = np.empty(betaTab.size)
epidemie_endigue=False

for k, beta in enumerate(betaTab):

    R0 = beta / gamma * S_init
    R0List[k] = R0
    S = np.empty(n)
    I = np.empty(n)
    R = np.empty(n)
    y = np.empty(3)
    y[0] = S_init
    y[1] = I_init
    y[2] = R_init

    params = {"beta": beta, "gamma": gamma}
    model = mathUtils.SIR(params)
    deriv = model.deriv_SIR
    euler = model.euler

    for i in range(n):
        S[i] = y[0]
        I[i] = y[1]
        R[i] = y[2]
        Rt= S[i]*R0
        if not epidemie_endigue and Rt<=1  :
            epidemie_endigue = True
            epidemie_endigue_list[k]=R[i]

        t = temps[i]
        y = euler(t, dt, y, deriv)

    immunisesFinal[k] = 100 * R[-1]
    
    
plt.figure()
plt.title("% de la popatOn immunisée à la fin de l'épidemie en fnctO de R0")
plt.plot(R0List, immunisesFinal)
plt.xlabel("R0")
plt.ylabel("Rt final")
plt.savefig("sir_Rtfinal_vs_R0.pdf", bbox_inches='tight')
plt.show()
    

# par virus

virus = [
    "Ebola",
    "Grippe saisonniere",
    "COVID initial",
    "COVID Delta",
    "COVID Omicron",
    "Variole",
    "Rougeole"
]

R0 = [
    1.95,   # Ebola
    1.27,   # grippe saisonniere
    2.79,   # COVID souche initiale
    5.08,   # Delta
    9.5,    # Omicron
    4.75,   # variole
    15.0    # rougeole
]

# DataFrame
df = pd.DataFrame({
    "Virus": virus,
    "R0": R0
})

display(df)

x = np.arange(len(virus))
plt.figure(figsize=(10, 5))
plt.bar(x, R0)
plt.xticks(x, virus, rotation=30)
plt.xlabel("Virus")
plt.ylabel("R0")
plt.title("Nombre de reproduction de base R0 selon le virus")

for i, v in enumerate(R0):
    plt.text(i, v + 0.2, str(v), ha="center")

plt.savefig("histogramme_R0_virus.pdf", bbox_inches='tight')
plt.show()


gamma = 0.2
dt = 0.1
tmax = 100.0
temps = np.arange(0, tmax + dt, dt)
n = temps.size

virus = [
    "MERS",
    "Grippe saisonniere",
    "Grippe H1N1 2009",
    "Ebola",
    "COVID initial",
    "Variole",
    "COVID Delta",
    "COVID Omicron",
    "Rougeole"
]

R0list = np.array([
    0.69,
    1.27,
    1.46,
    1.95,
    2.79,
    4.75,
    5.08,
    9.5,
    15.0
])

# beta = gamma * R0  (si S_init = 1)
betaTab = gamma * R0list

R0Final = np.empty(betaTab.size)
RFinal = np.empty(betaTab.size)
immunisesFinal = np.empty(betaTab.size)
epidemie_endigue_list = np.empty(betaTab.size)


for k, beta in enumerate(betaTab):
    epidemie_endigue=False
    R0 = beta / gamma * S_init
    R0Final[k] = R0

    S = np.empty(n)
    I = np.empty(n)
    R = np.empty(n)

    y = np.empty(3)
    y[0] = S_init
    y[1] = I_init
    y[2] = R_init

    params = {"beta": beta, "gamma": gamma}
    model = mathUtils.SIR(params)
    deriv = model.deriv_SIR
    euler = model.euler

    for i in range(n):
        S[i] = y[0]
        I[i] = y[1]
        R[i] = y[2]
        t = temps[i]
        y = euler(t, dt, y, deriv)
        Rt= S[i]*R0
        if not epidemie_endigue and Rt<=1  :
            epidemie_endigue = True
            epidemie_endigue_list[k]=(1-S[i])*100

    RFinal[k] = R[-1]
    immunisesFinal[k] = 100 * R[-1]

x = np.arange(len(virus))

labels = [
    f"{virus[i]}\n(R0={R0list[i]:.2f})"
    for i in range(len(virus))
]

plt.figure(figsize=(10, 5))
plt.bar(x, epidemie_endigue_list)

plt.xticks(x, labels, rotation=30)
plt.xlabel("Maladie (R0)")
plt.ylabel("Population immunisée")
plt.title("Population immunisée necessaire pour endiguer l'epidemie")

for i, v in enumerate(epidemie_endigue_list):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center")

plt.savefig("histogramme_immunités.pdf", bbox_inches='tight')
plt.show()



plt.figure(figsize=(10, 5))
plt.bar(x, immunisesFinal)

plt.xticks(x, labels, rotation=30)
plt.xlabel("Maladie (R0)")
plt.ylabel("Population immunisée")
plt.title("Population immunisée à la fin de la maladie (sans confinement etc)")

for i, v in enumerate(immunisesFinal):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center")

plt.savefig("histogramme_immunités fin de maladie", bbox_inches='tight')
plt.show()
    
    

