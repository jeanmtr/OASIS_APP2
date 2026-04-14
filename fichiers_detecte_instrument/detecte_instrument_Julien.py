import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np


def detecte_instrument(frequences,transformee_fourier,position_temps):
    #on applique une méthode de recherche de pics
    indices_peaks, _ = find_peaks(transformee_fourier, height=0)
    couples_indices_amplitude_pics=[(indice,transformee_fourier[indice]) for indice in indices_peaks]
    #on range la liste coupels_indices_amplitude_pics en ordre décroissant selon les amplitudes     
    liste_triee = sorted(couples_indices_amplitude_pics, key=lambda x: x[1], reverse=True)
    liste_triee=liste_triee[:5]#on ne garde que les 5 premiers plus gros pics pour éviter d'avoir trop de bruit et pouvoir quand même reconnaître l'instrument
    #on récupère les infos pour pouvoir tracer
    x=np.array([frequences[abs(liste_triee[i][0])] for i in range(len(liste_triee))])#valeur des fréquences
    y=np.array([abs(liste_triee[i][1]) for i in range(len(liste_triee))])#valeur des amplitudes
    #afficher le résultat
    print("frequences : ", x)
    print("amplitudes : ",y)
    plt.close()
    plt.plot(x,y,"x")
    plt.title(f"spectre de la note à la position {position_temps} 5 pics les plus gros")
    plt.xlabel("fréquence en Hz")
    plt.ylabel("amplitude")
    plt.show()



