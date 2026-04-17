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
    """print("frequences : ", x)
    print("amplitudes : ",y)
    plt.close()
    plt.plot(x,y,"x")
    plt.title(f"spectre de la note à la position {position_temps} 5 pics les plus gros")
    plt.xlabel("fréquence en Hz")
    plt.ylabel("amplitude")
    plt.show()"""
    # déterminer le type d'instrument
    # on fait un truc pourri mais qui renvoie un résultat
    if 2/3*y[0]>y[1]:#première fréquence plus forte que les suivantes
        return "guitare"
    if 1/5*abs(liste_triee[0][1]) > max(abs(elt) for elt in liste_triee[1:][1]):#fréquence fondamentale bien plus forte que toutes les autres fréquences
        return "drums"
    else:
        return "voice"



