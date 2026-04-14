import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def detecte_instrument(frequences,transformee_fourier):
    plt.close()
    plt.plot(frequences,transformee_fourier)
    plt.title("amplitude en fonction des fréquences")
    plt.show()
    #on applique une méthode de recherche de pics
    indices_peaks, _ = find_peaks(transformee_fourier, height=0)
    couples_indices_amplitude_pics=[(indice,transformee_fourier[indice]) for indice in indices_peaks]
    #on range la liste coupels_indices_amplitude_pics en ordre décroissant selon les amplitudes     
    liste_triee = sorted(couples_indices_amplitude_pics, key=lambda x: x[1], reverse=True)
    print(liste_triee)
    liste_triee=liste_triee[:4]#on ne garde que les 5 premiers plus gros pics pour éviter d'avoir trop de bruit et pouvoir quand même reconnaître l'instrument
    print("blablabla")
    print(liste_triee)




