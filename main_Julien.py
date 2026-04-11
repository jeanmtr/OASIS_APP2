import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import soundfile as sf
from scipy.signal import find_peaks,hilbert

SAMPLE_RATE = 44100

ref_notes=frequences = [
    # octave -1
    [16.35, "do-1"], [17.33, "do#-1"], [18.36, "ré-1"], [19.45, "ré#-1"],
    [20.60, "mi-1"], [21.83, "fa-1"], [23.13, "fa#-1"], [24.50, "sol-1"],
    [25.96, "sol#-1"], [27.50, "la-1"], [29.14, "la#-1"], [30.87, "si-1"],
    # octave 0
    [32.70, "do0"], [34.65, "do#0"], [36.71, "ré0"], [38.89, "ré#0"],
    [41.20, "mi0"], [43.65, "fa0"], [46.25, "fa#0"], [49.00, "sol0"],
    [51.91, "sol#0"], [55.00, "la0"], [58.27, "la#0"], [61.74, "si0"],
    # octave 1
    [65.41, "do1"], [69.30, "do#1"], [73.42, "ré1"], [77.78, "ré#1"],
    [82.41, "mi1"], [87.31, "fa1"], [92.50, "fa#1"], [98.00, "sol1"],
    [103.83, "sol#1"], [110.00, "la1"], [116.54, "la#1"], [123.47, "si1"],
    # octave 2
    [130.81, "do2"], [138.59, "do#2"], [146.83, "ré2"], [155.56, "ré#2"],
    [164.81, "mi2"], [174.61, "fa2"], [185.00, "fa#2"], [196.00, "sol2"],
    [207.65, "sol#2"], [220.00, "la2"], [233.08, "la#2"], [246.94, "si2"],
    # octave 3
    [261.63, "do3"], [277.18, "do#3"], [293.66, "ré3"], [311.13, "ré#3"],
    [329.63, "mi3"], [349.23, "fa3"], [369.99, "fa#3"], [392.00, "sol3"],
    [415.30, "sol#3"], [440.00, "la3"], [466.16, "la#3"], [493.88, "si3"],
    # octave 4
    [523.25, "do4"], [554.37, "do#4"], [587.33, "ré4"], [622.25, "ré#4"],
    [659.26, "mi4"], [698.46, "fa4"], [739.99, "fa#4"], [783.99, "sol4"],
    [830.61, "sol#4"], [880.00, "la4"], [932.33, "la#4"], [987.77, "si4"],
    # octave 5
    [1046.50, "do5"], [1108.73, "do#5"], [1174.66, "ré5"], [1244.51, "ré#5"],
    [1318.51, "mi5"], [1396.91, "fa5"], [1479.98, "fa#5"], [1567.98, "sol5"],
    [1661.22, "sol#5"], [1760.00, "la5"], [1864.66, "la#5"], [1975.53, "si5"],
    # octave 6
    [2093.00, "do6"], [2217.46, "do#6"], [2349.32, "ré6"], [2489.02, "ré#6"],
    [2637.02, "mi6"], [2793.83, "fa6"], [2959.96, "fa#6"], [3135.96, "sol6"],
    [3322.44, "sol#6"], [3520.00, "la6"], [3729.31, "la#6"], [3951.07, "si6"],
    # octave 7
    [4186.01, "do7"], [4434.92, "do#7"], [4698.64, "ré7"], [4978.03, "ré#7"],
    [5274.04, "mi7"], [5587.65, "fa7"], [5919.91, "fa#7"], [6271.93, "sol7"],
    [6644.88, "sol#7"], [7040.00, "la7"], [7458.62, "la#7"], [7902.13, "si7"],
    # octave 8
    [8372.02, "do8"], [8869.84, "do#8"], [9397.28, "ré8"], [9956.06, "ré#8"],
    [10548.08, "mi8"], [11175.30, "fa8"], [11839.82, "fa#8"], [12543.86, "sol8"],
    [13289.76, "sol#8"], [14080.00, "la8"], [14917.24, "la#8"], [15804.26, "si8"],
    # octave 9
    [16744.04, "do9"], [17739.68, "do#9"], [18794.56, "ré9"], [19912.12, "ré#9"],
    [21096.16, "mi9"], [22350.60, "fa9"], [23679.64, "fa#9"], [25087.72, "sol9"],
    [26579.52, "sol#9"], [28160.00, "la9"], [29834.48, "la#9"], [31608.52, "si9"],
]


def extract(array,n):
    """la fonction renvoie une liste de tableau numpy, chaque tableau (symbole) correspondant à un enregistrement de n échantillons"""
    L=[]
    d=0
    while(len(array)-d>0):#il reste encore des symboles à analyser dans le tableau
        L.append(array[d:d+n])
        d+=n
    return L


def trouver_note(freq):
    global ref_notes
    i=0
    while not ref_notes[i][0]<=freq<=ref_notes[i+1][0]:
        i+=1
    if abs(ref_notes[i][0]-freq)<abs(ref_notes[i+1][0]-freq):
        return ref_notes[i][1]
    else:
        return ref_notes[i+1][1]


def find_indice_max(array):
    max=array[0]
    indice_max=0
    for indice in range(1,len(array)):
        if array[indice]>max:
            max=array[indice]
            indice_max=indice
    return indice_max
    

def analyse_échantillon(array,Fe):
    transformee=np.fft.fft(array)
    #transformee=np.imag(hilbert(np.real(transformee)))#on prend la partie imaginaire de la transformee de hilbert de transformee
    frequences=np.arange(0,Fe,Fe/(len(array)))
    """plt.close()
    plt.plot(frequences,abs(transformee))
    plt.title("transformée de l'échantillon du signal")
    plt.xlabel("frequence (Hz)")
    plt.ylabel("abs(amplitude)*len(array)")
    plt.show()"""

    #détermination du pic maximal
    demie_transformee_abs=abs(transformee[:len(transformee)//2])

    #analyse du symbole
    indice_max=find_indice_max(demie_transformee_abs)
    frequence=frequences[indice_max]
    print("frequence trouvee ", frequence)
    return frequence



def main(file_name,nbr_échantillons):
    try:
        x, Fe = sf.read(f"sons/{file_name}")
        #x=x[::,0] #on se met en mono (=pareil oreille gauche/droite) ATTENTION si le son est mono cette ligne cause une erreur !!!!
        print("fréquence d'échantillonage : ",Fe)
        #récupérer que le premier coeff de chaque sous-tableau de x
        x=[x[i][0] for i in range(len(x))]
        #calculer le nombre de points par échantillon
        n=int(len(x)/nbr_échantillons+0.5)
        #tracer le signal d'origine
        t=np.linspace(0,len(x)/Fe,len(x))
        plt.close()
        plt.title("signal temporel d'origine non échantilloné")
        plt.plot(t,x)
        plt.xlabel("temps (s)")
        plt.ylabel("amplitude")
        plt.show()        

        #découper le signal en échantillons
        L=extract(x,n)#liste de tableaux numpy d'échantillons
        
        #analyser chaque échantillon 
        resultat=[]
        for elt in L:
            t=np.linspace(0,n/Fe,n)
            """plt.close()
            plt.title("échantillon de signal temporel d'origine")
            plt.plot(t,elt)
            plt.xlabel("temps (s)")
            plt.ylabel("amplitude")
            plt.show()"""
            freq=analyse_échantillon(elt,Fe)
            note=trouver_note(freq)
            print("note : ", note)
            resultat.append(note)
            """reponse=input("suivant ? (y/n) ")
            if reponse=="n":
                return"""
        print("résultat : ",resultat)
    except:
        raise SystemError("vérifier le nom du fichier, sinon c'est une erreur dans le decodeur")

os.read
nbr_échantillons=20
main('gamme_guitare_reel_do_majeur.wav',nbr_échantillons)
