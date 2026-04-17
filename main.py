import soundfile as sf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
import librosa
SAMPLE_RATE = 44100

def freq_to_note(freq):
    ref_notes= [
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
    i=0
    if freq <= ref_notes[0][0]:
        return "jsp"
    while not ref_notes[i][0]<=freq<=ref_notes[i+1][0]:
        i+=1
    if abs(ref_notes[i][0]-freq)<abs(ref_notes[i+1][0]-freq):
        return ref_notes[i][1]
    else:
        return ref_notes[i+1][1]
def enveloppe(signal, attaque):
    x2 = signal**2 
    e = [x2[0]*(1-attaque)]
    for i,ech in enumerate(x2):
        if i != 0:
            e.append(e[i-1]*attaque + (1 - attaque)*ech)


    return e
def deriv(signal, ordre):
    out = []
    for i in range(len(signal)):
        d = 0
        if i > ordre*2 - 1:
            for j in range(ordre):
                d += signal[i - j] - signal[i-j-ordre]
        out.append(d)
    return out
def methode_complexe(signal):
    nfft = 1024  # taille segment avec 0 padding
    F, T, Y_STFT = sp.signal.stft(signal, fs = SAMPLE_RATE, nfft = nfft, nperseg = 1024, noverlap = 512)
    arg = np.angle(Y_STFT)
    
    #on wrap sur -pi pi comme ça un petit changement est vers 0
     
    def wrap(arg):
        return (arg + np.pi) % (2*np.pi) - np.pi

    expected_arg = wrap(2* arg[:, 1:-1] - arg[:,:-2])
    expected_arg = np.pad(expected_arg, ((0,0),(2,0)))

    expected_amp = np.abs(np.pad(Y_STFT[:,:-1], ((0,0),(1,0))))
    
    
    return np.sum(np.abs(Y_STFT- expected_amp*np.exp(1j*expected_arg)),axis = 0)
def creer_dico_profil_notes(taille_fenetre,nb_harmoniques):
    freqs_fft=np.fft.rfftfreq(taille_fenetre,d=1/SAMPLE_RATE)
    notes=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    profils_de_note=[]
    noms_notes_dans_profils=[]
    octave_notes_dans_profils=[]
    for octave in range(2,7):
        for i_note,nom_note in enumerate(notes):
            #on prend comme reference le la4 qui est a 440hert 
            #i_note de la4 = 9, on a 12 demis tons par octave donc
            nb_demi_tons_entre_note_et_la4 = (i_note-9)+(octave-4)*12
            #augmenter d'un demi ton c'est multiplier par 2^(1/12)
            fondamentale = 440*2**(nb_demi_tons_entre_note_et_la4/12)

            profil_de_note=np.zeros(len(freqs_fft))
            for h in range(1,nb_harmoniques+1):
                freq = fondamentale*h 
                #si on depasse pas shannon 
                if freq<SAMPLE_RATE/2:
                    #on trouve le freq echantillonée la plus proche 
                    indice_freq_plus_proche = np.argmin(np.abs(freqs_fft-freq))
                    #en première approxhimation, l'amplitude des harmoniques est en 1/h
                    profil_de_note[indice_freq_plus_proche]+=1/h
            profils_de_note.append(profil_de_note)
            noms_notes_dans_profils.append(nom_note)
            octave_notes_dans_profils.append(octave)
    return np.array(profils_de_note),noms_notes_dans_profils,octave_notes_dans_profils
def detecte_note(signal,debut_notes):
    taille_fenetre=8192
    fft_notes = []
    #on calcule les fenetres où sont jouées les notes
    profils_de_note,nom_notes_dans_profils,octave_notes_dans_profils=creer_dico_profil_notes(taille_fenetre,5)
    fenetres = [(debut_notes[i],debut_notes[i]+taille_fenetre) for i in range(len(debut_notes))]
    #nnls prend une matrice (nb_notes,nb_freqs) mais là on a (nb_freqs,nb_notes) dc 
    profils_de_note = profils_de_note.T
    taille_signal = len(signal)
    toutes_notes_a_afficher=[]
    for (debut,fin) in fenetres : 
        signal_tronque=signal[debut:min(fin,taille_signal-1)]
        window=np.hanning(len(signal_tronque))
        signal_tronque*=window
        #rfft c comme fft mais que la moitié car on a la symetrie
        spectre_signal_tronque = np.abs(np.fft.rfft(signal_tronque)) 

        regression,_=sp.optimize.nnls(profils_de_note,spectre_signal_tronque)
        regression = regression/regression.sum()
        val_max = np.max(regression)
        noms_notes_deja_calculee=[]
        notes_a_afficher=[]
        for i,val in enumerate(regression):
            if val>val_max*0.7 and not (nom_notes_dans_profils[i] in noms_notes_deja_calculee):
                noms_notes_deja_calculee.append(nom_notes_dans_profils[i])
                notes_a_afficher.append(f"{nom_notes_dans_profils[i]}{octave_notes_dans_profils[i]}")
        print(notes_a_afficher)
        toutes_notes_a_afficher.append(notes_a_afficher)
    tableau_notes_elargies =[[] for i in range(len(signal))]

    for i in range(len(debut_notes)):
        if i+1==len(debut_notes):
            fin=len(signal) 
        else :
            fin=debut_notes[i+1]
        for j in range(debut_notes[i],fin):
            tableau_notes_elargies[j]=toutes_notes_a_afficher[i]
    tableau_notes_elargies=tableau_notes_elargies[:len(signal)]

    """
    # decommenter si besoin de plot les notes, ce print a été fait par IA car 
    #c'est juste un print
    ordre_chromatique = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    def rang_note(note_str):
        # note_str est du style "E3-0.123", on extrait juste la note
        note_str = note_str.split('-')[0]
        for i in range(len(note_str)-1, 0, -1):
            if note_str[i].isdigit():
                return int(note_str[i:]) * 12 + ordre_chromatique.index(note_str[:i])

    toutes_les_notes = sorted(
        {note for notes in tableau_notes_elargies for note in notes},
        key=rang_note
    )
    note_to_y = {note: i for i, note in enumerate(toutes_les_notes)}

    pas = max(1, SAMPLE_RATE // 100)
    t_axis, y_axis = [], []
    for n in range(0, len(tableau_notes_elargies), pas):
        for note in tableau_notes_elargies[n]:
            t_axis.append(n / SAMPLE_RATE)
            y_axis.append(note_to_y[note])

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.scatter(t_axis, y_axis, s=3, color='steelblue')
    ax.set_yticks(range(len(toutes_les_notes)))
    ax.set_yticklabels(toutes_les_notes, fontsize=8)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Note")
    ax.set_title("Notes jouées au fil du temps")
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show() """
    return tableau_notes_elargies 
def detecte_instrument(frequences,transformee_fourier,indices_peaks):
    #on applique une méthode de recherche de pics
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

    if 1/5*abs(liste_triee[0][1]) > max(abs(elt) for elt in liste_triee[1:][1]):#fréquence fondamentale bien plus forte que toutes les autres fréquences
        return "drums"
    return "guitare"
def analyse(sample,Fe):
    global SAMPLE_RATE
    SAMPLE_RATE=Fe 
    if(sample.ndim==2):
        sample = sample[::,0] 
    #0 padding pour eviter les erreurs de dimensions dans la découpe de fenetre
    sample = np.concatenate([sample,np.zeros(10000)])
    t = np.arange(len(sample))/SAMPLE_RATE
    mc = methode_complexe(sample)

    t_frames = np.linspace(0, 1, len(mc))
    t_samples = np.linspace(0, 1, len(sample))
    detec_upsample = np.interp(t_samples, t_frames,  mc)
    tableau_depart_notes =[]

    indices_pics, _ = find_peaks(detec_upsample,prominence =0.02,  distance=SAMPLE_RATE/20 )

    tableau_depart_notes = []
    for i in indices_pics:
        tableau_depart_notes.append(i)


    tableau_notes = detecte_note(sample,tableau_depart_notes)
    instrument = detecte_instrument(np.fft.fftfreq(len(sample),d=1/Fe),np.fft.fft(sample),indices_pics)
    return tableau_notes,instrument 
if __name__=="__main__":
    sample,f=sf.read("gamme__guitare.wav")
    print(analyse(sample,f))
