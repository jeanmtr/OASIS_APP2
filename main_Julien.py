import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import soundfile as sf
from scipy.signal import find_peaks,hilbert

SAMPLE_RATE = 44100
NOM_FICHIER='gamme_guitare_reel_do_majeur.wav'

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

def trouver_note(freq):
    global ref_notes
    i=0
    while not ref_notes[i][0]<=freq<=ref_notes[i+1][0]:
        i+=1
    if abs(ref_notes[i][0]-freq)<abs(ref_notes[i+1][0]-freq):
        return ref_notes[i][1]
    else:
        return ref_notes[i+1][1]

# ---------------------------------------------------------------------------
# Calcul de l'enveloppe d'énergie (filtre IIR du 1er ordre)
# ---------------------------------------------------------------------------
def enveloppe(signal, attaque):
    """
    Calcule l'enveloppe d'énergie du signal via un filtre récursif (IIR 1er ordre).
    
    Formule :  e[n] = alpha * e[n-1] + (1 - alpha) * x²[n]
    
    Plus `attaque` (alpha) est proche de 1, plus le lissage est fort
    (l'enveloppe réagit lentement aux variations brusques d'amplitude).
    
    Plot associé (dans detect_notes) :
        courbe rouge — enveloppe lissée superposée au signal brut bleu.
    """
    x2 = signal**2               # Puissance instantanée (énergie échantillon par échantillon)
    e = [x2[0] * (1 - attaque)]  # Initialisation : premier échantillon lissé
    for i, ech in enumerate(x2):
        if i != 0:
            # Lissage exponentiel : compromis entre la valeur passée et la puissance courante
            e.append(e[i-1] * attaque + (1 - attaque) * ech)
    return e


# ---------------------------------------------------------------------------
# Approximation de la dérivée par différences finies d'ordre N
# ---------------------------------------------------------------------------
def deriv(signal, ordre):
    """
    Calcule une dérivée discrète approximée par différences finies d'ordre `ordre`.
    
    Pour chaque échantillon i, on somme les différences :
        d[i] = Σ_{j=0}^{ordre-1}  (signal[i-j] - signal[i-j-ordre])
    
    Cela revient à comparer une fenêtre récente avec une fenêtre passée
    de même longueur. Plus `ordre` est grand, plus la dérivée est lissée
    et insensible au bruit rapide — au prix d'un retard (latence = 2*ordre).
    
    Les `2*ordre` premiers échantillons valent 0 (pas assez d'historique).
    
    Plot associé (dans detect_notes) :
        courbe jaune — dérivée seconde de l'enveloppe (mesure la vitesse
        de montée d'énergie, pic = attaque d'une note).
    """
    out = []
    for i in range(len(signal)):
        d = 0
        if i > ordre * 2 - 1:      # On attend d'avoir assez d'historique
            for j in range(ordre):
                d += signal[i - j] - signal[i - j - ordre]
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Détection des instants d'attaque des notes (onsets)
# ---------------------------------------------------------------------------
def detect_notes(signal, tresh, cooldown, attack, deriv_order):
    """
    Repère les instants où une nouvelle note commence (onset detection).
    
    Algorithme :
      1. Calcul de l'enveloppe d'énergie (lissage IIR).
      2. Calcul de la dérivée de l'enveloppe (vitesse de montée d'énergie).
      3. Détection par seuil relatif : on déclenche un onset quand
         l'enveloppe courante dépasse de 9 % la moyenne locale récente
         (dernière fenêtre de 100 échantillons).
      4. Un cooldown (période de silence forcée après chaque détection)
         évite les multi-détections sur la même note.
    
    Plot généré :
      ┌─────────────────────────────────────────────────────────┐
      │  signal brut      (bleu)                                │
      │  moyenne locale × 50   (vert)  — niveau de référence    │
      │  enveloppe × 50        (rouge) — énergie lissée         │
      │  dérivée 2nde × 50     (jaune) — pics = attaques        │
      │  lignes verticales rouges — positions des onsets détect.│
      └─────────────────────────────────────────────────────────┘
    Les facteurs ×50 servent uniquement à l'affichage (mise à l'échelle
    pour que les courbes soient visibles sur le même axe que le signal).
    
    Paramètres :
      tresh      : (non utilisé dans l'implémentation courante, conservé
                    pour compatibilité)
      cooldown   : durée minimale entre deux onsets (en échantillons)
      attack     : coefficient alpha du filtre IIR (0 < alpha < 1)
      deriv_order: ordre des différences finies pour la dérivée
    
    Retourne : liste des positions (en échantillons) des onsets détectés.
    """
    env = enveloppe(signal, attack)
    env_deriv = deriv(env, deriv_order)  # Dérivée de l'enveloppe (valeurs à optimiser)
    cd = 0             # Compteur de cooldown (nombre d'échantillons restants à ignorer)
    notes_pos = []     # Positions des onsets détectés
    mean = []          # Historique de la moyenne locale (pour l'affichage)
    max_val = np.max(env_deriv)

    for i, ech in enumerate(env_deriv):
        # Moyenne locale sur les 1000 derniers échantillons (≈ 23 ms) pour affichage
        recent_avg = np.mean(env[i - 1000 : i])
        mean.append(recent_avg)

        if cd > 0:
            cd -= 1   # On est encore en période de cooldown : on ignore cet échantillon
            continue
        else:
            # Seuil adaptatif : moyenne locale sur les 100 derniers échantillons
            recent_avg = np.mean(env[i - 100 : i])
            if env[i] > 1.09 * recent_avg:  # +9 % au-dessus de la moyenne locale → onset
                notes_pos.append(i)
                cd = cooldown  # Déclenche le cooldown pour éviter les doubles détections

    # --- Visualisation ---
    t = np.arange(len(signal)) / SAMPLE_RATE
    if notes_pos:
        plt.axvline(x=notes_pos[0] / SAMPLE_RATE, color='r', label="onsets détectés")
    for x in notes_pos:
        plt.axvline(x=x / SAMPLE_RATE, color='r')              # Trait rouge vertical = onset
    plt.plot(t, signal, color='b', label="signal brut")                 # Signal brut
    plt.plot(t, np.array(mean) * 50, color='g', label="moyenne locale ×50")   # Moyenne locale (×50 pour affichage)
    plt.plot(t, np.array(enveloppe(signal, attack)) * 50,
             color='r', label="enveloppe ×50")                            # Enveloppe (×50 pour affichage)
    # plt.plot(t, env_deriv, color='g', label="deriv 1e")       # Dérivée 1ère (désactivée)
    plt.plot(t, deriv(env_deriv, 10), color='y',
             label="dérivée 2nde ×50")                             # Dérivée 2nde (ordre 10)
    plt.title("Détection des onsets et enveloppe d'énergie")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude / niveau normalisé")
    plt.legend()
    plt.show()

    return notes_pos


# ---------------------------------------------------------------------------
# Estimation des pitchs par STFT
# ---------------------------------------------------------------------------
def trouver_pitchs(signal):
    """
    Pour chaque onset détecté, extrait la ou les fréquences dominantes
    via la STFT (Short-Time Fourier Transform).
    
    Étapes :
      1. STFT avec zero-padding (nfft=16384) pour une résolution fréquentielle
         fine (Δf = 44100/16384 ≈ 2.7 Hz).
      2. Détection des onsets (via detect_notes).
      3. Pour chaque onset :
         a. On localise la colonne STFT correspondant à l'instant de l'onset
            (+ pitch_offset = 10 trames pour laisser la note s'établir).
         b. On identifie le pic spectral principal (argmax de |STFT|).
         c. On collecte tous les pics secondaires > 2/3 du maximum
            (potentiels harmoniques ou notes simultanées), éloignés d'au
            moins 10 bins du principal.
         d. On trie ces pics par amplitude décroissante et on les convertit
            en noms de notes.
    
    Plot généré pour chaque onset :
      ┌─────────────────────────────────────────────────────────┐
      │  spectre |STFT[:, trame]|   (bleu)                      │
      │  seuil 2/3 du max           (tirets horizontaux)        │
      │  fréquences retenues        (traits verticaux rouges)   │
      └─────────────────────────────────────────────────────────┘
    L'axe X est en Hz (fréquences réelles).
    
    Retourne : liste de tuples (position_échantillon, temps_s, [noms_de_notes])
    """
    nfft = 4096 * 4   # Taille FFT avec zero-padding : haute résolution fréquentielle
    pitch_offset = 10  # Décalage en trames STFT après l'onset (laisse la note s'installer)

    # Calcul de la STFT complète du signal
    # F  : vecteur des fréquences (Hz)
    # T  : vecteur des instants (s)
    # Y_STFT : matrice complexe (fréquences × trames)
    F, T, Y_STFT = sp.signal.stft(signal, fs=SAMPLE_RATE, nperseg=1024, nfft=nfft)


    notes_pos = detect_notes(signal, 0.025, SAMPLE_RATE / 10, 0.9995, 400)

    out = []
    for note in notes_pos:
        time = note / SAMPLE_RATE  # Conversion échantillon → secondes
        closest = 0
        # Recherche de la trame STFT la plus proche de l'instant de l'onset
        while T[closest] < time:
            closest += 1

        # Analyse spectrale à la trame décalée (onset + offset)
        argmax = np.argmax(np.abs(Y_STFT[:, closest + pitch_offset]))
        max_val = np.abs(Y_STFT[argmax, closest + pitch_offset])
        freq_indexes = [[argmax, max_val]]  # Pic principal

        # Collecte des pics secondaires au-dessus de 2/3 du max,
        # éloignés d'au moins 10 bins (pour éviter de dupliquer le pic principal)
        for i, ech in enumerate(Y_STFT[:, closest + pitch_offset]):
            if ech > 2 / 3 * max_val and abs(i - argmax) > 10:
                freq_indexes.append([i, ech])

        # Tri des pics par amplitude décroissante
        freq_trie = [i for i, _ in sorted(freq_indexes, key=lambda x: x[1], reverse=True)]

        # Conversion bin FFT → fréquence réelle (Hz)
        freqs = np.array(freq_trie) * SAMPLE_RATE / nfft

        # Axe fréquentiel pour l'affichage du spectre complet
        t = np.arange(len(np.abs(Y_STFT[:, closest + pitch_offset]))) * SAMPLE_RATE / nfft

        # --- Visualisation du spectre pour cet onset ---
        plt.axhline(y=max_val * 2 / 3, color='k', linestyle='--', label='seuil 2/3 du max')
        plt.plot(t, np.abs(Y_STFT[:, closest + pitch_offset]), label='spectre STFT')      # Spectre en amplitude
        for i, x in enumerate(freqs):
            if i == 0:
                plt.axvline(x=x, color='r', linestyle='-', label='fréquences retenues')
            else:
                plt.axvline(x=x, color='r', linestyle='-')                             # Fréquences retenues
        plt.title(f"Spectre STFT au onset {note} ({time:.2f}s)")
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

        print([f"position temporelle : {note/SAMPLE_RATE}, notes trouvées : {[trouver_note(freq) for freq in freqs]}"])

def main(file_name):
    try:
        x, Fe = sf.read(f"sons/{file_name}")
        print("fréquence d'échantillonage : ",Fe)
        
        # Conversion en mono : si le fichier est stéréo (2 canaux),
        # on fait la moyenne des deux pour ne pas les traiter séparément.
        if len(x.shape) == 2:
            x = np.mean(x, axis=1)

        #tracer le signal d'origine
        t=np.linspace(0,len(x)/Fe,len(x))
        plt.close()
        plt.title("signal temporel d'origine non échantilloné")
        plt.plot(t,x)
        plt.xlabel("temps (s)")
        plt.ylabel("amplitude")
        plt.show()        

        #découper le signal en notes et analyser chaque note
        print("notes trouvées : ")
        trouver_pitchs(x)


    except:
        raise SystemError("vérifier le nom du fichier, sinon c'est une erreur dans le decodeur")

main(NOM_FICHIER)
