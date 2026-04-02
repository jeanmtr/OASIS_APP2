# Méthode de détection des notes : 


## 1. Détection des moments ou une note est jouée

Cf video filtrage 1/4:
- on calcule l'enveloppo du signal
- on calcule sa dérivée :
    - pic de la dérivée au dessus d'un certain seuil = note
    - possibilité de différencier batterie et piano grace aux caractéristiques de l'enveloppe détectée

Il faut aussi détecter la fin de la note:
- on pourrait utiliser le 1er retour à 0 de l'enveloppe ou retour à la moyenne du signal global.

A la fin de cette étape on a donc une liste de potentielles notes avec leurs durées.

## 2. Détection de la hauteur de note jouée:

On fait une tfd sur la durée de la note, on trouve la fréquence de plus haute amplitude (la fondamentale) et on a notre note.

Potentiellement en fonction des harmoniques on peut aussi distinguer le type d'instruments (batterie = bcp d'harmonique je crois)


## 3. Pré/Post processing:

Il faudra suement traiter le signal en amont, utiliser du fenétrage etc mais on verra en suite
