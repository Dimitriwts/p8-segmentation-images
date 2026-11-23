"""Chargement du modele de segmentation, cote service.

Le modele a ete entraine avec Keras, puis exporte au format ONNX a la fin du
notebook. ONNX est un format standard qui ne decrit que les calculs d'un reseau
deja entraine, sans tout l'outillage d'entrainement. On le fait tourner avec ONNX
Runtime, une bibliotheque concue uniquement pour predire.

Pourquoi ce detour : avec Keras et PyTorch, l'API demandait trop de memoire pour
les hebergeurs gratuits. Avec ONNX Runtime, elle tient largement. La derniere
section du notebook mesure les deux, et verifie que les masques produits sont les
memes que ceux de Keras.
"""

import os

import onnxruntime as ort

# doit correspondre a ce qui a servi a l'entrainement
HAUTEUR, LARGEUR = 256, 512
NB_CLASSES = 8

CATEGORIES = ["void", "flat", "construction", "object", "nature", "sky", "human", "vehicle"]

# une couleur par categorie, reprise de la palette officielle Cityscapes
PALETTE = [
    (0, 0, 0),        # void          noir
    (128, 64, 128),   # flat          violet (couleur de la route)
    (70, 70, 70),     # construction  gris   (couleur des batiments)
    (250, 170, 30),   # object        orange (couleur des feux)
    (107, 142, 35),   # nature        vert   (couleur de la vegetation)
    (70, 130, 180),   # sky           bleu ciel
    (220, 20, 60),    # human         rouge  (couleur des pietons)
    (0, 0, 142),      # vehicle       bleu fonce (couleur des voitures)
]


def charger(chemin_onnx):
    """Ouvre le modele ONNX et rend une session prete a predire."""
    options = ort.SessionOptions()

    # Nombre de coeurs utilises pour un calcul. Sur un hebergement gratuit on ne
    # dispose que d'une fraction de processeur : demander plusieurs coeurs ne
    # ferait que les mettre en concurrence. Reglable par variable d'environnement.
    options.intra_op_num_threads = int(os.environ.get("NB_COEURS", "1"))
    options.inter_op_num_threads = 1

    # ONNX Runtime garde par defaut une reserve de memoire pour aller plus vite.
    # La desactiver divise encore la memoire occupee, pour un ralentissement tres
    # leger (mesures dans la section 13 du notebook) : sur un hebergement limite
    # a 512 Mo, c'est le bon compromis.
    options.enable_cpu_mem_arena = False

    return ort.InferenceSession(str(chemin_onnx), options, providers=["CPUExecutionProvider"])


def predire_probabilites(session, lot):
    """Probabilites de chaque categorie pour chaque pixel.

    lot : tableau (n, 256, 512, 3) en valeurs 0-1. Le pretraitement propre a
    l'encodeur est inclus dans le modele exporte, rien d'autre a faire ici.
    """
    nom_entree = session.get_inputs()[0].name
    return session.run(None, {nom_entree: lot})[0]
