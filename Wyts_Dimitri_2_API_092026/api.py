"""API de segmentation d'images pour Future Vision Transport.

Recoit une image de camera embarquee, renvoie le masque de segmentation : une
image de meme taille ou chaque pixel porte le numero de la categorie a laquelle
il appartient, parmi les 8 categories principales de Cityscapes.

Lancement en local :
    uvicorn api:app

La documentation interactive est ensuite sur http://127.0.0.1:8000/docs
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response

import modele as m

# ------------------------------------------------------------------ chargement
RACINE = Path(__file__).parent
CONFIG = json.loads((RACINE / "config.json").read_text(encoding="utf-8"))
EXEMPLES = RACINE / "exemples"

app = FastAPI(
    title="Segmentation d'images embarquees",
    description=(
        "Segmentation semantique d'images de rue en 8 categories, pour le systeme "
        "de vision de Future Vision Transport. Envoyez une image, recevez son masque."
    ),
    version="1.0",
)

# Le modele est charge une seule fois au demarrage. Le recharger a chaque
# requete couterait du temps a chaque appel pour rien.
print(f"chargement du modele {CONFIG['architecture']} ...", flush=True)
_t0 = time.perf_counter()
MODELE = m.charger(RACINE / CONFIG["modele"])
print(f"modele charge en {time.perf_counter() - _t0:.1f}s", flush=True)

PALETTE = np.array(m.PALETTE, dtype=np.uint8)


# ------------------------------------------------------------------ utilitaires
def predire_masque(image_rgb):
    """Segmente une image et rend le masque a la taille de l'image d'entree.

    Le modele travaille sur des images de 512x256. On redimensionne donc en
    entree, puis on ramene le masque a la taille d'origine pour que l'appelant
    recoive un masque directement superposable a son image.

    Le retour a la taille d'origine se fait au plus proche voisin : interpoler
    des numeros de categorie creerait des categories qui n'existent pas.
    """
    hauteur, largeur = image_rgb.shape[:2]

    petite = cv2.resize(image_rgb, (m.LARGEUR, m.HAUTEUR), interpolation=cv2.INTER_AREA)
    lot = petite.astype(np.float32)[None] / 255.0     # un lot d'une seule image, en 0-1

    probabilites = m.predire_probabilites(MODELE, lot)[0]
    masque = probabilites.argmax(axis=-1).astype(np.uint8)

    return cv2.resize(masque, (largeur, hauteur), interpolation=cv2.INTER_NEAREST)


def png(tableau):
    """Encode un tableau numpy en PNG et le rend sous forme d'octets."""
    if tableau.ndim == 3:
        tableau = cv2.cvtColor(tableau, cv2.COLOR_RGB2BGR)   # OpenCV ecrit en BGR
    ok, tampon = cv2.imencode(".png", tableau)
    if not ok:
        raise HTTPException(500, "echec de l'encodage PNG")
    return tampon.tobytes()


def lire_exemple(dossier, identifiant, suffixe):
    chemin = EXEMPLES / dossier / f"{identifiant}{suffixe}"
    if not chemin.exists():
        raise HTTPException(404, f"image inconnue : {identifiant}")
    return chemin


def lire_image_envoyee(octets):
    """Decode une image recue, ou refuse proprement si elle est illisible."""
    tableau = cv2.imdecode(np.frombuffer(octets, np.uint8), cv2.IMREAD_COLOR)
    if tableau is None:
        raise HTTPException(400, "fichier illisible : envoyez une image PNG ou JPEG")
    return cv2.cvtColor(tableau, cv2.COLOR_BGR2RGB)


# -------------------------------------------------------------------- routes
@app.get("/", include_in_schema=False)
def accueil():
    return RedirectResponse("/docs")


@app.get("/info", summary="Informations sur le modele servi")
def info():
    """Ce que l'API sert, et dans quel format."""
    return {
        "architecture": CONFIG["architecture"],
        "miou_test": CONFIG.get("miou_test"),
        "taille_traitee": {"largeur": m.LARGEUR, "hauteur": m.HAUTEUR},
        "categories": {i: nom for i, nom in enumerate(m.CATEGORIES)},
        # ONNX ne garde pas ce compte : il est ecrit dans config.json au moment de l'export
        "parametres": CONFIG.get("parametres"),
    }


@app.get("/images", summary="Liste des images de demonstration")
def liste_images():
    """Les identifiants des images embarquees pour la demonstration.

    Ce sont des images du jeu de test, que le modele n'a jamais vues pendant
    son entrainement.
    """
    ids = sorted(p.stem.replace("_leftImg8bit", "")
                 for p in (EXEMPLES / "images").glob("*.png"))
    return {"nombre": len(ids), "identifiants": ids}


@app.get("/images/{identifiant}", summary="Une image de demonstration")
def image(identifiant: str):
    chemin = lire_exemple("images", identifiant, "_leftImg8bit.png")
    return Response(chemin.read_bytes(), media_type="image/png")


@app.get("/masques/{identifiant}", summary="Le masque reel d'une image de demonstration")
def masque_reel(identifiant: str,
                colorise: bool = Query(True, description="rendre le masque en couleurs")):
    """La verite terrain annotee a la main, pour comparer avec la prediction."""
    chemin = lire_exemple("masques", identifiant, "_gtFine_labelIds.png")
    masque = cv2.imread(str(chemin), cv2.IMREAD_GRAYSCALE)
    return Response(png(PALETTE[masque] if colorise else masque), media_type="image/png")


@app.post("/predire", summary="Segmenter une image")
async def predire(
    fichier: UploadFile = File(..., description="image de camera embarquee (PNG ou JPEG)"),
    colorise: bool = Query(False, description="masque en couleurs plutot qu'en numeros de categorie"),
):
    """Segmente l'image envoyee et rend le masque predit.

    Par defaut le masque est un PNG en niveaux de gris ou la valeur de chaque
    pixel est le numero de sa categorie, de 0 a 7. C'est le format exploitable
    directement par le systeme de decision en aval.

    Avec colorise=true, le masque est rendu en couleurs, pour l'affichage humain.
    """
    image_rgb = lire_image_envoyee(await fichier.read())

    t0 = time.perf_counter()
    masque = predire_masque(image_rgb)
    duree = time.perf_counter() - t0

    sortie = PALETTE[masque] if colorise else masque
    return Response(
        png(sortie),
        media_type="image/png",
        headers={
            "X-Duree-Ms": f"{duree * 1000:.0f}",
            "X-Taille": f"{masque.shape[1]}x{masque.shape[0]}",
        },
    )


@app.post("/predire/proportions", summary="Segmenter et rendre les proportions par categorie")
async def predire_proportions(fichier: UploadFile = File(...)):
    """Meme prediction, mais rend la part de l'image occupee par chaque categorie.

    Utile au systeme de decision pour une lecture rapide de la scene sans avoir
    a analyser le masque pixel par pixel.
    """
    image_rgb = lire_image_envoyee(await fichier.read())
    masque = predire_masque(image_rgb)
    comptes = np.bincount(masque.ravel(), minlength=m.NB_CLASSES)

    return JSONResponse({
        "taille": {"largeur": int(masque.shape[1]), "hauteur": int(masque.shape[0])},
        "proportions": {
            nom: round(float(c) / masque.size, 4)
            for nom, c in zip(m.CATEGORIES, comptes)
        },
    })
