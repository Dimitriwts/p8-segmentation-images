"""Application web de demonstration du modele de segmentation.

Elle ne contient aucun modele : elle interroge l'API par le reseau, exactement
comme le ferait le systeme de decision en aval. Si l'affichage est correct, c'est
que l'API fonctionne pour un client exterieur, et pas seulement que le modele
tourne sur la machine de developpement.

Lancement en local :
    python app.py

L'adresse de l'API se regle par la variable d'environnement URL_API.
"""

import os

import requests
from flask import Flask, Response, abort, render_template, request

app = Flask(__name__)

# En local l'API tourne a cote ; une fois deployee, on passe son adresse par
# variable d'environnement plutot que de la coder en dur.
URL_API = os.environ.get("URL_API", "http://127.0.0.1:8000").rstrip("/")

# temps d'attente maximal : la premiere prediction reveille parfois un serveur
# endormi sur les hebergements gratuits, ce qui peut prendre une minute
DELAI = int(os.environ.get("DELAI_API", "90"))

# les couleurs doivent correspondre a la palette de l'API, cote modele
CATEGORIES = [
    ("void", "#000000", "non annote, capot du vehicule, bords de l'image"),
    ("flat", "#804080", "route, trottoir, parking, voies ferrees"),
    ("construction", "#464646", "batiments, murs, barrieres, ponts, tunnels"),
    ("object", "#faaa1e", "poteaux, feux de circulation, panneaux"),
    ("nature", "#6b8e23", "vegetation, terrain"),
    ("sky", "#4682b4", "ciel"),
    ("human", "#dc143c", "pietons, cyclistes"),
    ("vehicle", "#00008e", "voitures, camions, bus, trains, deux-roues"),
]


def appeler_api(chemin, **kwargs):
    """Appelle l'API et transmet une erreur lisible si elle ne repond pas."""
    try:
        reponse = requests.get(f"{URL_API}{chemin}", timeout=DELAI, **kwargs)
        reponse.raise_for_status()
        return reponse
    except requests.exceptions.ConnectionError:
        abort(503, f"l'API ne repond pas a l'adresse {URL_API}")
    except requests.exceptions.Timeout:
        abort(504, f"l'API n'a pas repondu en {DELAI} secondes")
    except requests.exceptions.HTTPError as e:
        abort(e.response.status_code, f"l'API a renvoye une erreur : {e}")


@app.route("/")
def accueil():
    """Page principale : le choix de l'image et l'affichage des trois vues."""
    try:
        identifiants = appeler_api("/images").json()["identifiants"]
        info = appeler_api("/info").json()
        erreur = None
    except Exception as e:
        identifiants, info, erreur = [], {}, str(e)

    choisi = request.args.get("id") or (identifiants[0] if identifiants else None)

    return render_template(
        "index.html",
        identifiants=identifiants,
        choisi=choisi,
        info=info,
        categories=CATEGORIES,
        erreur=erreur,
        url_api=URL_API,
    )


@app.route("/vue/image/<identifiant>")
def vue_image(identifiant):
    """Relais vers l'image d'origine servie par l'API."""
    return Response(appeler_api(f"/images/{identifiant}").content, mimetype="image/png")


@app.route("/vue/masque/<identifiant>")
def vue_masque(identifiant):
    """Relais vers le masque annote a la main, la verite terrain."""
    r = appeler_api(f"/masques/{identifiant}", params={"colorise": "true"})
    return Response(r.content, mimetype="image/png")


@app.route("/vue/prediction/<identifiant>")
def vue_prediction(identifiant):
    """Recupere l'image aupres de l'API, la lui renvoie pour prediction, et
    transmet le masque predit.

    Ce trajet peut sembler indirect, mais il reproduit exactement l'usage reel :
    un client qui possede une image et demande sa segmentation.
    """
    image = appeler_api(f"/images/{identifiant}").content

    try:
        reponse = requests.post(
            f"{URL_API}/predire",
            params={"colorise": "true"},
            files={"fichier": (f"{identifiant}.png", image, "image/png")},
            timeout=DELAI,
        )
        reponse.raise_for_status()
    except requests.exceptions.RequestException as e:
        abort(503, f"la prediction a echoue : {e}")

    return Response(
        reponse.content,
        mimetype="image/png",
        headers={"X-Duree-Ms": reponse.headers.get("X-Duree-Ms", "")},
    )


@app.route("/vue/proportions/<identifiant>")
def vue_proportions(identifiant):
    """Les proportions de chaque categorie dans l'image, telles que vues par le modele."""
    image = appeler_api(f"/images/{identifiant}").content
    try:
        reponse = requests.post(
            f"{URL_API}/predire/proportions",
            files={"fichier": (f"{identifiant}.png", image, "image/png")},
            timeout=DELAI,
        )
        reponse.raise_for_status()
    except requests.exceptions.RequestException as e:
        abort(503, f"la prediction a echoue : {e}")
    return reponse.json()


if __name__ == "__main__":
    # host 0.0.0.0 pour que l'application soit joignable une fois deployee
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
