"""Obtient un jeton pCloud SANS jamais transmettre le mot de passe.

Motif (Essi, 2026-08-26 : « mais mon mot de passe ira en clair... »). La methode
evidente, `userinfo?getauth=1&username=...&password=...`, chiffre bien le mot de passe
en transit par TLS, mais le laisse dans la barre d'adresse, dans l'historique du
navigateur et, tres probablement, dans les journaux d'acces du serveur, ou les chaines
de requete sont couramment enregistrees. Ce n'est pas acceptable pour un mot de passe
principal. L'enregistrement d'une application OAuth2, qui reglerait la question, est
soumis a une revue MANUELLE de pCloud et reste en attente.

La methode par condense evite les deux problemes : le serveur fournit un `digest` a
usage unique, le client calcule sha1(mot_de_passe + sha1(courriel) + digest), et
n'envoie que ce hache. Le mot de passe ne quitte jamais la machine.

    python .runs/quebec/jeton_pcloud.py

A lancer depuis un VRAI terminal : par l'invite de Claude Code, l'entree standard n'est
pas un terminal et la saisie masquee echoue sur une fin de fichier.
"""
import getpass
import hashlib
import json
import ssl
import sys
import urllib.parse
import urllib.request

# MAGASIN DE CERTIFICATS DU SYSTEME. Sur un poste ministeriel, le trafic TLS est
# inspecte et resigne par un proxy dont l'autorite racine est installee dans Windows
# mais inconnue du paquet de certificats embarque par Python : d'ou un
# CERTIFICATE_VERIFY_FAILED qui ressemble a un refus d'identifiants alors qu'aucune
# requete n'a abouti (constate le 2026-08-26). `truststore` delegue la verification au
# systeme : on accepte l'autorite de l'entreprise SANS cesser de verifier, ce qui
# importe sur un echange qui porte un mot de passe.
try:
    import truststore
    truststore.inject_into_ssl()
    _CTX = None
except ImportError:
    _CTX = None
    try:
        import certifi
        _CTX = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass

# Un compte pCloud vit soit aux Etats-Unis soit en Europe ; interroger le mauvais hote
# renvoie une erreur trompeuse. Celui d'Essi est americain (2026-08-26).
HOTES = ("api.pcloud.com", "eapi.pcloud.com")


def appeler(hote, methode, **params):
    url = f"https://{hote}/{methode}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30, context=_CTX) as r:
        return json.loads(r.read().decode())


def identifier(hote, courriel, mot_de_passe):
    """Identification par condense. Rend la reponse brute : soit un jeton, soit une
    demande de second facteur. `login` plutot que `userinfo`, car c'est l'entree
    documentee pour le second facteur et elle rend le meme champ `auth` sinon."""
    digest = appeler(hote, "getdigest")["digest"]
    h_courriel = hashlib.sha1(courriel.lower().encode()).hexdigest()
    hache = hashlib.sha1((mot_de_passe + h_courriel + digest).encode()).hexdigest()
    return appeler(hote, "login", getauth=1, logout=0, username=courriel,
                   digest=digest, passworddigest=hache)


def second_facteur(hote, rep):
    """Confirme le second facteur. pCloud range le jeton intermediaire sous un nom qui
    a varie selon les versions de l'API : on essaie les candidats connus et, a defaut,
    on MONTRE la reponse au lieu de deviner (le champ `token` etait vide au premier
    essai, 2026-08-26)."""
    tok = next((rep[k] for k in ("token", "authtoken", "tfatoken") if rep.get(k)), None)
    if not tok:
        champs = ", ".join(f"{k}={v!r}" for k, v in rep.items() if k != "result")
        print(f"  reponse pCloud : {champs}", file=sys.stderr)
        return None
    quoi = {1: "application d'authentification", 2: "message texte",
            4: "courriel"}.get(rep.get("tfatype"), "second facteur")
    if rep.get("tfatype") in (2, 4):
        try:
            appeler(hote, "tfa_sendcode", token=tok)
            print(f"  code envoye par {quoi}")
        except Exception as exc:
            print(f"  envoi du code impossible ({exc})", file=sys.stderr)
    code = input(f"code de verification ({quoi}) : ").strip()
    # trustdevice=0 : ce jeton servira sur une machine louee et ephemere, on ne lui
    # accorde pas le statut d'appareil de confiance.
    return appeler(hote, "login", getauth=1, logout=0, authtoken=tok,
                   code=code, trustdevice=0)


def afficher(hote, auth):
    print(f"\nhote  : {hote}")
    print(f"jeton : {auth}")
    print("\nA utiliser tel quel, sans le ranger dans un fichier du depot :")
    print(f"  PCLOUD_TOKEN='{auth}' PCLOUD_HOST={hote} \\")
    print("    PCLOUD_DIR=/meandre bash .runs/quebec/amorcer_pod.sh")
    print("\nRevocation : fermer les sessions actives dans les reglages pCloud.")


def main():
    courriel = input("courriel pCloud : ").strip()
    mdp = getpass.getpass("mot de passe (invisible) : ")
    for hote in HOTES:
        try:
            rep = identifier(hote, courriel, mdp)
        except Exception as exc:
            print(f"  {hote} : injoignable ({exc})", file=sys.stderr)
            if "CERTIFICATE_VERIFY_FAILED" in str(exc):
                print("     -> proxy d'inspection TLS ; installer truststore :"
                      "  uv add truststore", file=sys.stderr)
            continue
        if rep.get("result") == 0 and rep.get("auth"):
            afficher(hote, rep["auth"])
            return 0
        if rep.get("result") in (2297, 2306):
            print(f"\n{hote} : double authentification active.")
            rep2 = second_facteur(hote, rep)
            if rep2 and rep2.get("result") == 0 and rep2.get("auth"):
                afficher(hote, rep2["auth"])
                return 0
            if rep2:
                print(f"  refuse : {rep2.get('error', rep2.get('result'))}",
                      file=sys.stderr)
            return 2
        print(f"  {hote} : refuse ({rep.get('error', rep.get('result'))})",
              file=sys.stderr)
    print("\nAucun hote n'a accepte ces identifiants.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
