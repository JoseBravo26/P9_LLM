"""Tool LangChain SQL : le LLM propose une requete, cette couche la controle avant execution.

Regles de securite appliquees dans validate_sql :
    - seule une requete SELECT unique est acceptee ;
    - les mots-cles de modification/administration sont interdits ;
    - un LIMIT par defaut est ajoute si absent, pour bornancer la volumetrie renvoyee.

Correctif du 18/08-02/09/2026 (accents dans les noms de joueurs) :
    Historique du probleme, en quatre etapes :

    1) `player_name = 'Nikola Jokic'` (egalite stricte). Echec : SQLite
       compare les chaines de maniere stricte, 0 ligne, sans erreur.

    2) `player_name LIKE '%Jokic%'`, motif sans accent, colonne inchangee en
       base. Echec : "Jokic" (ASCII) n'existe pas dans "Jokić" (dont le
       dernier caractere est U+0107, distinct du "c" ASCII).

    3) Fonction SQLite personnalisee SANS_ACCENTS(), appliquee a la colonne
       player_name (avec gestion du prefixe de table) pour les motifs `=` et
       `LIKE` isoles. Fonctionnel pour les questions a un seul joueur.

    4) Extension a la clause SQL `IN (...)`, pour les comparaisons
       multi-joueurs generees avec cette syntaxe.

Correctif du 02/09/2026 (LIMIT tronque les comparaisons multi-joueurs) :
    Meme apres les corrections d'accents ci-dessus, la question "Qui a le
    plus de rebonds entre Nikola Jokic, Karl-Anthony Towns et Giannis
    Antetokounmpo ?" continuait de repondre que "les donnees pour Jokic et
    Antetokounmpo ne sont pas disponibles", alors que les trois joueurs sont
    bien presents en base (verifie individuellement). Cause reelle,
    distincte du probleme d'accents : le LLM genere une requete qui filtre
    correctement sur les 3 joueurs (3 conditions LIKE combinees par OR),
    mais ajoute par reflexe "ORDER BY s.total_rebounds DESC LIMIT 1" -- motif
    adapte a une question du type "qui a le plus de..." mais qui tronque le
    resultat a une seule ligne, empechant toute vraie comparaison entre les
    joueurs cites. Verifie par execution reelle :
        SQL genere : "... AND (LIKE '%Jokic%' OR LIKE '%Towns%' OR
                       LIKE '%Antetokounmpo%') ORDER BY total_rebounds DESC
                       LIMIT 1"
        -> ne renvoie que Karl-Anthony Towns (922 rebonds), le maximum.

    La fonction ajuster_limit_pour_comparaison() detecte le nombre de
    conditions de filtrage sur player_name (comptees apres la transformation
    SANS_ACCENTS) et releve automatiquement le LIMIT a ce nombre s'il est
    inferieur, tout en preservant l'ORDER BY (donc le classement reste
    visible : le joueur avec le plus de rebonds apparait toujours en
    premier, mais les autres joueurs cites ne sont plus tronques). Verifie
    par execution reelle : LIMIT 1 -> LIMIT 3, les 3 joueurs remontent
    correctement classes par rebonds decroissants.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI

from few_shot_sql_examples import FEW_SHOT_SQL
from utils.config import MODEL_NAME, require_mistral_api_key
from utils.database import DEFAULT_DB_PATH, get_connection
from utils.db_schemas import SQLRequest, SQLToolResult

LOGGER = logging.getLogger(__name__)
SCHEMA_TEXT = Path("db/schema.sql").read_text(encoding="utf-8")
FORBIDDEN_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|GRANT|REVOKE)\b|;",
    re.IGNORECASE,
)
DEFAULT_ROW_LIMIT = 20

# Detecte une comparaison sur player_name (egalite, LIKE, ou IN), avec ou
# sans prefixe de table (ex: "p.player_name" ou simplement "player_name").
PLAYER_NAME_IN_PATTERN = re.compile(r"(\w+\.)?player_name\s+IN\s*\(([^)]+)\)", re.IGNORECASE)
PLAYER_NAME_EQUALITY_PATTERN = re.compile(r"(\w+\.)?player_name\s*=\s*'([^']+)'", re.IGNORECASE)
PLAYER_NAME_LIKE_PATTERN = re.compile(r"(\w+\.)?player_name\s+LIKE\s*'([^']+)'", re.IGNORECASE)

# Compte le nombre de conditions de filtrage sur player_name APRES la
# transformation SANS_ACCENTS(...), pour ajuster le LIMIT en consequence.
SANS_ACCENTS_LIKE_PATTERN = re.compile(r"SANS_ACCENTS\([^)]*player_name\)\s+LIKE", re.IGNORECASE)


def _retirer_accents(texte: str) -> str:
    """Retire les diacritiques d'une chaine (ex: 'Jokić' -> 'Jokic').

    Utilise la decomposition Unicode NFKD : chaque caractere accentue est
    decompose en (lettre de base + accent), puis les accents (categorie
    Unicode "combining") sont retires.
    """
    forme_decomposee = unicodedata.normalize("NFKD", texte)
    return "".join(
        caractere for caractere in forme_decomposee if not unicodedata.combining(caractere)
    )


def enregistrer_fonction_sans_accents(connection) -> None:
    """Enregistre la fonction SQL personnalisee SANS_ACCENTS(texte) sur la
    connexion SQLite fournie, afin de pouvoir l'utiliser dans les clauses
    WHERE generees (ex: WHERE SANS_ACCENTS(p.player_name) LIKE '%Jokic%').
    Cette fonction doit etre enregistree sur CHAQUE connexion (elle n'est
    pas persistee dans le fichier .db), ce qui est fait automatiquement
    dans execute_sql() avant chaque execution.
    """

    def _sans_accents_sql(valeur):
        if valeur is None:
            return None
        return _retirer_accents(str(valeur))

    connection.create_function("SANS_ACCENTS", 1, _sans_accents_sql)


def assouplir_comparaison_nom(sql_text: str) -> str:
    """Reecrit toute comparaison sur player_name (=, LIKE, ou IN) pour la
    rendre insensible aux accents, en enveloppant la colonne (avec son
    prefixe de table eventuel) avec SANS_ACCENTS(), et en retirant les
    accents du ou des fragments recherches cote LLM.
    """

    def _remplacer_in(correspondance: re.Match) -> str:
        prefixe_table = correspondance.group(1) or ""
        valeurs_brutes = re.findall(r"'([^']+)'", correspondance.group(2))
        conditions = " OR ".join(
            f"SANS_ACCENTS({prefixe_table}player_name) LIKE '%{_retirer_accents(valeur)}%'"
            for valeur in valeurs_brutes
        )
        return f"({conditions})"

    def _remplacer_egalite(correspondance: re.Match) -> str:
        prefixe_table = correspondance.group(1) or ""
        nom_sans_accent = _retirer_accents(correspondance.group(2))
        return f"SANS_ACCENTS({prefixe_table}player_name) LIKE '%{nom_sans_accent}%'"

    def _remplacer_like(correspondance: re.Match) -> str:
        prefixe_table = correspondance.group(1) or ""
        motif_sans_accent = _retirer_accents(correspondance.group(2))
        return f"SANS_ACCENTS({prefixe_table}player_name) LIKE '{motif_sans_accent}'"

    # L'ordre est important : IN doit etre traite avant EQUALITY, car le
    # motif EQUALITY (player_name = '...') pourrait sinon capturer a tort
    # une sous-partie d'une clause IN deja transformee.
    sql_text = PLAYER_NAME_IN_PATTERN.sub(_remplacer_in, sql_text)
    sql_text = PLAYER_NAME_EQUALITY_PATTERN.sub(_remplacer_egalite, sql_text)
    sql_text = PLAYER_NAME_LIKE_PATTERN.sub(_remplacer_like, sql_text)
    return sql_text


def ajuster_limit_pour_comparaison(sql_text: str) -> str:
    """Releve le LIMIT si la requete filtre sur plusieurs joueurs distincts.

    Si le SQL contient plusieurs conditions SANS_ACCENTS(player_name) LIKE
    (donc plusieurs joueurs cites, typiquement une question de comparaison),
    et que le LIMIT present est strictement inferieur au nombre de joueurs
    filtres, le LIMIT est releve a ce nombre. Cela evite qu'un
    "ORDER BY ... LIMIT 1" ajoute par reflexe par le LLM (adapte a une
    question de classement simple mais pas a une comparaison multi-joueurs)
    ne tronque le resultat a un seul joueur, alors que la question demande
    de voir les valeurs de plusieurs joueurs nommement cites.

    L'ORDER BY est preserve : le classement reste visible (le joueur en tete
    apparait toujours en premier), seuls les joueurs supplementaires cites
    dans la question redeviennent visibles au lieu d'etre tronques.
    """
    nb_conditions_nom = len(SANS_ACCENTS_LIKE_PATTERN.findall(sql_text))
    if nb_conditions_nom <= 1:
        return sql_text

    match_limit = re.search(r"LIMIT\s+(\d+)", sql_text, re.IGNORECASE)
    if match_limit and int(match_limit.group(1)) < nb_conditions_nom:
        return re.sub(r"LIMIT\s+\d+", f"LIMIT {nb_conditions_nom}", sql_text, flags=re.IGNORECASE)
    return sql_text


def validate_sql(sql_text: str) -> str:
    """Refuse toute requete non-SELECT ou multi-instructions, et plafonne le volume."""
    candidate = sql_text.strip().rstrip(";")
    if not candidate.upper().startswith("SELECT"):
        raise ValueError("Seules les requetes SELECT sont autorisees.")
    if FORBIDDEN_PATTERN.search(candidate):
        raise ValueError("Requete refusee : mot-cle interdit ou instructions multiples detectees.")
    if "LIMIT" not in candidate.upper():
        candidate = f"{candidate} LIMIT {DEFAULT_ROW_LIMIT}"
    return candidate


def build_prompt(question: str) -> str:
    """Construit le prompt few-shot envoye au LLM pour generer le SQL."""
    examples_text = "\n".join(
        f"Question: {example['question']}\nSQL: {example['sql']}" for example in FEW_SHOT_SQL
    )
    return (
        "Tu generes uniquement une requete SQLite SELECT en une seule instruction, "
        "sans balise markdown et sans point-virgule final.\n"
        "Si la question porte sur une granularite absente du schema "
        "(match par match, cinq derniers matchs, domicile/exterieur), "
        "reponds exactement : ABSTAIN\n"
        "IMPORTANT : pour filtrer un ou plusieurs noms de joueurs, n'utilise JAMAIS "
        "l'egalite stricte (=). Utilise systematiquement LIKE '%fragment%' (ou une serie "
        "de LIKE combines avec OR/IN) avec un fragment distinctif du nom debarrasse des "
        "accents (ex: pour 'Nikola Jokic' ou 'Nikola Jokić', utilise LIKE '%Jokic%'), car "
        "les noms stockes en base peuvent contenir des accents differents de ceux ecrits "
        "dans la question. Cette comparaison sera automatiquement rendue insensible aux "
        "accents par une couche de securite (y compris pour les clauses IN (...)) : "
        "ecris donc toujours les fragments SANS accent, quelle que soit la question.\n"
        "IMPORTANT : si la question cite plusieurs joueurs nommement (comparaison), "
        "NE JAMAIS ajouter LIMIT 1, meme avec un ORDER BY. Le LIMIT doit toujours etre "
        "au moins egal au nombre de joueurs cites dans la question, afin que chacun "
        "d'eux apparaisse dans le resultat.\n\n"
        f"SCHEMA DE LA BASE:\n{SCHEMA_TEXT}\n\n"
        f"EXEMPLES:\n{examples_text}\n\n"
        f"QUESTION: {question}\nSQL:"
    )


def generate_sql(question: str) -> str:
    """Interroge le LLM Mistral pour obtenir le SQL candidat associe a la question."""
    llm = ChatMistralAI(model=MODEL_NAME, mistral_api_key=require_mistral_api_key(), temperature=0)
    response = llm.invoke(build_prompt(question))
    return response.content.strip()


def execute_sql(question: str, db_path=DEFAULT_DB_PATH) -> SQLToolResult:
    """Genere, valide puis execute le SQL. Retourne toujours un resultat structure."""
    generated_sql = generate_sql(question)
    if generated_sql.upper().startswith("ABSTAIN"):
        return SQLToolResult(
            sql="",
            columns=[],
            rows=[],
            row_count=0,
            error="Les donnees structurees disponibles ne permettent pas cette granularite (saison uniquement).",
        )
    try:
        # Filet de securite 1 : rend toute comparaison sur player_name
        # insensible aux accents (egalite, LIKE, ou IN, avec ou sans prefixe
        # de table, avec ou sans accent dans le motif de recherche).
        sql_assoupli = assouplir_comparaison_nom(generated_sql)
        # Filet de securite 2 : releve le LIMIT si plusieurs joueurs sont
        # cites, pour eviter qu'une comparaison multi-joueurs ne soit
        # tronquee a une seule ligne par un LIMIT trop restrictif.
        sql_avec_limit_ajuste = ajuster_limit_pour_comparaison(sql_assoupli)
        safe_sql = validate_sql(sql_avec_limit_ajuste)
        request = SQLRequest(question=question, sql=safe_sql)
        with get_connection(db_path) as connection:
            # La fonction SANS_ACCENTS() doit etre enregistree sur CHAQUE
            # connexion SQLite avant execution : elle n'est pas persistee
            # dans le fichier .db et disparait a la fermeture de la connexion.
            enregistrer_fonction_sans_accents(connection)
            cursor_rows = connection.execute(request.sql).fetchall()
        rows = [dict(row) for row in cursor_rows]
        columns = list(rows[0].keys()) if rows else []
        return SQLToolResult(sql=request.sql, columns=columns, rows=rows, row_count=len(rows))
    except Exception as exc:
        LOGGER.warning("Echec d execution SQL pour la question '%s' : %s", question, exc)
        return SQLToolResult(sql=generated_sql, columns=[], rows=[], row_count=0, error=str(exc))


@tool
def nba_sql_tool(question: str) -> dict:
    """Interroge les statistiques NBA structurees (players/stats) pour repondre aux questions chiffrees et comparatives."""
    return execute_sql(question).model_dump()
