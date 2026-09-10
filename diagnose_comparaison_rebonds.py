"""
Script de diagnostic (etape 8) : la question "comparaison_rebonds" continue
d'echouer pour Jokic et Antetokounmpo malgre le correctif de la clause IN
dans sql_tool.py. Ce script affiche le SQL BRUT genere par le LLM (avant
assouplir_comparaison_nom) et le SQL APRES transformation, pour voir
exactement quelle syntaxe le LLM a produite cette fois-ci et pourquoi elle
echappe aux trois motifs regex actuels (IN, =, LIKE).

Usage :
    python diagnose_comparaison_rebonds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.sql_tool import assouplir_comparaison_nom, generate_sql, enregistrer_fonction_sans_accents  # noqa: E402
from utils.database import DEFAULT_DB_PATH, get_connection  # noqa: E402

QUESTION = "Qui a le plus de rebonds entre Nikola Jokić, Karl-Anthony Towns et Giannis Antetokounmpo ?"


def main() -> None:
    print(f"Question : {QUESTION!r}\n")

    sql_brut = generate_sql(QUESTION)
    print("=== SQL BRUT genere par le LLM ===")
    print(sql_brut)

    sql_transforme = assouplir_comparaison_nom(sql_brut)
    print("\n=== SQL APRES assouplir_comparaison_nom() ===")
    print(sql_transforme)

    print("\n=== Execution du SQL transforme ===")
    try:
        with get_connection(DEFAULT_DB_PATH) as connection:
            enregistrer_fonction_sans_accents(connection)
            candidate = sql_transforme.strip().rstrip(";")
            if "LIMIT" not in candidate.upper():
                candidate = f"{candidate} LIMIT 20"
            lignes = connection.execute(candidate).fetchall()
            print(f"{len(lignes)} ligne(s) : {[dict(r) for r in lignes]}")
    except Exception as exc:
        print(f"Erreur d'execution : {exc}")

    print("\n=== Verification directe : les 3 joueurs existent-ils bien en base ? ===")
    with get_connection(DEFAULT_DB_PATH) as connection:
        enregistrer_fonction_sans_accents(connection)
        for fragment in ["Jokic", "Towns", "Antetokounmpo"]:
            res = connection.execute(
                "SELECT p.player_name, s.total_rebounds FROM stats s "
                "JOIN players p ON p.player_id = s.player_id "
                "WHERE SANS_ACCENTS(p.player_name) LIKE ?",
                (f"%{fragment}%",),
            ).fetchall()
            print(f"  {fragment!r} -> {[dict(r) for r in res]}")


if __name__ == "__main__":
    main()
