# Etape 2 -- Integration Excel/PDF, base relationnelle et SQL Tool

## Objectif
Ajouter une voie de reponse deterministe pour les questions chiffrees (statistiques
joueurs/equipes) en complement du RAG documentaire deja en place pour les rapports
narratifs. La base relationnelle reste volontairement simple : `players`, `matches`,
`stats`, `reports`.

## Installation
Les dependances principales (`pandas`, `PyPDF2`, `langchain-mistralai`, `pydantic`)
sont deja presentes dans `requirements.txt` du projet. Ajouter si besoin le contenu
de `requirements_etape_2.txt` :
```bash
pip install -r requirements.txt
```

## Modele de donnees
- `players` : referentiel joueur/equipe, alimente par le classeur Excel.
- `stats` : une ligne par joueur et par saison (`granularity = 'season'`,
  `match_id = NULL`). Correspond exactement aux colonnes disponibles dans
  `regular NBA.xlsx` (points, rebonds, passes, pourcentages de tir, ratings...).
- `matches` : prete pour des matchs identifies avec certitude dans un rapport PDF.
  Aucune ligne n est creee automatiquement si la date, l equipe a domicile ou le score
  ne sont pas clairement identifiables dans le texte source.
- `reports` : contenu integral de chaque PDF de commentaires, avec fichier source et
  nombre de pages, relie a `matches` uniquement si un match a ete identifie.

Le schema SQL complet est dans `db/schema.sql`, avec contraintes `CHECK`, cles
etrangeres et index sur les colonnes de jointure et de filtrage frequentes.

## Ingestion
```bash
python load_excel_to_db.py --excel "inputs/regular NBA.xlsx" --reports-dir inputs --season 2024-2025
```
Etapes realisees :
1. Lecture du classeur avec pandas, renommage des colonnes vers le schema metier.
2. Validation ligne par ligne avec `SeasonStatRow` (Pydantic) : bornes de valeurs,
   coherence tirs reussis/tentes, age plausible.
3. Insertion en base via upsert sur `players` puis `stats`.
4. Extraction du texte de chaque PDF du dossier `--reports-dir` et insertion dans
   `reports` (upsert par `source_file`).
5. Journalisation du nombre de lignes acceptees et rejetees.

## SQL Tool
`utils/sql_tool.py` expose :
- `nba_sql_tool` : outil LangChain (`@tool`) utilisable par un agent.
- `execute_sql(question)` : fonction testable independamment de LangChain.

Fonctionnement :
1. Le LLM recoit le schema de la base et des exemples few-shot (`few_shot_sql_examples.py`).
2. Il genere une requete SQLite ou repond `ABSTAIN` si la granularite demandee est absente.
3. `validate_sql` bloque toute requete qui n est pas un `SELECT` unique, refuse les
   mots-cles de modification/administration, et ajoute `LIMIT 20` par defaut.
4. La requete validee est executee sur `database/nba_analytics.db` et le resultat est
   retourne sous forme structuree (`SQLToolResult`), y compris en cas d erreur ou d abstention.

## Integration dans l agent
`utils/rag_pipeline_router.py` ajoute un routage avant reponse :
- `route_question(question)` classe la question en `SQL` ou `RAG` via un agent Pydantic AI dedie.
- `answer_with_sql(question)` appelle le SQL Tool puis fait synthetiser le resultat par un
  second agent Pydantic AI, avec abstention automatique si le Tool renvoie une erreur.
- Les questions narratives continuent d emprunter le pipeline RAG existant
  (`utils/rag_pipeline.py`), sans modification de celui-ci.

## Exemples de requetes types
Voir `db/queries_examples.sql` pour :
- Top 10 des pourcentages a trois points avec seuil de tentatives.
- Comparaison de rebonds entre plusieurs joueurs nommes.
- Agregation multicritere (points/rebonds/passes par match).
- Moyenne de net rating par equipe.
- Jointure `reports` / `matches` pour les rapports lies a un match identifie.

## Limite assumee et documentee
Le classeur Excel fourni est agrege par saison et ne contient ni date de match, ni
statut domicile/exterieur, ni decoupage sur les cinq derniers matchs. Ces questions
declenchent une abstention explicite du SQL Tool plutot qu un calcul errone. Si un
futur export match par match est fourni, il s inserera dans `stats` avec
`granularity = 'match'` et un `match_id` renseigne, sans changer le schema.

## Tests
```bash
pytest -q tests/test_sql_tool.py tests/test_ingestion.py
```
