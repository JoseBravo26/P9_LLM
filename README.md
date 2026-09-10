# NBA Analyst AI — Assistant RAG + SQL pour l'analyse de performance

Assistant intelligent combinant :
- **RAG** (Retrieval-Augmented Generation) sur des rapports PDF de matchs NBA
- **SQL Tool** déterministe sur une base de statistiques saisonnières
- **Routeur SQL/RAG** qui oriente automatiquement chaque question vers la source appropriée

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/JoseBravo26/P9_LLM.git
cd P9_LLM
```

### 2. Créer et activer l'environnement virtuel

**Windows (Git Bash) :**
```bash
python -m venv venv
source venv/Scripts/activate
```

**macOS / Linux :**
```bash
python3 -m venv venv
source venv/bin/activate
```

> **Important (Windows)** : Si `pip install` installe dans `AppData\Roaming` au lieu du `venv`, désactivez le mode « user » par défaut :
> ```bash
> pip config set global.user false
> ```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API

Créez un fichier `.env` à la racine (à·partir de `.env.example` si présent) :

```bash
cp .env.example .env
```

Puis éditez `.env` et renseignez :

```
MISTRAL_API_KEY=votre_clé_ici
```

Optionnel : configurez Logfire pour visualiser les traces si vous utilisez cet outil.

### Activer SSL verification si proxy present
Dans votre fichier .env à la racine du projet :

MISTRAL_API_KEY=votre_clé
DISABLE_SSL_VERIFICATION=1

ou

Sur votre ordi personnel (réseau normal)
Dans votre fichier .env :

MISTRAL_API_KEY=votre_clé
DISABLE_SSL_VERIFICATION=0
ou simplement ne pas mettre la ligne (la valeur par défaut est 0).
---

## Exécution

### Lancer les tests

```bash
pytest -q
```

**Note (Windows)** : Si `pytest` n'est pas trouvé malgré l'activation du venv, utilisez :
```bash
python -m pytest -q
```

### Ingérer les données (Excel + PDF)

```bash
python load_excel_to_db.py --excel "inputs/regular NBA.xlsx" --reports-dir inputs --season 2024-2025
```

Étapes réalisées :
1. Lecture du classeur Excel avec pandas
2. Validation ligne par ligne avec Pydantic (`SeasonStatRow`)
3. Insertion en base (`players`, `stats`)
4. Extraction du texte des PDF et insertion dans `reports`



### Indexer les documents pour le RAG

```bash
python indexer.py
```

### Évaluer la qualité du RAG (RAGAS)

```bash
python evaluate_ragas.py
```

Les résultats sont écrits dans :
- `reports/ragas_results.csv` (detailed)
- `reports/ragas_results.md` (synthèse)

### Lancer l'interface Streamlit

```bash
streamlit run MistralChat.py
```

L'interface affiche :
- Un badge indiquant la source utilisée : 📊 **Statistiques** (SQL) ou 📄 **Analyse documentaire** (RAG)
- Les identifiants de chunks cités si le RAG a été utilisé
- Un message d'abstention explicite si la question ne peut pas être traitée (granularité indisponible)

---

## Architecture technique

### Modèle de données

Base SQLite (`database/nba_analytics.db`) :

| Table | Description |
|-------|-------------|
| `players` | Référentiel joueur/équipe (alimenté par Excel) |
| `stats` | Statistiques par joueur et par saison (`granularity='season'`) |
| `matches` | Prête pour des matchs identifiés avec certitude dans les PDF |
| `reports` | Contenu intégraal de chaque PDF, relié à `matches` si possible |

Le schéma SQL complet est dans `db/schema.sql`, avec contraintes `CHECK`, clés étrangères et index.

### SQL Tool

`utils/sql_tool.py` expose :
- `nba_sql_tool` : outil LangChain (`@tool`) utilisable par un agent
- `execute_sql(question)` : fonction testable indépendamment de LangChain

Fonctionnement :
1. Le LLM reçoit le schéma de la base et des exemples few-shot (`few_shot_sql_examples.py`)
2. Il génére une requête SQLite ou répond `ABSTAIN` si la granularité demandée est absente
3. `validate_sql` bloque toute requête qui n'est pas un `SELECT` unique, refuse les mots-clés de modification/administration, et ajoute `LIMIT 20` par déaut
4. La requête validée est exécutée sur `database/nba_analytics.db`

### Routeur SQL/RAG

`utils/rag_pipeline_router.py` ajoute un routage avant réponsé :

- `route_question(question)` : classe la question en `SQL` ou `RAG` via un agent Pydantic AI dédie
- `answer_with_sql(question)` : appelle le SQL Tool puis fait synthétiser le réultat par un second agent Pydantic AI, avec abstention automatique si le Tool renvoie une erreur
- `answer_with_rag(question)` : délégue au pipeline RAG existant (`utils/rag_pipeline.py`)

En cas d'erreur de classification (réseau, quota, etc.), le routeur se replie automatiquement sur la branche RAG.

```
Question utilisateur
        |
        v
route_question(question)   -> agent Pydantic AI dédie, sortie "SQL" ou "RAG"
        |
   +----+----+
   |         |
  SQL       RAG
   |         |
answer_with_sql   answer_with_rag
   |         |
execute_sql()   answer_question() [utils/rag_pipeline.py existant]
   |         |
synthèse LLM   réponsé structuree AssistantAnswer
   |         |
   +----+----+
        |
        v
AssistantAnswer affichée dans Streamlit
```

### Choix techniques

- **Pydantic** impose un contrat sur les documents, chunks, requêtes, réultats et jeu d'évaluation
- **Pydantic AI** force une réponsé LLM structuree : texte, chunks cite, niveau de confiance et abstention
- **Logfire** trace la requête, le nombre de chunks et l'appel Pydantic AI
- **RAGAS** mesure `faithfulness`, `response_relevancy`, `context_precision` et `context_recall`
- Aucun vecteur nul n'est crée si un embedding échoue : le processus s'arrête pour ne jamais dégrader l'index silencieusement

---

## Tests

```bash
# Tous les tests
pytest -q

# Tests du SQL Tool et de l'ingestion
pytest -q tests/test_sql_tool.py tests/test_ingestion.py

# Tests du routeur
pytest -q tests/test_router.py
```

Les tests du routeur (`tests/test_router.py`) vérifient :
- L'abstention en cas d'erreur SQL
- Le repli sur RAG en cas d'échec du classifieur
- Sans appel réseau réel (mocks)

---

## Exemples de questions

| Question | Route attendue | Véification |
|----------|----------------|-------------|
| Quel est le % à 3 points de Nikola Jokic ? | SQL | Badge "📊 Statistiques", réponsé chiffrée exacte |
| Compare les rebonds de Jokic et Towns | SQL | Badge SQL, tableau de valeurs synthétisé |
| Que disent les rapports sur la déense de Denver ? | RAG | Badge "📄 Analyse documentaire", chunks cite |
| Meilleur 3P% sur les 5 derniers matchs ? | SQL | Abstention explicite (message `st.info`) |
| Compare les rebonds à domicile et à l'extérieur | SQL | Abstention explicite (message `st.info`) |

---

## Limites documentées

### Granularité des données

Le classeur Excel fourni est agréé par saison et ne contient ni date de match, ni statut domicile/extérieur, ni déoupage sur les cinq derniers matchs. Ces questions déclenchent une abstention explicite du SQL Tool plutôt qu'un calcul erroné.

Si un futur export match par match est fourni, il s'insérera dans `stats` avec `granularity='match'` et un `match_id` renseigné, sans changer le schéma.

### Classifieur SQL/RAG

Le classifieur est un agent LLM zero-shot sur des libellés simples ; il peut occasionnellement mal router une question ambigue (ex. une question mixte chiffrée et narrative). L'évaluation continue (étape 3) devra inclure des cas de test spéifiques au routage, en plus des catégories déà définies dans `data/eval_dataset.jsonl`.

---

## Fichiers principaux

| Fichier | Rô·le |
|---------|--------|
| `requirements.txt` | Dépendances Python |
| `pyproject.toml` | Configuration Pytest (`pythonpath`, `testpaths`) |
| `utils/__init__.py` | Rend `utils` importable comme package |
| `utils/rag_pipeline_router.py` | Routeur SQL/RAG + fonction `answer(question)` |
| `utils/sql_tool.py` | SQL Tool (génération + validation + exécution) |
| `utils/rag_pipeline.py` | Pipeline RAG existant (FAISS + Mistral) |
| `utils/vector_store.py` | Gestion de l'index FAISS |
| `utils/db_schemas.py` | Schémas Pydantic pour les tables SQL |
| `utils/schemas.py` | Schémas Pydantic pour RAG (`RAGQuery`, `AssistantAnswer`, etc.) |
| `load_excel_to_db.py` | Script d'ingestion Excel + PDF vers SQLite |
| `indexer.py` | Indexation des documents pour le RAG |
| `evaluate_ragas.py` | Évaluation de la qualité du RAG avec RAGAS |
| `MistralChat.py` | Interface Streamlit |
| `tests/` | Tests unitaires (ingestion, SQL Tool, routeur, schémas) |
| `db/schema.sql` | Schéma SQL complet de la base |
| `db/queries_examples.sql` | Exemples de requêtes types |
| `.github/workflows/ci.yml` | CI GitHub Actions (tests automatiques) |

---

## CI / GitHub Actions

Le workflow `.github/workflows/ci.yml` s'exécute sur chaque push et pull request :

- Installe Python 3.12
- Installe les dépendances (`requirements.txt`)
- Lance `pytest -q` avec une clé `MISTRAL_API_KEY` factice (les tests mockent les appels réels)

Pour ajouter une étape de diagnostic (versions installées) :

```yaml
- name: Afficher les paquets installés
  run: pip list
```

---

## Dépendances principales

| Catégorie | Paquets |
|-------------|---------|
| Interface & LLM | `streamlit`, `mistralai`, `langchain`, `langchain-mistralai`, `pydantic-ai-slim[mistral]` |
| RAG / vecteurs | `faiss-cpu` |
| Documents | `PyPDF2`, `pymupdf`, `python-docx`, `easyocr`, `pillow` |
| Données | `pandas`, `openpyxl`, `sqlalchemy` |
| Config & rééseau | `python-dotenv`, `requests` |
| Qualité & validation | `pydantic`, `pytest`, `tabulate` |
| Observabilité & évaluation | `logfire`, `ragas`, `datasets` |
| Utilitaire | `tqdm` |

---

## Dépannage

### `pytest: command not found` (Windows)

Utilisez `python -m pytest -q` ou réinstallez dans le venv :

```bash
pip config set global.user false
pip install --force-reinstall -r requirements.txt
```

### `MISTRAL_API_KEY` manquante

Assurez-vous que le fichier `.env` existe à la racine et contient :

```
MISTRAL_API_KEY=votre_clé_ici
```

### Imports `utils.*` échouent en CI

Vérifiez que `utils/__init__.py` et `pyproject.toml` sont préents à la racine du dépot.

---

## Licence

Projet académique — OpenClassrooms, Projet 9 : Évaluez les performances d'un LLM.