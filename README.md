# NBA Analyst AI

Assistant intelligent d’analyse de performance NBA combinant :

- un pipeline **RAG** (*Retrieval-Augmented Generation*) sur des rapports PDF ;
- un outil **SQL** déterministe sur des statistiques saisonnières ;
- une route **MIXED** qui combine SQL et RAG pour les questions hybrides (texte + chiffres) ;
- un routeur **SQL/RAG/MIXED** qui choisit automatiquement la meilleure stratégie ;
- une interface conversationnelle **Streamlit** ;
- une évaluation de la qualité du système avec **RAGAS** (dataset v2 incluant des questions MIXED).

---

## Objectif

NBA Analyst AI aide les entraîneurs, analystes et préparateurs physiques à retrouver rapidement trois types d’informations :

| Besoin | Exemple de question | Route / source utilisée |
|---|---|---|
| Statistiques saisonnières fiables | « Quel est le pourcentage à 3 points de Nikola Jokić ? » | Route SQL, base SQLite via SQL Tool |
| Analyse narrative de rapports | « Que disent les rapports sur la défense de Denver ? » | Route RAG, PDF indexés via RAG/FAISS |
| Question mixte texte + chiffres | « Compare les rebonds de Julius Randle et Nikola Jokić, puis explique ce que les rapports disent du jeu de Randle. » | Route MIXED : SQL + RAG sur la même question |
| Question non couverte par les données | « Quel est le meilleur 3P% sur les cinq derniers matchs ? » | Abstention explicite (granularité indisponible) |

Le projet évite de répondre avec une valeur inventée lorsqu’une granularité demandée n’existe pas dans les données disponibles, aussi bien pour les questions purement chiffrées que pour les questions mixtes.

---

## Architecture

```
Question utilisateur
        |
        v
Routeur SQL / RAG / MIXED ─────────────── utils/rag_pipeline_router.py
        |
        +-------------------------+-------------------------+
        |                         |                         |
        v                         v                         v
    Route SQL                 Route RAG                 Route MIXED
        |                         |                         |
        v                         v                         v
    SQL Tool             Recherche vectorielle       Orchestrateur MIXED :
utils/sql_tool.py        FAISS + pipeline RAG       - appelle SQL Tool
        |               utils/rag_pipeline.py       - appelle RAG
        v                         |                 - assemble les réponses
      SQLite                     v
  nba_analytics.db        Rapports PDF indexés
                              vector_db/
        |                         |                         |
        +-------------+-----------+-------------------------+
                      |
                      v
           AssistantAnswer structuré
      (réponse, chunks cités, confiance, abstention)
                      |
                      v
             Interface Streamlit
             MistralChat.py
```

### Composants principaux

| Composant | Rôle |
|---|---|
| `MistralChat.py` | Interface conversationnelle Streamlit |
| `utils/rag_pipeline_router.py` | Routage d’une question vers SQL, RAG ou MIXED |
| `utils/sql_tool.py` | Génération, validation et exécution sécurisée de requêtes SQL |
| `utils/rag_pipeline.py` | Recherche de contexte et génération de réponse RAG |
| `utils/vector_store.py` | Création, chargement et recherche dans l’index FAISS |
| `load_excel_to_db.py` | Ingestion des données Excel et PDF vers SQLite |
| `indexer.py` | Création de l’index vectoriel à partir du dossier `inputs/` |
| `evaluate_ragas.py` | Évaluation automatique de la qualité des réponses via RAGAS (dataset v2 avec cas MIXED) |

---

## Prérequis

- Python 3.12 recommandé ;
- une clé API Mistral ;
- Git si vous clonez le dépôt.

---

## Installation

### 1. Cloner le dépôt

```bash
git clone [https://github.com/JoseBravo26/P9_LLM.git](https://github.com/JoseBravo26/P9_LLM.git)
cd P9_LLM
```

### 2. Créer un environnement virtuel

Windows — Git Bash :

```bash
python -m venv venv
source venv/Scripts/activate
```

Windows — PowerShell :

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS / Linux :

```bash
python3 -m venv venv
source venv/bin/activate
```

Si, sous Windows, `pip` installe les bibliothèques dans `AppData\Roaming` plutôt que dans l’environnement virtuel, désactivez l’option utilisateur globale :

```bash
pip config set global.user false
```

### 3. Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurer les variables d’environnement

Créez un fichier `.env` à la racine du dépôt :

```env
MISTRAL_API_KEY=votre_cle_api_mistral
MODEL_NAME=mistral-small-latest
EMBEDDING_MODEL=mistral-embed
LOGFIRE_TOKEN=votre_jeton_logfire
DISABLE_SSL_VERIFICATION=0
```

Sur un réseau d’entreprise ou derrière un proxy qui intercepte les certificats SSL :

```env
DISABLE_SSL_VERIFICATION=1
```

---

## Préparer les données

Le projet contient des rapports PDF dans `inputs/` et un fichier Excel de statistiques NBA saisonnières.

### 1. Alimenter la base SQL

Cette commande valide les lignes du fichier Excel avec Pydantic, alimente les tables `players` et `stats`, puis extrait les PDF pour alimenter la table `reports` :

```bash
python load_excel_to_db.py \
  --excel "inputs/regular NBA.xlsx" \
  --reports-dir inputs \
  --season 2024-2025
```

La base produite se trouve dans :

```
database/nba_analytics.db
```

### 2. Construire l’index vectoriel RAG

```bash
python indexer.py
```

Cette étape lit les documents `inputs/*.pdf`, les découpe en segments (chunks), calcule leurs embeddings avec Mistral et enregistre l’index FAISS dans :

```
vector_db/faiss_index.idx
vector_db/document_chunks.pkl
```

Les fichiers Excel sont explicitement exclus de l’index RAG et réservés au pipeline SQL, afin d’éviter l’injection de tableaux ou de valeurs non textuelles dans le contexte documentaire.

---

## Utilisation

### Lancer les tests

```bash
python -m pytest -q
```

Tests ciblés :

```bash
python -m pytest -q tests/test_schemas.py
python -m pytest -q tests/test_ingestion.py
python -m pytest -q tests/test_sql_tool.py
python -m pytest -q tests/test_router.py
```

### Lancer l’application

```bash
streamlit run MistralChat.py
```

Streamlit ouvre généralement l’interface sur `http://localhost:8501`.

L’interface affiche la source de réponse :

- 📊 Statistiques lorsque la question est traitée par SQL ;
- 📄 Analyse documentaire lorsque la réponse vient des rapports PDF (RAG) ;
- 🔀 Réponse mixte lorsque la question est routée vers MIXED ;
- une abstention explicite lorsque les données ne permettent pas une réponse fiable.

### Exemples de questions

| Question | Route attendue | Résultat attendu |
|---|---|---|
| « Quel est le pourcentage à 3 points de Nikola Jokić ? » | SQL | Valeur extraite des statistiques de saison |
| « Compare les rebonds de Jokić et Towns. » | SQL | Comparaison chiffrée synthétisée |
| « Que disent les rapports sur la défense de Denver ? » | RAG | Synthèse issue des passages PDF pertinents ou abstention documentaire si les sources sont insuffisantes |
| « Compare les rebonds de Julius Randle et Nikola Jokić, puis explique ce que les rapports disent du jeu de Randle. » | MIXED | Partie SQL pour les rebonds, partie RAG pour l’analyse de Randle, avec abstention partielle si les sources manquent |
| « Qui a le meilleur 3P% sur les cinq derniers matchs ? » | SQL | Abstention : données match par match indisponibles |
| « Compare les rebonds à domicile et à l’extérieur. » | SQL | Abstention : statut domicile/extérieur indisponible |

---

## Sécurité et fiabilité

### Validation SQL

Le SQL Tool s’appuie sur le schéma `db/schema.sql` et des exemples few-shot contenus dans `few_shot_sql_examples.py`. Avant exécution, il applique les contrôles suivants :

- seule une requête `SELECT` unique est acceptée ;
- les instructions de modification ou d’administration sont refusées ;
- une limite de 20 lignes est appliquée par défaut ;
- les demandes incompatibles avec la granularité des données génèrent une abstention.

La base `stats` contient actuellement des statistiques de saison. Elle ne fournit pas les dates de matchs, les données domicile/extérieur ni les cinq derniers matchs.

### Validation et observabilité

- Les contrats de données sont définis par des modèles Pydantic.
- Les réponses sont structurées via `AssistantAnswer` (texte, chunks cités, confiance, abstention).
- Logfire peut tracer les étapes du pipeline (routage, recherche vectorielle, scores, consommation de tokens).
- Les clés API restent dans le fichier `.env`.

---

## Évaluation RAGAS (dataset v2, avec questions MIXED)

L’évaluation finale s’appuie sur `data/eval_dataset_v2.jsonl`, qui étend le jeu d’évaluation en ajoutant 4 questions de catégorie `mixed` (questions hybrides SQL + RAG).

Exemple de commande :

```bash
python evaluate_ragas.py \
  --dataset data/eval_dataset_v2.jsonl \
  --run-name final_sql_rag_text_numeric \
  --full-metrics
```

Les sorties sont générées dans :

```
reports/final_sql_rag_text_numeric_YYYY-MM-DD_HH-MM-SS.csv
reports/final_sql_rag_text_numeric_YYYY-MM-DD_HH-MM-SS.md
reports/final_sql_rag_text_numeric_YYYY-MM-DD_HH-MM-SS_routes.md
reports/final_sql_rag_text_numeric_YYYY-MM-DD_HH-MM-SS_abstentions.md
```

### Résultats par route

| Route | Nombre de questions | Taux d’abstention correcte |
|---|---:|---:|
| SQL   | 32 | 1,00 |
| RAG   | 10 | 0,80 |
| MIXED |  4 | 0,50 |

### Résultats par catégorie et route

| Catégorie / route        | Faithfulness | Answer relevancy | Taux d’abstention correcte |
|:-------------------------|------------:|-----------------:|---------------------------:|
| `bruitee` — SQL         | 1,000       | 0,782            | 1,00                      |
| `complexe` — SQL        | 1,000       | 0,765            | 1,00                      |
| `hors_perimetre` — RAG  | 0,000       | 0,000            | 1,00                      |
| `hors_perimetre` — SQL  | 1,000       | 0,000            | 1,00                      |
| `non_repondable` — SQL  | 1,000       | 0,000            | 1,00                      |
| `simple` — SQL          | 1,000       | 0,972            | 1,00                      |
| `textuelle` — RAG       | 0,720       | 0,670            | 0,75                      |
| `mixed` — MIXED         | 1,000       | 0,211            | 0,50                      |

Ces résultats montrent une branche SQL très fiable, une branche RAG adaptée mais encore perfectible, et une branche MIXED fonctionnelle qui nécessite des améliorations ciblées sur la pertinence globale et la gestion des réponses partielles.

---

## Structure du projet

```
P9_LLM/
├── .github/workflows/ci.yml       # Pipeline GitHub Actions
├── data/
│   ├── eval_dataset.jsonl         # Jeu d’évaluation RAGAS v1
│   └── eval_dataset_v2.jsonl      # Jeu d’évaluation RAGAS v2 avec cas MIXED
├── database/
│   └── nba_analytics.db           # Base SQLite générée
├── db/
│   ├── schema.sql                 # Schéma relationnel
│   └── queries_examples.sql       # Exemples de requêtes SQL
├── inputs/
│   ├── regular NBA.xlsx           # Statistiques saisonnières
│   └── Reddit *.pdf               # Rapports/commentaires de matchs
├── reports/
│   ├── ragas_results.csv
│   ├── ragas_results.md
│   └── final_sql_rag_text_numeric_*.md / *.csv
├── tests/
│   ├── test_ingestion.py
│   ├── test_router.py
│   ├── test_schemas.py
│   └── test_sql_tool.py
├── utils/
│   ├── config.py                  # Configuration et variables d’environnement
│   ├── data_loader.py             # Chargement de documents
│   ├── database.py                # Connexion SQLite
│   ├── db_schemas.py              # Schémas Pydantic SQL
│   ├── rag_pipeline.py            # Pipeline RAG
│   ├── rag_pipeline_router.py     # Routeur SQL/RAG/MIXED
│   ├── schemas.py                 # Schémas Pydantic RAG
│   ├── sql_tool.py                # SQL Tool sécurisé
│   └── vector_store.py            # Index FAISS et embeddings
├── MistralChat.py                 # Application Streamlit
├── evaluate_ragas.py              # Script d’évaluation RAGAS
├── few_shot_sql_examples.py       # Exemples de génération SQL
├── indexer.py                     # Script de création de l’index
├── load_excel_to_db.py            # Script d’ingestion
├── pyproject.toml                 # Configuration Pytest
└── requirements.txt               # Dépendances Python
```

---

## CI GitHub Actions et dépannage

Le workflow `.github/workflows/ci.yml` :

1. configure Python 3.12 ;
2. installe les dépendances de `requirements.txt` ;
3. lance `pytest -q` avec une clé Mistral factice, les appels réseau étant mockés dans les tests concernés.

Dépannage rapide :

- `pytest: command not found` : `python -m pytest -q`, puis réinstaller les dépendances dans l’environnement virtuel si nécessaire.
- `MISTRAL_API_KEY` manquante : vérifier `.env` à la racine du projet.
- Erreur d’import `utils.*` : lancer les commandes depuis la racine et vérifier `utils/__init__.py` et `pyproject.toml`.
- Index FAISS absent ou obsolète : reconstruire l’index après toute modification dans `inputs/` avec `python indexer.py`.

---

## Licence

Projet académique — OpenClassrooms, Projet 9 : Évaluez les performances d’un LLM.