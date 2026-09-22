# Rapport de mise en place et d’évaluation du système LLM

## NBA Analyst AI — Projet 9 OpenClassrooms

**Mission : Évaluez les performances d’un LLM**  
**Organisation : SportSee**  
**Domaine : analyse de performance basketball NBA**

---

## Résumé exécutif

SportSee souhaite valoriser des archives documentaires et des statistiques NBA afin d’aider les entraîneurs, analystes et préparateurs physiques à accéder plus rapidement à l’information pertinente. Le prototype initial reposait sur un système RAG (*Retrieval-Augmented Generation*) documentaire. L’évaluation a montré qu’un RAG seul est inadapté aux questions chiffrées : les valeurs saisonnières, les comparaisons de joueurs et les classements doivent être calculés à partir d’une source structurée et déterministe.

Le système final est donc un assistant hybride **SQL/RAG** :

- les questions narratives sont traitées par un RAG sur les rapports PDF ;
- les questions statistiques sont traitées par un SQL Tool sur une base SQLite ;
- un routeur LLM choisit la voie adaptée ;
- une abstention explicite est produite lorsque les sources ne permettent pas une réponse fiable ;
- Logfire assure la traçabilité des routes, recherches vectorielles, chunks récupérés et appels LLM.

La validation finale a confirmé le bon fonctionnement de la suite de tests avec **15 tests réussis**. Des démonstrations fonctionnelles ont validé un cas SQL positif, un cas RAG positif, une abstention SQL liée à la granularité et une abstention RAG liée à l’insuffisance documentaire.

---

## 1. Contexte et objectif métier

SportSee développe un assistant d’analyse de performance destiné aux clubs de basketball. Les utilisateurs cibles — entraîneurs, analystes vidéo et préparateurs physiques — doivent pouvoir interroger rapidement des rapports de match et des données statistiques afin de préparer une séance, un match ou un suivi d’athlète.

Le besoin initial consistait à exploiter des documents non structurés par RAG. Cependant, les sources disponibles sont hétérogènes :

- des rapports ou discussions de match au format PDF, utiles pour les analyses narratives ;
- un classeur Excel contenant des statistiques saisonnières, utile pour les réponses chiffrées.

L’objectif du projet est d’évaluer le comportement du LLM, d’identifier les limites d’un RAG documentaire seul, puis de construire un système plus fiable, explicable et reproductible.

---

## 2. Évaluation du prototype RAG seul

### 2.1 Limite fonctionnelle observée

Un RAG documentaire repose sur la récupération de passages textuels puis sur une génération LLM contrainte par ce contexte. Cette approche convient aux questions qualitatives, par exemple :

> « Que disent les rapports sur Julius Randle ? »

En revanche, elle est moins adaptée aux questions statistiques, par exemple :

> « Quel est le pourcentage à 3 points de Nikola Jokić ? »

Une valeur numérique présente dans un fichier Excel peut être mal extraite, mal associée au joueur ou absente des chunks récupérés. Demander au LLM de produire une réponse chiffrée à partir de texte extrait crée un risque d’erreur ou d’hallucination.

### 2.2 Résultats RAGAS disponibles

Le jeu d’évaluation est stocké dans `data/eval_dataset.jsonl`. Le script `evaluate_ragas.py` produit un fichier détaillé `reports/ragas_results.csv` et une synthèse `reports/ragas_results.md`.

Les métriques suivies sont :

- **Faithfulness** : mesure dans quelle mesure la réponse est soutenue par le contexte récupéré ;
- **Response Relevancy** : mesure l’adéquation de la réponse à la question ;
- **Context Precision** : évalue la pertinence des chunks récupérés ;
- **Context Recall** : évalue la couverture du contexte de référence.

L’évaluation du prototype a mis en évidence deux constats :

1. les questions narratives peuvent être correctement traitées lorsque les PDF contiennent des passages explicites ;
2. les questions chiffrées, les requêtes bruitées et les demandes de granularité non disponible ne doivent pas dépendre du RAG documentaire.

La mise en place du routeur SQL/RAG répond directement à cette limite.

---

## 3. Architecture cible SQL/RAG

### 3.1 Principe de routage

Le système final distingue la nature de la demande avant de générer une réponse.

```text
Question utilisateur
        |
        v
Routeur SQL / RAG
        |
        +-----------------------------+
        |                             |
        v                             v
Question statistique             Question narrative
        |                             |
        v                             v
SQL Tool sécurisé                RAG documentaire
SQLite                            PDF → chunks → FAISS
        |                             |
        +-------------+---------------+
                      |
                      v
     Réponse structurée ou abstention explicite
                      |
                      v
            Interface Streamlit + Logfire
```

Le routeur est implémenté dans `utils/rag_pipeline_router.py`. Il classe les questions en deux routes :

| Route | Cas d’usage | Composants |
|---|---|---|
| SQL | Pourcentages, rebonds, comparaisons, agrégations, classements | `utils/sql_tool.py`, SQLite, `database/nba_analytics.db` |
| RAG | Analyse tactique, avis, discussions et contexte de match | `utils/rag_pipeline.py`, `utils/vector_store.py`, FAISS |

### 3.2 Chaîne RAG documentaire

Les documents textuels sont chargés depuis `inputs/`, découpés en chunks de 1 500 caractères avec un chevauchement de 150 caractères, puis convertis en embeddings Mistral. Les embeddings sont normalisés et stockés dans un index FAISS `IndexFlatIP` ; le score de recherche correspond donc à une similarité cosinus.

Le pipeline RAG suit les étapes suivantes :

1. génération de l’embedding de la question ;
2. récupération des `top_k=5` chunks les plus similaires ;
3. constitution d’un contexte comprenant les identifiants de chunks et les sources ;
4. appel d’un agent Pydantic AI ;
5. production d’une `AssistantAnswer` structurée : réponse, chunks cités, confiance et statut d’abstention.

Le prompt RAG impose que le modèle utilise exclusivement les informations présentes dans les chunks récupérés. Si le contexte est ambigu ou insuffisant, le système répond exactement qu’il ne peut pas répondre avec les sources disponibles.

### 3.3 Chaîne SQL déterministe

Les statistiques du classeur Excel sont chargées dans SQLite par `load_excel_to_db.py`. Les données sont validées avec Pydantic avant insertion dans les tables `players`, `stats`, `matches` et `reports`.

Le SQL Tool :

- reçoit le schéma SQL et des exemples few-shot ;
- génère une requête SQLite ou la décision `ABSTAIN` ;
- n’accepte qu’une requête `SELECT` unique ;
- bloque les mots-clés de modification et d’administration ;
- ajoute une limite de 20 lignes lorsque nécessaire ;
- fournit une réponse synthétisée uniquement à partir du résultat SQL.

La granularité actuellement disponible est saisonnière. Les demandes sur les cinq derniers matchs, les matchs individuels, les dates ou le domicile/extérieur doivent conduire à une abstention explicite plutôt qu’à une valeur inventée.

---

## 4. Préparation des données et amélioration du corpus

### 4.1 Séparation des données structurées et narratives

Le diagnostic initial a montré que l’index FAISS contenait à la fois les PDF narratifs et le fichier Excel. Les recherches RAG pouvaient alors remonter des chunks contenant des valeurs `NaN`, des en-têtes de tableaux ou des définitions de métriques. Ces passages dégradaient la pertinence du contexte transmis au LLM.

Exemple observé lors d’une question sur Nikola Jokić : la recherche récupérait des chunks Excel avec un score de similarité comparable aux chunks PDF, mais ces chunks étaient impossibles à utiliser pour une analyse narrative.

La correction appliquée consiste à **exclure les fichiers structurés, notamment `.xlsx`, de l’index RAG**. Les documents Excel sont désormais réservés au pipeline SQL. L’index FAISS est construit uniquement à partir des documents narratifs compatibles, principalement les PDF.

Cette séparation suit le principe suivant :

| Type de donnée | Stockage et traitement | Usage métier |
|---|---|---|
| PDF / texte narratif | Chunks, embeddings Mistral, FAISS, RAG | Analyse documentaire et qualitative |
| Excel structuré | Validation Pydantic, SQLite, SQL Tool | Statistiques, comparaisons et agrégations |

### 4.2 Traçabilité de la récupération

Le module `utils/vector_store.py` trace dans Logfire :

- la question vectorisée ;
- le nombre de résultats récupérés ;
- les identifiants de chunks ;
- les noms de fichiers sources ;
- les scores de similarité ;
- un extrait des passages récupérés.

L’interface Streamlit affiche, en cas d’abstention RAG, une section « Extraits récupérés pour diagnostic ». Ces extraits ne sont pas présentés comme des sources justifiant une réponse : ils expliquent pourquoi la réponse n’est pas suffisamment étayée.

---

## 5. Fiabilité, garde-fous et observabilité

### 5.1 Validation des données et sorties structurées

Les modèles Pydantic définissent des contrats pour les documents, chunks, requêtes, résultats SQL, cas d’évaluation et réponses finales. Cette validation réduit les erreurs silencieuses liées aux structures de données incomplètes ou incohérentes.

La réponse finale est modélisée par `AssistantAnswer` :

```text
answer            Texte retourné à l’utilisateur
cited_chunk_ids   Identifiants des chunks réellement utilisés
confidence        high, medium ou low
abstained         true si le système ne peut pas répondre fiablement
```

### 5.2 Abstention contrôlée

L’abstention est une fonctionnalité centrale du système :

- la route SQL s’abstient lorsque la granularité demandée n’existe pas dans les données structurées ;
- la route RAG s’abstient lorsque les chunks récupérés ne contiennent pas de preuve explicite ;
- l’application n’invente pas de chiffres ni d’analyse documentaire non sourcée.

### 5.3 Observabilité avec Logfire

Logfire est utilisé pour tracer les principales étapes :

- classification SQL/RAG ;
- exécution de la branche SQL ou RAG ;
- génération des embeddings ;
- recherche vectorielle ;
- nombre de chunks récupérés ;
- consommation de tokens ;
- réponse affichée et statut d’abstention.

Cette instrumentation a permis d’identifier puis de corriger le mélange entre Excel et PDF dans l’index vectoriel.

---

## 6. Validation et résultats

### 6.1 Tests automatisés

La suite de tests a été exécutée avec succès :

```text
15 passed
```

Les tests couvrent notamment :

| Fichier de test | Objet couvert |
|---|---|
| `tests/test_schemas.py` | Validation des schémas Pydantic |
| `tests/test_ingestion.py` | Ingestion et validation des données |
| `tests/test_sql_tool.py` | Sécurité et exécution du SQL Tool |
| `tests/test_router.py` | Routage SQL/RAG, abstention SQL et repli contrôlé |

Un avertissement de dépréciation provenant de FAISS et NumPy peut apparaître dans l’environnement de développement. Il provient d’une dépendance externe et ne bloque pas les tests ni l’exécution de l’application.

### 6.2 Scénarios fonctionnels validés

| Scénario | Question | Route observée | Résultat |
|---|---|---|---|
| Statistique saisonnière | « Quel est le pourcentage à 3 points de Nikola Jokic ? » | SQL | Réponse : 41,7 % |
| Analyse documentaire positive | « Que disent les rapports sur Julius Randle ? » | RAG | Synthèse narrative avec sources `0_1` et `0_0` |
| Granularité indisponible | « Quel est le meilleur pourcentage à 3 points sur les cinq derniers matchs ? » | SQL | Abstention : données saisonnières uniquement |
| Information documentaire insuffisante | « Que disent les rapports sur Nikola Jokic ? » | RAG | Abstention justifiée et extraits de diagnostic affichés |

Le cas Julius Randle démontre que le RAG peut produire une synthèse fidèle lorsque le corpus contient une information explicite. La réponse indique notamment son impact physique, son efficacité offensive, sa lecture des prises à deux et son implication défensive, en citant les chunks utilisés.

Le cas Nikola Jokić démontre le comportement de sûreté : les passages récupérés ne parlaient pas d’une analyse de son jeu, mais notamment de supporters ou de sujets hors périmètre. Le système s’est donc abstenu au lieu de générer une analyse infondée.

---

## 7. Limites identifiées

Le système final reste soumis aux limites suivantes :

- le corpus PDF est limité et composé de discussions Reddit parfois bruitées ;
- toutes les équipes et tous les joueurs ne sont pas documentés de manière homogène ;
- les statistiques Excel sont agrégées à la saison ; elles ne permettent pas encore l’analyse match par match ou domicile/extérieur ;
- le routeur étant basé sur un LLM, une question hybride ou ambiguë peut être mal classifiée ;
- RAGAS dépend de la disponibilité de références et de contextes adaptés ; certaines métriques peuvent être indisponibles selon les cas d’évaluation.

---

## 8. Recommandations d’évolution

1. Ajouter des rapports de match éditorialisés, structurés et fiables afin d’améliorer la couverture narrative.
2. Ajouter un export statistique match par match avec dates, adversaires et indicateurs domicile/extérieur.
3. Mettre en place un reranker après FAISS afin de renforcer la précision des passages transmis au LLM.
4. Étendre le jeu d’évaluation avec des cas spécifiques de routage, des questions mixtes et des formulations bruitées.
5. Suivre dans Logfire les scores de similarité et taux d’abstention pour détecter les régressions après chaque évolution du corpus.
6. Ajouter un test automatisé garantissant que les fichiers Excel ne sont jamais inclus dans le corpus RAG.

---

## 9. Reproductibilité

Les principaux fichiers nécessaires à la reproduction sont :

```text
requirements.txt
pyproject.toml
README.md
indexer.py
load_excel_to_db.py
evaluate_ragas.py
MistralChat.py
data/eval_dataset.jsonl
tests/
utils/
db/
```

Procédure minimale :

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

# Créer .env et renseigner MISTRAL_API_KEY
python load_excel_to_db.py --excel "inputs/regular NBA.xlsx" --reports-dir inputs --season 2024-2025
python indexer.py
python -m pytest -q
streamlit run MistralChat.py
```

Le fichier `.env` est volontairement exclu du contrôle de version. Il ne doit jamais être inclus dans une archive de livraison.

---

## Conclusion

L’évaluation du prototype a montré les limites d’un RAG documentaire seul pour répondre aux questions chiffrées. Le système final SQL/RAG sépare les responsabilités : les statistiques structurées sont traitées de façon déterministe par SQL, tandis que les documents non structurés sont analysés par un pipeline RAG traçable.

Cette architecture améliore la fiabilité opérationnelle de l’assistant : elle fournit des réponses chiffrées exactes lorsque les données existent, produit des synthèses narratives sourcées lorsque les documents sont pertinents, et s’abstient explicitement lorsque les sources ne permettent pas de répondre de manière fondée.
