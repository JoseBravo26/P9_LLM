# Rapport de mise en place et d’évaluation du système LLM

## NBA Analyst AI — Projet 9 OpenClassrooms

Mission : Évaluez les performances d’un LLM  
Organisation : SportSee  
Domaine : analyse de performance basketball NBA

---

## Résumé exécutif

SportSee souhaite valoriser des archives documentaires et des statistiques NBA afin d’aider les entraîneurs, analystes et préparateurs physiques à accéder plus rapidement à l’information pertinente. Le prototype initial reposait sur un système RAG documentaire. L’évaluation a montré qu’un RAG seul est inadapté aux questions chiffrées : les valeurs saisonnières, les comparaisons de joueurs et les classements doivent être calculés à partir d’une source structurée et déterministe.

Le système final est donc un assistant hybride SQL/RAG/MIXED :

- les questions narratives sont traitées par un RAG sur les rapports PDF ;
- les questions statistiques sont traitées par un SQL Tool sur une base SQLite ;
- les questions mixtes (texte + chiffres) sont traitées par une route MIXED qui orchestre SQL et RAG sur la même requête ;
- un routeur LLM choisit automatiquement la voie adaptée (SQL, RAG ou MIXED) ;
- une abstention explicite est produite lorsque les sources ne permettent pas une réponse fiable ;
- Logfire assure la traçabilité des routes, recherches vectorielles, chunks récupérés et appels LLM.

La validation finale confirme le bon fonctionnement de la suite de tests Pytest et une évaluation RAGAS v2 sur 46 questions, incluant 4 cas de catégorie `mixed`. La branche SQL est parfaitement fiable, la branche RAG reste adaptée aux questions textuelles et la branche MIXED est fonctionnelle mais encore perfectible.

---

## 1. Contexte et objectif métier

SportSee développe un assistant d’analyse de performance destiné aux clubs de basketball. Les utilisateurs cibles sont des entraîneurs, analystes vidéo et préparateurs physiques. Ils doivent pouvoir interroger rapidement :

- des rapports de match pour obtenir une analyse narrative ;
- des données statistiques pour obtenir des valeurs chiffrées fiables.

Les sources disponibles sont hétérogènes :

- des rapports ou discussions de match au format PDF, utiles pour les analyses narratives ;
- un classeur Excel contenant des statistiques saisonnières, utile pour les réponses chiffrées.

L’objectif du projet est d’évaluer le comportement du LLM, d’identifier les limites d’un RAG documentaire seul, puis de construire un système plus fiable, explicable et reproductible, y compris pour des questions mixtes qui combinent statistiques et analyse.

---

## 2. Évaluation du prototype RAG seul

### 2.1 Limite fonctionnelle observée

Un RAG documentaire repose sur la récupération de passages textuels puis sur une génération LLM contrainte par ce contexte. Cette approche convient aux questions qualitatives, par exemple :

« Que disent les rapports sur Julius Randle ? »

En revanche, elle est moins adaptée aux questions statistiques, par exemple :

« Quel est le pourcentage à 3 points de Nikola Jokić ? »

Une valeur numérique présente dans un fichier Excel peut être mal extraite, mal associée au joueur ou absente des chunks récupérés. Demander au LLM de produire une réponse chiffrée à partir de texte extrait crée un risque d’erreur ou d’hallucination, notamment pour les pourcentages, comparaisons et classements.

### 2.2 Résultats RAGAS du prototype RAG seul

Le jeu d’évaluation initial (dataset v1) est stocké dans `data/eval_dataset.jsonl`. Le script `evaluate_ragas.py` produit :

- un fichier détaillé `reports/ragas_results.csv` ;
- une synthèse `reports/ragas_results.md`.

Les métriques suivies sont :

- faithfulness : mesure dans quelle mesure la réponse est soutenue par le contexte récupéré ;
- response_relevancy : mesure l’adéquation de la réponse à la question ;
- context_precision : évalue la pertinence des chunks récupérés ;
- context_recall : évalue la couverture du contexte de référence.

L’évaluation du prototype a mis en évidence :

- des questions narratives correctement traitées lorsque les PDF contiennent des passages explicites ;
- des questions chiffrées et des demandes de granularité non disponible qui ne doivent pas dépendre du RAG documentaire seul.

Ces constats motivent la mise en place d’un routeur SQL/RAG puis l’extension à la route MIXED.

---

## 3. Architecture cible SQL/RAG/MIXED

### 3.1 Principe de routage

Le système final distingue la nature de la demande avant de générer une réponse.

```
Question utilisateur
        |
        v
MistralChat.py
        |
        v
answer(question)
        |
        v
route_question(question)
        |
        +----------------------------+----------------------------+----------------------------+
        |                            |                            |
        v                            v                            v
      "SQL"                        "RAG"                        "MIXED"
        |                            |                            |
        v                            v                            v
answer_with_sql()             answer_with_rag()               answer_mixed()
        |                            |                            |
        v                            v                            v
execute_sql()                 answer_question()         SQL + RAG + assemblage
        |                            |                            |
        v                            v                            v
SQLite + validation     FAISS + chunks PDF + LLM   AssistantAnswer combinée
        |                            |                            |
        +---------------+------------+----------------------------+
                        |
                        v
AssistantAnswer structuré (réponse, chunks cités, confiance, abstention)
                        |
                        v
Affichage Streamlit + Logfire
```

Exemples de règles de décision :

- questions purement chiffrées → route SQL ;
- questions purement narratives → route RAG ;
- questions combinant une demande chiffrée et une analyse narrative dans une même phrase → route MIXED.

### 3.2 Chaîne RAG documentaire

Les documents textuels sont chargés depuis `inputs/`, découpés en chunks de 1 500 caractères avec un chevauchement de 150 caractères, puis convertis en embeddings Mistral. Les embeddings sont normalisés et stockés dans un index FAISS `IndexFlatIP` ; le score de recherche correspond à une similarité cosinus.

Le pipeline RAG :

1. génère l’embedding de la question ;
2. récupère les `top_k=5` chunks les plus similaires ;
3. construit un contexte contenant texte et métadonnées ;
4. appelle un agent Pydantic AI ;
5. produit une réponse structurée avec statut d’abstention.

Le prompt impose que le modèle utilise exclusivement les informations présentes dans les chunks. Si le contexte est ambigu, hors sujet ou insuffisant, la réponse est une abstention documentaire.

### 3.3 Chaîne SQL déterministe

Les statistiques du classeur Excel sont chargées dans SQLite par `load_excel_to_db.py`. Les données sont validées avec Pydantic avant insertion dans les tables `players`, `stats`, `matches` et `reports`.

Le SQL Tool :

- reçoit le schéma SQL et des exemples few-shot ;
- génère une requête SQLite ou la décision `ABSTAIN` ;
- n’accepte qu’une requête `SELECT` unique ;
- bloque les mots-clés de modification et d’administration ;
- applique une limite de 20 lignes par défaut ;
- fournit une réponse synthétisée uniquement à partir du résultat SQL.

La granularité disponible est saisonnière. Les demandes sur cinq derniers matchs, quart‑temps, adversaires ou domicile/extérieur conduisent à une abstention structurée.

---

## 4. Préparation des données et amélioration du corpus

### 4.1 Séparation des données structurées et narratives

Le diagnostic initial a montré que l’index FAISS contenait à la fois les PDF narratifs et le fichier Excel. Les recherches RAG pouvaient remonter des chunks contenant des valeurs non textuelles ou des en‑têtes de tableaux, inutiles pour l’analyse documentaire.

La correction consiste à exclure les fichiers Excel (`.xlsx`) de l’index RAG et à les réserver au pipeline SQL. L’index FAISS est construit uniquement à partir des documents narratifs compatibles, principalement les PDF.

Résumé :

- PDF : chunks, embeddings, FAISS, RAG ;
- Excel : validation Pydantic, SQLite, SQL Tool.

### 4.2 Traçabilité de la récupération

Le module `utils/vector_store.py` trace dans Logfire :

- la question vectorisée ;
- le nombre de résultats récupérés ;
- les identifiants de chunks ;
- les fichiers sources ;
- les scores de similarité ;
- des extraits textuels de diagnostic.

L’interface Streamlit affiche, en cas d’abstention RAG, une section « Extraits récupérés pour diagnostic » expliquant pourquoi aucune réponse fiable ne peut être donnée.

---

## 5. Fiabilité, garde-fous et observabilité

### 5.1 Validation des données et sorties structurées

Les modèles Pydantic définissent des contrats pour les documents, chunks, requêtes SQL, résultats SQL, cas d’évaluation et réponses finales.

La réponse finale est modélisée par un objet `AssistantAnswer` :

- texte retourné à l’utilisateur ;
- identifiants des chunks utilisés ;
- niveau de confiance (`high`, `medium`, `low`) ;
- indicateur d’abstention (`True` ou `False`).

### 5.2 Abstention contrôlée

L’abstention est une fonctionnalité centrale :

- la route SQL s’abstient lorsque la granularité demandée n’existe pas ;
- la route RAG s’abstient lorsque les chunks ne contiennent pas de preuve explicite ;
- la route MIXED s’abstient globalement lorsque les deux branches (SQL et RAG) échouent simultanément ;
- aucune valeur chiffrée ni analyse narrative n’est inventée.

### 5.3 Observabilité avec Logfire

Logfire trace :

- le routage (SQL, RAG, MIXED) ;
- l’exécution de chaque branche ;
- la génération des embeddings ;
- la recherche vectorielle FAISS ;
- les scores de similarité ;
- la consommation de tokens ;
- les réponses finales et les statuts d’abstention.

Cette instrumentation a permis d’identifier le mélange Excel/PDF dans l’index vectoriel et certains comportements de la route MIXED à améliorer.

---

## 6. Validation et résultats (dataset v2, avec MIXED)

### 6.1 Tests automatisés

La suite de tests Pytest couvre :

- les schémas Pydantic ;
- l’ingestion des données ;
- la sécurité et l’exécution du SQL Tool ;
- le routage SQL/RAG/MIXED.

Les tests se terminent avec succès. Un avertissement de dépréciation FAISS/NumPy peut apparaître sans bloquer l’exécution.

### 6.2 Protocole d’évaluation final

L’évaluation finale (dataset v2) porte sur 46 questions réparties en sept catégories :

- simple ;
- complexe ;
- bruitée ;
- textuelle ;
- non répondable ;
- hors périmètre ;
- mixed.

Les métriques suivies sont :

- faithfulness ;
- answer_relevancy ;
- routage correct ;
- abstention correcte.

### 6.3 Routage et abstention

Par route :

| Route  | Nombre de questions | Taux d’abstention correcte |
|---|---:|---:|
| SQL   | 32 | 1,00 |
| RAG   | 10 | 0,80 |
| MIXED |  4 | 0,50 |

Par catégorie :

| Catégorie       | Nombre de questions | Abstentions attendues | Abstentions observées | Taux d’abstention correcte |
|:---------------|--------------------:|-----------------------:|-----------------------:|---------------------------:|
| bruitée        | 7                  | 0                     | 0                     | 1,00                      |
| complexe       | 9                  | 0                     | 0                     | 1,00                      |
| hors périmètre | 4                  | 4                     | 4                     | 1,00                      |
| non répondable | 6                  | 6                     | 6                     | 1,00                      |
| simple         | 8                  | 0                     | 0                     | 1,00                      |
| textuelle      | 8                  | 0                     | 2                     | 0,75                      |
| mixed          | 4                  | 0                     | 2                     | 0,50                      |

La route SQL obtient un taux d’abstention correcte de 100 %. La route RAG atteint 80 % d’abstention correcte, ce qui reste satisfaisant mais montre une légère marge de progression. La route MIXED atteint 50 % d’abstention correcte : la fonctionnalité est en place, mais l’orchestrateur doit mieux gérer les cas où une réponse partielle est acceptable.

### 6.4 Métriques de qualité

Par couple catégorie / route :

| Catégorie et route        | Faithfulness | Answer relevancy | Abstention correcte |
|:--------------------------|------------:|-----------------:|-------------------:|
| Bruitée — SQL            | 1,000       | 0,782            | 1,00              |
| Complexe — SQL           | 1,000       | 0,765            | 1,00              |
| Hors périmètre — RAG     | 0,000       | 0,000            | 1,00              |
| Hors périmètre — SQL     | 1,000       | 0,000            | 1,00              |
| Non répondable — SQL     | 1,000       | 0,000            | 1,00              |
| Simple — SQL             | 1,000       | 0,972            | 1,00              |
| Textuelle — RAG          | 0,720       | 0,670            | 0,75              |
| Mixed — MIXED            | 1,000       | 0,211            | 0,50              |

Interprétation :

- Les questions simples et complexes routées vers SQL conservent une fidélité parfaite et une forte pertinence, ce qui confirme le choix d’un pipeline SQL déterministe pour les statistiques structurées.
- La branche RAG textuelle reste adaptée, avec une légère baisse de pertinence par rapport à la version précédente, due principalement aux limites du corpus PDF.
- La branche MIXED présente une fidélité parfaite mais une pertinence globale faible : certaines questions hybrides sont partiellement traitées, d’autres conduisent à une abstention totale alors qu’une réponse partielle serait acceptable.

### 6.5 Scénarios fonctionnels

Exemples :

- question simple SQL : « Quel est le pourcentage à 3 points de Nikola Jokić ? » → réponse 41,7 % ;
- question textuelle RAG : « Que disent les rapports sur Julius Randle ? » → synthèse narrative basée sur plusieurs passages ;
- question non répondable SQL : « Quel est le meilleur 3P% sur les cinq derniers matchs ? » → abstention structurée ;
- question MIXED avec succès partiel : « Quel est le pourcentage à 3 points de Nikola Jokić et que disent les rapports sur lui ? » → partie SQL correcte, partie RAG en abstention ;
- question MIXED avec abstention globale : « Compare les rebonds de Julius Randle et Nikola Jokić, puis explique ce que les rapports disent du jeu de Julius Randle. » → abstention globale si les sources ne couvrent pas correctement les deux volets.

---

## 7. Limites identifiées

Les principales limites sont :

- corpus PDF limité et composé de discussions Reddit parfois bruitées ;
- couverture inégale des joueurs et équipes ;
- statistiques Excel agrégées à la saison (pas d’analyse match par match, domicile/extérieur, quart‑temps) ;
- routeur LLM susceptible de mal router certaines questions hybrides ou ambiguës ;
- route MIXED qui ne gère pas encore de façon optimale toutes les combinaisons de réponses partielles ;
- métriques RAGAS dépendantes des références et contextes disponibles.

---

## 8. Recommandations d’évolution

1. Enrichir le corpus PDF avec des rapports de match éditorialisés et structurés.
2. Ajouter un export statistique match par match avec dates, adversaires et indicateurs domicile/extérieur.
3. Mettre en place un reranker après FAISS pour améliorer la pertinence des chunks transmis au LLM.
4. Améliorer l’orchestrateur MIXED en décomposant explicitement chaque question hybride en sous‑questions SQL et RAG, puis en acceptant des réponses partielles.
5. Étendre le jeu d’évaluation avec davantage de cas MIXED et de formulations bruitées.
6. Suivre dans Logfire des métriques spécifiques à la route MIXED (scores de similarité, taux d’abstentions partielles et totales).

---

## 9. Reproductibilité

Les principaux fichiers nécessaires à la reproduction sont :

```
requirements.txt
pyproject.toml
README.md
indexer.py
load_excel_to_db.py
evaluate_ragas.py
MistralChat.py
data/eval_dataset.jsonl
data/eval_dataset_v2.jsonl
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

Le fichier `.env` est exclu du contrôle de version et ne doit pas être inclus dans une archive de livraison.

---

## Conclusion

L’évaluation du prototype a montré les limites d’un RAG documentaire seul pour répondre aux questions chiffrées. Le système final SQL/RAG/MIXED sépare les responsabilités :

- statistiques structurées traitées de façon déterministe par SQL ;
- documents non structurés analysés par un pipeline RAG traçable ;
- questions mixtes prises en charge par une route MIXED qui combine les deux voies.

Cette architecture améliore la fiabilité opérationnelle : elle fournit des réponses chiffrées exactes lorsque les données existent, des synthèses narratives sourcées lorsque les documents sont pertinents, et des abstentions explicites lorsque les sources ne permettent pas de répondre de manière fondée. La route MIXED ouvre la voie à des interactions plus riches entre statistiques et analyse, tout en offrant un cadre clair pour les améliorations futures.