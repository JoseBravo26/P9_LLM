# Etape 2b -- Branchement du routeur SQL/RAG dans l interface

## Objectif
Faire beneficier l utilisateur final du routage SQL/RAG deja construit et teste :
`MistralChat.py` n appelait jusqu ici que le pipeline RAG (recherche FAISS), sans
jamais solliciter le SQL Tool sur `database/nba_analytics.db`. Ce lot corrige ce point.

## Fichiers modifies ou ajoutes
- `utils/rag_pipeline_router.py` (mis a jour) : ajoute une fonction unique `answer(question)`
  qui route puis repond, avec tracage Logfire a chaque etape (classification, branche SQL,
  branche RAG) et repli automatique sur RAG si la classification echoue.
- `MistralChat.py` (remplace) : appelle desormais `answer(question)` au lieu d effectuer
  lui-meme la recherche FAISS et l appel Mistral. Affiche un badge indiquant la source
  utilisee (📊 base SQL ou 📄 rapports PDF) et les identifiants de chunks cites si le RAG
  a ete utilise.
- `tests/test_router.py` (ajoute) : verifie l abstention en cas d erreur SQL et le repli
  sur RAG en cas d echec du classifieur, sans appel reseau reel (mocks).

## Correction technique importante
La version precedente de `MistralChat.py` construisait le contexte avec
`result.metadata.get("source", ...)`, alors que `RetrievedChunk.metadata` est un objet
Pydantic `SourceMetadata`, pas un dictionnaire. Cet appel aurait leve une `AttributeError`
a l execution. La nouvelle version ne reconstruit plus ce contexte manuellement dans
l interface : c est `utils/rag_pipeline.py` qui s en charge deja correctement en interne.

## Fonctionnement du routage
```
Question utilisateur
        |
        v
route_question(question)   -> agent Pydantic AI dedie, sortie "SQL" ou "RAG"
        |
   +----+----+
   |         |
  SQL       RAG
   |         |
answer_with_sql   answer_with_rag
   |         |
execute_sql()   answer_question() [utils/rag_pipeline.py existant]
   |         |
synthese LLM   reponse structuree AssistantAnswer
   |         |
   +----+----+
        |
        v
AssistantAnswer affiche dans Streamlit
```

En cas d erreur de classification (reseau, quota, etc.), le routeur se replie
automatiquement sur la branche RAG plutot que de bloquer l utilisateur.

## Installation
Aucune nouvelle dependance : ce lot reutilise `pydantic-ai`, `langchain-mistralai`,
`streamlit` et `logfire` deja presents dans `requirements.txt`.

## Execution
```bash
# Verifier le routage avant de lancer l interface
pytest -q tests/test_router.py

# Lancer l assistant avec le routage actif
streamlit run MistralChat.py
```

## Verification manuelle recommandee
| Question | Route attendue | Verification |
|---|---|---|
| Quel est le % a 3 points de Nikola Jokic ? | SQL | Badge "📊 Statistiques", reponse chiffree exacte |
| Compare les rebonds de Jokic et Towns | SQL | Badge SQL, tableau de valeurs synthetise |
| Que disent les rapports sur la defense de Denver ? | RAG | Badge "📄 Analyse documentaire", chunks cites |
| Meilleur 3P% sur les 5 derniers matchs ? | SQL | Abstention explicite (message st.info) |
| Compare les rebonds a domicile et a l exterieur | SQL | Abstention explicite (message st.info) |

## Limite residuelle
Le classifieur SQL/RAG est un agent LLM zero-shot sur des libelles simples ; il peut
occasionnellement mal router une question ambigue (ex. une question mixte chiffree et
narrative). Le README de l etape 3 (evaluation continue) devra inclure des cas de test
specifiques au routage, en plus des categories deja definies dans `data/eval_dataset.jsonl`.
