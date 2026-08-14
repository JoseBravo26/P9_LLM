# Étape 1 — Fiabilisation et évaluation RAG

## Installation
```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```
Renseigner `MISTRAL_API_KEY` dans `.env`. Configurer Logfire selon votre espace de travail pour visualiser les traces.

## Exécution
```bash
pytest -q
python indexer.py
python evaluate_ragas.py
```
Les résultats détaillés sont écrits dans `reports/ragas_results.csv` et la synthèse par catégorie dans `reports/ragas_results.md`.

## Choix techniques
- Pydantic impose un contrat sur les documents, chunks, requêtes, résultats et jeu d'évaluation.
- Pydantic AI force une réponse LLM structurée : texte, chunks cités, niveau de confiance et abstention.
- Logfire trace la requête, le nombre de chunks et l'appel Pydantic AI.
- RAGAS mesure `faithfulness`, `response_relevancy`, `context_precision` et `context_recall`.
- Aucun vecteur nul n'est créé si un embedding échoue : le processus s'arrête pour ne jamais dégrader l'index silencieusement.

## Limite métier documentée
Le classeur fourni regroupe des statistiques saisonnières par joueur. Il ne permet pas de calculer les cinq derniers matchs ni les écarts domicile/extérieur. Ces questions doivent entraîner une abstention ; en production elles seront routées vers une source match-par-match Pandas/SQL.
