"""Évaluation RAGAS résiliente du routeur SQL/RAG.

Exécution :
    python evaluate_ragas_resilient.py --dataset data/eval_dataset_v2.jsonl

Fonctionnalités :
- réessaie les réponses du pipeline en cas de rate limit Mistral (HTTP 429) ;
- crée un checkpoint JSONL après chaque question traitée ;
- reprend automatiquement les questions déjà enregistrées après un arrêt ;
- limite la concurrence de RAGAS pour réduire les erreurs 429 ;
- produit un rapport détaillé, des moyennes par catégorie/route et un
  tableau de contrôle des abstentions.
"""
import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from utils.config import MISTRAL_API_KEY, MODEL_NAME
from utils.rag_pipeline_router import answer as answer_with_router
from utils.schemas import EvaluationCase

RAGAS_RUN_CONFIG = RunConfig(
    max_workers=1,
    max_retries=10,
    max_wait=60,
    timeout=180,
)


class PipelineRateLimitError(RuntimeError):
    """Erreur levée lorsqu'un appel pipeline reste limité après les retries."""


def is_rate_limit_error(error: Exception) -> bool:
    """Détecte les variantes d'erreur 429 renvoyées par les bibliothèques Mistral."""
    message = str(error).lower()
    return "429" in message or "rate limit" in message or "rate_limited" in message


def call_router_with_retry(question: str, max_attempts: int = 8) -> dict:
    """Appelle le routeur avec backoff exponentiel et jitter en cas de 429."""
    for attempt in range(max_attempts):
        try:
            return answer_with_router(question)
        except Exception as error:
            last_attempt = attempt == max_attempts - 1
            if not is_rate_limit_error(error):
                raise
            if last_attempt:
                raise PipelineRateLimitError(
                    f"Rate limit Mistral persistant après {max_attempts} tentatives."
                ) from error

            wait_seconds = min(90, (2**attempt) + random.uniform(0, 1))
            print(
                f"[429] Routeur limité pour : {question!r}\n"
                f"      Nouvelle tentative dans {wait_seconds:.1f} s "
                f"({attempt + 1}/{max_attempts})."
            )
            time.sleep(wait_seconds)

    raise AssertionError("Cette ligne ne doit jamais être atteinte.")


def extract_contexts(route: str, result: dict) -> list[str]:
    """Retourne les contextes réellement utilisés par chaque branche."""
    if route == "RAG":
        chunks = result.get("contexts", [])
        return [f"[chunk_id={chunk.id}] {chunk.text}" for chunk in chunks]

    response = result["response"]
    return [response.answer] if response.answer else []


def load_cases(dataset_path: Path) -> list[EvaluationCase]:
    return [
        EvaluationCase.model_validate(json.loads(line))
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_checkpoint(checkpoint_path: Path) -> dict[str, dict]:
    """Charge les réponses déjà calculées afin de reprendre après interruption."""
    if not checkpoint_path.exists():
        return {}

    completed: dict[str, dict] = {}
    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            completed[row["id"]] = row
    return completed


def append_checkpoint(checkpoint_path: Path, row: dict) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_row(case: EvaluationCase, router_result: dict) -> dict:
    route = router_result["route"]
    response = router_result["response"]
    contexts = extract_contexts(route, router_result)

    return {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "reference_answer": case.reference_answer,
        "reference_contexts": case.reference_contexts,
        "answer": response.answer,
        "route": route,
        "abstained": response.abstained,
        "expected_abstention": case.expected_abstention,
        "abstention_correcte": response.abstained == case.expected_abstention,
        "confidence": response.confidence,
        "contexts": contexts,
    }


def evaluate_rows(rows: list[dict], output_path: Path) -> pd.DataFrame:
    """Évalue les réponses récupérées avec RAGAS et sauvegarde les rapports."""
    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=row["contexts"] or [""],
            reference=row["reference_answer"],
            reference_contexts=row["reference_contexts"],
        )
        for row in rows
    ]

    evaluator_llm = LangchainLLMWrapper(
        ChatMistralAI(
            model=MODEL_NAME,
            mistral_api_key=MISTRAL_API_KEY,
            temperature=0,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
    )

    scores = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(llm=evaluator_llm),
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            LLMContextPrecisionWithReference(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
        ],
        run_config=RAGAS_RUN_CONFIG,
    ).to_pandas()

    report_rows = []
    for row in rows:
        report_rows.append(
            {
                **row,
                "contexts": json.dumps(row["contexts"], ensure_ascii=False),
                "reference_contexts": json.dumps(
                    row["reference_contexts"], ensure_ascii=False
                ),
            }
        )

    report = pd.concat(
        [
            pd.DataFrame(report_rows),
            scores.drop(
                columns=["user_input", "response", "retrieved_contexts", "reference"],
                errors="ignore",
            ),
        ],
        axis=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False, encoding="utf-8-sig")

    metrics = [
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_with_reference",
        "context_recall",
        "abstention_correcte",
    ]
    by_category_route = report.groupby(["category", "route"])[metrics].mean().round(3)
    by_category_route.to_markdown(output_path.with_suffix(".md"))

    abstention_summary = (
        report.groupby("category")
        .agg(
            nombre_questions=("id", "count"),
            abstentions_attendues=("expected_abstention", "sum"),
            abstentions_observees=("abstained", "sum"),
            taux_abstention_correcte=("abstention_correcte", "mean"),
        )
        .round(3)
    )
    abstention_path = output_path.with_name(f"{output_path.stem}_abstentions.md")
    abstention_summary.to_markdown(abstention_path)

    return report


def run(dataset: str, output: str, evaluate_ragas: bool = True) -> None:
    if not MISTRAL_API_KEY:
        raise RuntimeError("Ajoutez MISTRAL_API_KEY dans le fichier .env.")

    dataset_path = Path(dataset)
    output_path = Path(output)
    checkpoint_path = output_path.with_name(f"{output_path.stem}_checkpoint.jsonl")

    cases = load_cases(dataset_path)
    completed = load_checkpoint(checkpoint_path)
    print(f"Checkpoint : {len(completed)}/{len(cases)} réponses déjà disponibles.")

    for index, case in enumerate(cases, start=1):
        if case.id in completed:
            print(f"[{index}/{len(cases)}] Déjà traité : {case.id}")
            continue

        print(f"[{index}/{len(cases)}] Traitement : {case.id}")
        router_result = call_router_with_retry(case.question)
        row = build_row(case, router_result)
        append_checkpoint(checkpoint_path, row)
        completed[case.id] = row

        # Petite pause entre deux questions pour lisser le débit vers Mistral.
        time.sleep(1.5)

    rows = [completed[case.id] for case in cases if case.id in completed]
    if len(rows) != len(cases):
        raise RuntimeError(
            f"Checkpoint incomplet : {len(rows)}/{len(cases)} réponses disponibles."
        )

    if not evaluate_ragas:
        print(f"Réponses pipeline enregistrées dans : {checkpoint_path}")
        return

    print("Démarrage de l'évaluation RAGAS (concurrence limitée à 1 worker)...")
    evaluate_rows(rows, output_path)
    print(f"Rapport détaillé : {output_path}")
    print(f"Moyennes par catégorie et route : {output_path.with_suffix('.md')}")
    print(
        "Contrôle des abstentions : "
        f"{output_path.with_name(f'{output_path.stem}_abstentions.md')}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval_dataset_v2.jsonl")
    parser.add_argument("--output", default="reports/ragas_final_v2.csv")
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Génère uniquement les réponses et le checkpoint, sans les métriques RAGAS.",
    )
    args = parser.parse_args()
    run(args.dataset, args.output, evaluate_ragas=not args.skip_ragas)
