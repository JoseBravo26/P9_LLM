"""
Évaluation RAGAS — version finale hybride SQL/RAG.

Cette version appelle utils.rag_pipeline_router.answer(), comme l'application
Streamlit. Les questions chiffrées sont routées vers SQL et les questions
narratives sont routées vers le RAG FAISS.

Exemples :
# Test de 5 cas sans lancer les métriques RAGAS
python evaluate_ragas.py --limit 5 --skip-ragas --run-name test_final_5cas

# Évaluation complète avec les quatre métriques
python evaluate_ragas.py --run-name final_sql_rag_42cas --full-metrics... # Reprendre le même run à partir du checkpoint
python evaluate_ragas.py --run-name final_sql_rag_42cas --full-metrics
"""

import argparse
import json
import random
import time
from datetime import datetime
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

HISTORY_DIR = Path("reports") / "Historique"

RAGAS_RUN_CONFIG = RunConfig(
    max_workers=1,
    max_retries=8,
    max_wait=60,
    timeout=180,
)


def is_rate_limit_error(error: Exception) -> bool:
    """Détecte les erreurs Mistral de type HTTP 429."""
    message = str(error).lower()
    return (
        "429" in message
        or "rate limit" in message
        or "rate_limited" in message
    )


def load_cases(dataset_path: Path, limit: int | None = None) -> list[EvaluationCase]:
    """Charge et valide toutes les lignes JSONL du dataset."""
    cases = [
        EvaluationCase.model_validate(json.loads(line))
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    return cases[:limit] if limit is not None else cases


def load_checkpoint(checkpoint_path: Path) -> dict[str, dict]:
    """Charge les réponses déjà générées, indexées par identifiant de cas."""
    if not checkpoint_path.exists():
        return {}

    rows_by_id: dict[str, dict] = {}

    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        if row.get("id"):
            rows_by_id[row["id"]] = row

    return rows_by_id


def append_checkpoint(checkpoint_path: Path, row: dict) -> None:
    """Enregistre immédiatement chaque réponse pipeline."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with checkpoint_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_router_with_retry(question: str, max_attempts: int = 8) -> dict:
    """Appelle le routeur avec retry exponentiel uniquement lors d'un 429."""
    for attempt in range(max_attempts):
        try:
            return answer_with_router(question)

        except Exception as error:
            if not is_rate_limit_error(error) or attempt == max_attempts - 1:
                raise

            wait_seconds = min(
                90,
                (2 ** attempt) + random.uniform(0.5, 1.5),
            )

            print(
                f"[429] Limitation Mistral. Nouvelle tentative dans "
                f"{wait_seconds:.1f} s ({attempt + 1}/{max_attempts})."
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Nombre maximal de tentatives atteint.")


def extract_contexts(route: str, result: dict) -> list[str]:
    """
    Prépare les contextes transmis à RAGAS.

    - Route RAG : texte réel des chunks récupérés.
    - Route SQL : réponse construite depuis le résultat SQL. Le routeur ne
      remonte pas encore les lignes SQL brutes ; cette réponse est donc le
      meilleur contexte disponible dans l'interface actuelle.
    """
    if route == "RAG":
        return [
            f"[chunk_id={chunk.id}; source={chunk.metadata.source}]\n{chunk.text}"
            for chunk in result.get("contexts", [])
        ]

    response = result["response"]
    return [response.answer] if response.answer else []


def build_row(case: EvaluationCase, result: dict) -> dict:
    """Construit une ligne de résultat stable et sérialisable."""
    route = result["route"]
    response = result["response"]
    contexts = extract_contexts(route, result)

    return {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "reference_answer": case.reference_answer,
        "answer": response.answer,
        "route": route,
        "abstained": response.abstained,
        "expected_abstention": case.expected_abstention,
        "abstention_correcte": response.abstained == case.expected_abstention,
        "confidence": response.confidence,
        "cited_chunk_ids": json.dumps(
            response.cited_chunk_ids,
            ensure_ascii=False,
        ),
        "contexts": json.dumps(contexts, ensure_ascii=False),
        "reference_contexts": json.dumps(
            case.reference_contexts,
            ensure_ascii=False,
        ),
    }


def generate_rows(
    cases: list[EvaluationCase],
    checkpoint_path: Path,
    pause_seconds: float,
) -> list[dict]:
    """Produit les réponses du routeur ou reprend celles du checkpoint."""
    rows_by_id = load_checkpoint(checkpoint_path)

    existing_count = sum(case.id in rows_by_id for case in cases)

    print(
        f"Checkpoint : {existing_count}/{len(cases)} réponse(s) "
        f"déjà disponible(s)."
    )

    for position, case in enumerate(cases, start=1):
        if case.id in rows_by_id:
            print(f"[{position}/{len(cases)}] Déjà traité : {case.id}")
            continue

        print(f"[{position}/{len(cases)}] Traitement : {case.id}")

        router_result = call_router_with_retry(case.question)
        row = build_row(case, router_result)

        append_checkpoint(checkpoint_path, row)
        rows_by_id[case.id] = row

        if pause_seconds > 0 and position < len(cases):
            time.sleep(pause_seconds)

    missing_ids = [case.id for case in cases if case.id not in rows_by_id]

    if missing_ids:
        raise RuntimeError(
            "Checkpoint incomplet : " + ", ".join(missing_ids)
        )

    return [rows_by_id[case.id] for case in cases]


def build_samples(rows: list[dict]) -> list[SingleTurnSample]:
    """Convertit les lignes checkpointées en échantillons RAGAS."""
    samples: list[SingleTurnSample] = []

    for row in rows:
        contexts = json.loads(row["contexts"])
        reference_contexts = json.loads(row["reference_contexts"])

        samples.append(
            SingleTurnSample(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=contexts if contexts else [""],
                reference=row["reference_answer"],
                reference_contexts=reference_contexts,
            )
        )

    return samples


def evaluate_with_ragas(rows: list[dict], full_metrics: bool) -> pd.DataFrame:
    """Lance RAGAS séquentiellement pour limiter les erreurs 429."""
    judge_llm = LangchainLLMWrapper(
        ChatMistralAI(
            model=MODEL_NAME,
            mistral_api_key=MISTRAL_API_KEY,
            temperature=0,
        )
    )

    judge_embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
    )

    metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(
            llm=judge_llm,
            embeddings=judge_embeddings,
        ),
    ]

    if full_metrics:
        metrics.extend(
            [
                LLMContextPrecisionWithReference(llm=judge_llm),
                LLMContextRecall(llm=judge_llm),
            ]
        )

    print(
        "Métriques RAGAS : "
        + ", ".join(type(metric).__name__ for metric in metrics)
    )

    result = evaluate(
        EvaluationDataset(samples=build_samples(rows)),
        metrics=metrics,
        run_config=RAGAS_RUN_CONFIG,
        show_progress=True,
    )

    return result.to_pandas()


def save_reports(
    rows: list[dict],
    scores: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """Écrit le CSV détaillé et les synthèses Markdown."""
    report = pd.DataFrame(rows)

    if scores is not None:
        scores = scores.drop(
            columns=[
                "user_input",
                "response",
                "retrieved_contexts",
                "reference",
            ],
            errors="ignore",
        )

        report = pd.concat(
            [
                report.reset_index(drop=True),
                scores.reset_index(drop=True),
            ],
            axis=1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False, encoding="utf-8-sig")

    metric_columns = [
        column
        for column in [
            "faithfulness",
            "answer_relevancy",
            "llm_context_precision_with_reference",
            "context_recall",
            "abstention_correcte",
        ]
        if column in report.columns
    ]

    summary = (
        report.groupby(["category", "route"])[metric_columns]
        .mean()
        .round(3)
    )

    summary_path = output_path.with_suffix(".md")
    summary.to_markdown(summary_path)

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

    abstention_path = output_path.with_name(
        f"{output_path.stem}_abstentions.md"
    )
    abstention_summary.to_markdown(abstention_path)

    route_summary = (
        report.groupby("route")
        .agg(
            nombre_questions=("id", "count"),
            taux_abstention_correcte=("abstention_correcte", "mean"),
        )
        .round(3)
    )

    route_path = output_path.with_name(
        f"{output_path.stem}_routes.md"
    )
    route_summary.to_markdown(route_path)

    print("\nRapports créés :")
    print(f"- CSV détaillé : {output_path}")
    print(f"- Synthèse RAGAS : {summary_path}")
    print(f"- Synthèse abstention : {abstention_path}")
    print(f"- Synthèse par route : {route_path}")


def build_default_output(run_name: str) -> Path:
    """Construit un nom unique et daté dans reports/Historique/."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = run_name.strip().replace(" ", "_") or "final_sql_rag"

    return HISTORY_DIR / f"{safe_name}_{timestamp}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation RAGAS de la version finale SQL/RAG."
    )

    parser.add_argument(
        "--dataset",
        default="data/eval_dataset_v2.jsonl",
        help="Chemin du dataset JSONL.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Chemin CSV optionnel. Sans cette option, un nom daté est créé "
            "dans reports/Historique/."
        ),
    )

    parser.add_argument(
        "--run-name",
        default="final_sql_rag_text_numeric",
        help="Préfixe des rapports historisés.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre maximal de cas, utile pour un test court.",
    )

    parser.add_argument(
        "--pause",
        type=float,
        default=1.5,
        help="Pause entre deux réponses du pipeline.",
    )

    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Génère les réponses et le checkpoint sans appeler le juge RAGAS.",
    )

    parser.add_argument(
        "--full-metrics",
        action="store_true",
        help="Active les quatre métriques RAGAS.",
    )

    args = parser.parse_args()

    if not MISTRAL_API_KEY:
        raise RuntimeError(
            "MISTRAL_API_KEY est absente : vérifiez le fichier .env."
        )

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {dataset_path}"
        )

    cases = load_cases(dataset_path, args.limit)

    if not cases:
        raise RuntimeError("Le dataset est vide.")

    output_path = (
        Path(args.output)
        if args.output
        else build_default_output(args.run_name)
    )

    checkpoint_path = output_path.with_name(
        f"{output_path.stem}_checkpoint.jsonl"
    )

    print(f"Dataset : {dataset_path}")
    print("Version : finale — routeur SQL/RAG")
    print(f"Cas évalués : {len(cases)}")
    print(f"Rapport CSV : {output_path}")
    print(f"Checkpoint : {checkpoint_path}")

    rows = generate_rows(
        cases=cases,
        checkpoint_path=checkpoint_path,
        pause_seconds=args.pause,
    )

    if args.skip_ragas:
        save_reports(rows, scores=None, output_path=output_path)
        print(
            "\nMode --skip-ragas : aucune métrique RAGAS n'a été exécutée."
        )
        return

    scores = evaluate_with_ragas(
        rows=rows,
        full_metrics=args.full_metrics,
    )

    save_reports(
        rows=rows,
        scores=scores,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()