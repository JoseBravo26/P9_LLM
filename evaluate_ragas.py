"""Exécution : python evaluate_ragas.py --dataset data/eval_dataset.jsonl"""
import argparse,json
from pathlib import Path
import pandas as pd
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset,SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness,ResponseRelevancy,LLMContextPrecisionWithReference,LLMContextRecall
from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from utils.config import MISTRAL_API_KEY,MODEL_NAME
from utils.rag_pipeline import answer_question
from utils.schemas import EvaluationCase
def run(dataset,output):
 if not MISTRAL_API_KEY: raise RuntimeError('Ajoutez MISTRAL_API_KEY dans .env.')
 cases=[EvaluationCase.model_validate(json.loads(x)) for x in Path(dataset).read_text(encoding='utf-8').splitlines() if x]
 rows=[]; samples=[]
 for c in cases:
  r=answer_question(c.question); ctx=[x.text for x in r.contexts]
  rows.append({'id':c.id,'category':c.category,'question':c.question,'reference_answer':c.reference_answer,'answer':r.response.answer,'abstained':r.response.abstained,'expected_abstention':c.expected_abstention,'contexts':json.dumps(ctx,ensure_ascii=False)})
  samples.append(SingleTurnSample(user_input=c.question,response=r.response.answer,retrieved_contexts=ctx,reference=c.reference_answer,reference_contexts=c.reference_contexts))
 llm=LangchainLLMWrapper(ChatMistralAI(model=MODEL_NAME,mistral_api_key=MISTRAL_API_KEY,temperature=0)); emb=LangchainEmbeddingsWrapper(MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY))
 score=evaluate(EvaluationDataset(samples=samples),metrics=[Faithfulness(llm=llm),ResponseRelevancy(llm=llm,embeddings=emb),LLMContextPrecisionWithReference(llm=llm),LLMContextRecall(llm=llm)]).to_pandas()
 report=pd.concat([pd.DataFrame(rows),score.drop(columns=['user_input','response','retrieved_contexts','reference'],errors='ignore')],axis=1); output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); report.to_csv(output,index=False)
 report.groupby('category')[report.select_dtypes('number').columns].mean().round(3).to_markdown(output.with_suffix('.md')); print(f'Rapports créés : {output} et {output.with_suffix(".md")}')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='data/eval_dataset.jsonl');p.add_argument('--output',default='reports/ragas_results.csv');a=p.parse_args();run(a.dataset,a.output)
