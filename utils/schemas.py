from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
class SourceMetadata(BaseModel):
 model_config=ConfigDict(extra='forbid'); source:str=Field(min_length=1); filename:str=Field(min_length=1); category:str='root'; sheet:str|None=None; full_path:str|None=None; chunk_id_in_doc:int|None=Field(None,ge=0); start_index:int|None=Field(None,ge=0)
class SourceDocument(BaseModel):
 model_config=ConfigDict(extra='forbid'); page_content:str=Field(min_length=1); metadata:SourceMetadata
 @field_validator('page_content')
 @classmethod
 def clean(cls,v):
  v=' '.join(v.split())
  if not v: raise ValueError('Document vide après nettoyage.')
  return v
class Chunk(BaseModel):
 model_config=ConfigDict(extra='forbid'); id:str=Field(min_length=1); text:str=Field(min_length=1); metadata:SourceMetadata
class RetrievedChunk(Chunk): score:float=Field(ge=-1,le=1)
class RAGQuery(BaseModel):
 model_config=ConfigDict(extra='forbid'); question:str=Field(min_length=3,max_length=1000); top_k:int=Field(5,ge=1,le=20)
 @field_validator('question')
 @classmethod
 def normalize(cls,v): return ' '.join(v.split())
class AssistantAnswer(BaseModel):
 answer:str=Field(min_length=1); cited_chunk_ids:list[str]=Field(default_factory=list); confidence:Literal['high','medium','low']; abstained:bool=False
class EvaluationCase(BaseModel):
 model_config=ConfigDict(extra='forbid'); id:str; category:Literal['simple','complexe','bruitee','hors_perimetre','non_repondable']; question:str; reference_answer:str; reference_contexts:list[str]=Field(default_factory=list); expected_abstention:bool=False
class PipelineResult(BaseModel): query:RAGQuery; contexts:list[RetrievedChunk]; response:AssistantAnswer
