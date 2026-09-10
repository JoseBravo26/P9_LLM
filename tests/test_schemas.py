import pytest
from pydantic import ValidationError
from utils.schemas import RAGQuery,SourceDocument
def test_normalisation(): assert RAGQuery(question='  Quel   joueur ? ').question=='Quel joueur ?'
def test_document_vide_refuse():
 with pytest.raises(ValidationError): SourceDocument(page_content='  ',metadata={'source':'x','filename':'x.txt'})
def test_top_k_refuse():
 with pytest.raises(ValidationError): RAGQuery(question='Question valide',top_k=0)
