|                           |   faithfulness |   answer_relevancy |   llm_context_precision_with_reference |   context_recall |   abstention_correcte |
|:--------------------------|---------------:|-------------------:|---------------------------------------:|-----------------:|----------------------:|
| ('bruitee', 'SQL')        |           1    |              0.655 |                                  0.714 |            0.714 |                   1   |
| ('complexe', 'SQL')       |           1    |              0.783 |                                  0.889 |            0.689 |                   1   |
| ('hors_perimetre', 'RAG') |           0    |              0     |                                  0.167 |            0.667 |                   1   |
| ('hors_perimetre', 'SQL') |           1    |              0     |                                  1     |            1     |                   0   |
| ('non_repondable', 'SQL') |           1    |              0     |                                  1     |            1     |                   0.5 |
| ('simple', 'SQL')         |           1    |              0.951 |                                  1     |            0.938 |                   1   |
| ('textuelle', 'RAG')      |           0.88 |              0.878 |                                  0.915 |            0.875 |                   1   |