import dspy
from typing import List, Union, Optional
from .VectorDB import CRMA
class MyDSPyRMClient_notneeded(dspy.Retrieve):
    def __init__(self, url:str, port:int=None, k:int=3):
        super().__init__(k=k)
        self.url=url
        
        self.retriever=CRMA().getRM()
    def rm(self):    
        return self.retriever
    def forward(self, query_or_queries: str,  k: Optional[int])  -> dspy.Prediction:
        resp=self.retriever.query(query_texts=query_or_queries,n_results=k)
        
        return dspy.Prediction(passages=resp["documents"][0])
if __name__=="__main__":
    res=MyDSPyRMClient("test").forward('What happned to my claim124?',2)
    print(res)