import chromadb
from dspy.retrieve.chromadb_rm import ChromadbRM
from .MyExample import ClaimDemo

class CRMA:
    def getChroma(self):
        
        chroma_client = chromadb.PersistentClient(path="mychroma")
        
        collection = chroma_client.get_or_create_collection(name="my_collection", )
        
        return collection
    # def getRM(self):
    #     return ChromadbRM(collection_name="my_collection", persist_directory="mychroma")