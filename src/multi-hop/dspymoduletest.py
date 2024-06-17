import dspy
from data import getSample
from signatures import GenerateAnswer, GenerateSearchQuery
from pipeline import SimplifiedBaleen
from data.MyExample import ClaimDemo
from data.VectorDB import CRMA
from dspy.retrieve.chromadb_rm import ChromadbRM
turbo = dspy.OpenAI(model='gpt-3.5-turbo')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

CRMA().getChroma()
rmc=ChromadbRM(collection_name="my_collection", persist_directory="mychroma")
dspy.settings.configure(lm=turbo, rm=rmc)

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Pass signature to ProgramOfThought Module
def potexample():
    pot = dspy.ProgramOfThought(GenerateAnswer)

    #Call the ProgramOfThought module on a particular input
    question = 'Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?'
    result = pot(question=question)

    print(f"Question: {question}")
    print(f"Final Predicted Answer (after ProgramOfThought process): {result.answer}")
    print(result)
    turbo.inspect_history(n=3)
def cothintexample():
    pot = dspy.ChainOfThoughtWithHint(GenerateAnswer)

    #Call the ProgramOfThought module on a particular input
    question = 'What is sunset time today?'
    hint="You live in Sydney"
    result = pot(question=question, hint=hint)

    print(f"Question: {question}")
    print(f"Final Predicted Answer : {result.answer}")
    print(result)
    turbo.inspect_history(n=3)
def multichainexample():#Needs to test this
    pot = dspy.MultiChainComparison(GenerateAnswer)

    #Call the ProgramOfThought module on a particular input
    question = 'What is sunset time today?'
    hint="You live in Sydney"
    result = pot(question=question, hint=hint)

    print(f"Question: {question}")
    print(f"Final Predicted Answer : {result.answer}")
    print(result)
    turbo.inspect_history(n=3)
    
def reactexample():
    # Define a simple signature for basic question answering
    class BasicQA(dspy.Signature):
        """Answer questions with short factoid answers."""
        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")

    # Pass signature to ReAct module
    react_module = dspy.ReAct(BasicQA,num_results=1)
    # Call the ReAct module on a particular input
    #question = "What is the current age of the captain who won the first ICC mens worldcup for india"
    
    question ="What is the status of claim claimh9999?"
    result = react_module(question=question)

    

    print(f"Question: {question}")
    print(f"Final Predicted Answer (after ReAct process): {result.answer}")
    print(result)
    turbo.inspect_history(n=3)
if __name__=="__main__":
    reactexample()