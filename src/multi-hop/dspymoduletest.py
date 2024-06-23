import dspy
import dspy.evaluate
from data import getSample
from signatures import GenerateAnswer, GenerateSearchQuery
from pipeline import SimplifiedBaleen
from data.MyExample import ClaimDemo
from data.VectorDB import CRMA
from dspy.retrieve.chromadb_rm import ChromadbRM
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot
turbo = dspy.OpenAI(model='gpt-3.5-turbo')
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

collection=CRMA().getChroma()
docs,ids=ClaimDemo().getData()
collection.add(documents=docs, ids=ids)
rmc=ChromadbRM(collection_name="my_collection", persist_directory="mychroma")
dspy.settings.configure(lm=turbo, rm=rmc)

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often in one sentence")

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
        """You continue to search for resubitted claims"""
        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")
    trainset=ClaimDemo().createExample()
    # Pass signature to ReAct module
    react_module = dspy.ReAct(BasicQA,num_results=1)
    
    opt=BootstrapFewShot( metric=dspy.evaluate.answer_exact_match )

    compiled_react=opt.compile(react_module, trainset=trainset)
    # Call the ReAct module on a particular input
    question = "What is the current age of the captain who won the first ICC mens worldcup for india"
    
    question ="What happened to claim claimh2512?"
    result = react_module(question=question)
    react_module.save("reactuncompiled.txt")
    print(f"Question: {question}")
    print(f"Final Predicted Answer (after ReAct process): {result.answer}")
    #turbo.inspect_history(n=3)
    result=compiled_react(question=question)
    compiled_react.save("reactcompiled.txt")
    print(f"Final Predicted Answer (after Compiled ReAct process): {result.answer}")

    
    turbo.inspect_history(n=3)


    config = dict(num_threads=8, display_progress=True, display_table=5)
    evaluate = Evaluate(devset=devset, metric=dspy.evaluate.answer_exact_match, **config)

    evaluate(agent)
if __name__=="__main__":
    reactexample()