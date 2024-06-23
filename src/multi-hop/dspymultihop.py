import dspy
from data import getSample
from signatures import GenerateAnswer, GenerateSearchQuery
from pipeline import SimplifiedBaleen
from data.MyExample import ClaimDemo
from data.VectorDB import CRMA
from dspy.retrieve.chromadb_rm import ChromadbRM
turbo = dspy.OpenAI(model='gpt-3.5-turbo')
#colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

CRMA().getChroma()
rmc=ChromadbRM(collection_name="my_collection", persist_directory="mychroma")

dspy.settings.configure(lm=turbo, rm=rmc)



#trainset, devset=getSample()
trainset=ClaimDemo().createExample()

# Ask any question you like to this simple RAG program.
claim_question="What happened to my claim claimh9999?"

# Get the prediction. This contains `pred.context` and `pred.answer`.
uncompiled_baleen = SimplifiedBaleen()  # uncompiled (i.e., zero-shot) program
#print(uncompiled_baleen.predictors())
uncompiled_baleen.save("multihop_uncompiledclaim.txt")
pred = uncompiled_baleen(claim_question)

# Print the contexts and the answer.
print(f"Question: {claim_question}")
print("Answer from UnCompiled Baleen")
#print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
#print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")

#turbo.inspect_history(n=3)

def validate_context_and_answer_and_hops(example, pred, trace=None):
    if not dspy.evaluate.answer_exact_match(example, pred): return False
    if not dspy.evaluate.answer_passage_match(example, pred): return False

    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True

from dspy.teleprompt import BootstrapFewShot

teleprompter = BootstrapFewShot(metric=validate_context_and_answer_and_hops)
#compiled_baleen = teleprompter.compile(SimplifiedBaleen(), teacher=SimplifiedBaleen(passages_per_hop=2), trainset=trainset)
#compiled_baleen.save("multihopdspyclaim.txt")
compiled_baleen=SimplifiedBaleen()
compiled_baleen.load("multihopdspyclaim.txt")

pred = compiled_baleen(claim_question)

# Print the contexts and the answer.
print("Answer from Compiled Baleen")
#print(f"Question: {claim_question}")
print(f"Predicted Answer: {pred.answer}")
#print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")



#turbo.inspect_history(n=3)