import dspy
class GenerateAnswer(dspy.Signature):
    """Answer the questions from non reversed claim."""

    context = dspy.InputField(desc="may contain claim information status")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often in few sentences")

class GenerateSearchQuery(dspy.Signature):
    """Retrieve the claim status from """

class GenerateSearchQuery(dspy.Signature):
    """ReWrite the question with rekeyed claimid only if rekeyed claimid found context"""
    context = dspy.InputField(desc="may contain claim status and rekeyed clainid")
    #context=GenerateAnswer.signature.context
    question = dspy.InputField()
    query = dspy.OutputField()