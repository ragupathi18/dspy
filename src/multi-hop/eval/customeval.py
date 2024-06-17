import dspy
import dspy.evaluate


class Evaluator:
    """ The predicted answer matches the gold answer.
        The retrieved context contains the gold answer.
        None of the generated queries is rambling (i.e., none exceeds 100 characters in length).
        None of the generated queries is roughly repeated (i.e., none is within 0.8 or higher F1 score of earlier queries).
    """
    def validate_context_and_answer_and_hops(example, pred, trace=None):
        if not dspy.evaluate.answer_exact_match(example, pred): return False
        if not dspy.evaluate.answer_passage_match(example, pred): return False

        dspy.evaluate.metrics.answer_exact_match

        hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

        if max([len(h) for h in hops]) > 100: return False
        if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

        return True
