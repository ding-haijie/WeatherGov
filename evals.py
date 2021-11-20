import nltk


class BleuScore(object):
    def __init__(self):
        self.list_gens = []
        self.list_refs = []

    def reset_gens(self):
        self.list_gens = []

    def set_refs(self, list_refs):
        self.list_refs = list_refs

    def add_gen(self, text_gen):
        self.list_gens.append(text_gen)

    def calculate(self):
        hypotheses = []
        list_of_references = []
        assert (len(self.list_refs) == len(self.list_gens))
        for text_gen, text_ref in zip(self.list_gens, self.list_refs):
            hypotheses.append(
                [token for token in text_gen.split()]
            )
            list_of_references.append(
                [
                    [token for token in text_ref.split()]
                ]
            )
        bleu_score = nltk.translate.bleu_score.corpus_bleu(
            list_of_references=list_of_references, hypotheses=hypotheses)
        return bleu_score * 100
