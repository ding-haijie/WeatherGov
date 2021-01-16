import nltk
from sklearn.metrics import f1_score


class F1Score(object):
    def __init__(self):
        self.list_golds = []
        self.list_aligns = []

    def reset_aligns(self):
        self.list_aligns = []

    def set_golds(self, list_golds):
        self.list_golds = list_golds

    def add_align(self, align):
        self.list_aligns.append(align)

    def calculate(self):
        assert (len(self.list_golds) == len(self.list_aligns))  # sanity check
        macro_f1, micro_f1 = 0., 0.
        for golds, aligns in zip(self.list_golds, self.list_aligns):
            macro_f1 += f1_score(y_true=golds, y_pred=aligns, average='macro')
            micro_f1 += f1_score(y_true=golds, y_pred=aligns, average='micro')
        macro_f1 /= len(self.list_golds)
        micro_f1 /= len(self.list_golds)

        return macro_f1 * 100, micro_f1 * 100


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
