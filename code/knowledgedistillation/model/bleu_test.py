from scripts.bleu import *

reference = [[['This', 'is', 'irl']]]
generated = [['This', 'is', 'irl']]

bs = compute_bleu(reference, generated, max_order = 4)
print(bs)