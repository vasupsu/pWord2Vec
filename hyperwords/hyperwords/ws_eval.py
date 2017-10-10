from docopt import docopt
from scipy.stats.stats import spearmanr

from representations.representation_factory import create_representation


def main():
    args = docopt("""
    Usage:
        ws_eval.py [options] <representation> <representation_path> <task_path>
    
    Options:
        --neg NUM    Number of negative samples; subtracts its log from PMI (only applicable to PPMI) [default: 1]
        --w+c        Use ensemble of word and context vectors (not applicable to PPMI)
        --eig NUM    Weighted exponent of the eigenvalue matrix (only applicable to SVD) [default: 0.5]
    """)
    
    data = read_test_set(args['<task_path>'])
    representation = create_representation(args)
    correlation = evaluate(representation, data)
    print 'Word Similarity', '\t%0.3f' % correlation


def read_test_set(path):
    test = []
    with open(path) as f:
        for line in f:
            x, y, sim = line.strip().lower().split()
            test.append(((x, y), sim))
    return test 


def evaluate(representation, data):
    results = []
    scores=[]
    i=0
    for (x, y), sim in data:
        results.append((representation.similarity(x, y), sim))
#        print (x, y, representation.similarity(x, y), sim)
        scores.append ((i, representation.similarity(x, y)))
        i = i + 1
    scores.sort(key=lambda tup: tup[1]) 
#    print  (scores)
    i=0
    scores_new=[]
    for L in scores:
        scores_new.append ((L[0], i))
        i = i + 1
    scores_new.sort(key=lambda tup: tup[0]) 
    actual, expected = zip(*results)
    return spearmanr(actual, expected)[0]



if __name__ == '__main__':
    main()
