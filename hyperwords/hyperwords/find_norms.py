from docopt import docopt
import numpy as np
def main():
    args = docopt("""
    Usage:
        text2numpy.py <path>
    """)
    path = args['<path>']
    matrix = read_vectors(path)

def read_vectors(path):
    vectors = {}
    with open(path) as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            tokens = line.strip().split(' ')
            vect = np.asarray([float(x) for x in tokens[1:]])
            vectors[tokens[0]] = np.sqrt(np.sum(vect * vect, axis=0))
#            print vectors[tokens[0]]
    return vectors

if __name__ == '__main__':
    main()
