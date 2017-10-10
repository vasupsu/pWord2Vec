CXX = icpc
CC = icc
MPICC = mpiicpc
CPPFLAGS = -std=c++11 -qopenmp -O3 -D USE_MKL -mkl=sequential -Wall -xhost
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: pSGNScc pWord2Vec word2vec

pSGNScc: pSGNScc.cpp
	$(CXX) pSGNScc.cpp -o pSGNScc $(CPPFLAGS)
pWord2Vec: pWord2Vec.cpp
	$(CXX) pWord2Vec.cpp -o pWord2Vec $(CPPFLAGS)
word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
pWord2Vec_mpi: pWord2Vec_mpi.cpp
	$(MPICC) pWord2Vec_mpi.cpp -o pWord2Vec_mpi $(CPPFLAGS)
clean:
	rm -rf pWord2Vec pWord2Vec_mpi 
