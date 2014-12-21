librnn
======
This repository contains implementation of LSTM and GRU experiments
of the paper:  
"Empirical Evaluation of Gated Recurrent Neural Networks", Chung et al., NIPS2014DLRLW.

However, we try to put both implementations of LSTM:  
1) "Generating Sequences with Recurrent Neural Networks", A. Graves  
2) "Recurrent Neural Network Regularization", W. Zaremba et al.

You can find the experimental settings in yaml files (since there have
been updates in implementation the number could be slightly different).

You need the latest version of Theano and Pylearn2.  
To give more precise information which commit hash to use:  
Theano commit: 0a0821717ea86fe33276d5770d41aaa921f6f06b  
Pylearn2 commit: 852c71fe0e83b7a4896e68babb2cfcf866697c73

GRU is currently in procces to be added on lisa-lab/pylearn2,  
hence, for the moment we cannot access the code.  
You can find the actual implementation of GRU and LSTM [here](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/rnn/models/rnn.py)
