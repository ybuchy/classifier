# classifier

This is an implementation of a fully connected neural network with variable activation functions, hidden layer amount, layer sizes and optional bias. It does use softmax loss (softmax + categorical cross entropy) for classification. There is also a preprocessor written for the mnist database coming from yann lecuns website. It's implemented to be able to do batch processing (mini batch gradient descent) and use a validation set to stop training depending on it's accuracy.

Currently at ~98% accuracy on mnist
