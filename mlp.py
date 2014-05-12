'''
Naive implementation of multi-layer perceptron (feed-forward neural network) using Theano
Currently only one hidden layer with a softmax output layer.


It is important to use negative log-likelihood in the cost function instead of traditional
suqare error.

Get error rate 3.3% in MNIST (can still improve by running longer time and smaller learning rate)

by Joe Ng
'''
import cv2
import pickle
import gzip
import numpy as np
import theano
import theano.tensor as T


class MLP:

    def __init__(self, n_in, n_out, n_hidden=500):
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.learning_rate = 0.1

        # initialization copy from theano code example
        W1_init = np.random.uniform(low=-np.sqrt(6./(n_in+n_hidden)),
                                    high=np.sqrt(6./(n_in+n_hidden)),
                                    size=(n_in, n_hidden))
        W2_init = np.zeros((n_hidden, n_out))
        b1_init = np.zeros(n_hidden) # np.random.randn(n_hidden)
        b2_init = np.zeros(n_out) # np.random.randn(n_out)
        self.W1 = theano.shared(W1_init, name='W1')
        self.W2 = theano.shared(W2_init, name='W2')
        self.b1 = theano.shared(b1_init, name='b1')
        self.b2 = theano.shared(b2_init, name='b2')

        X = T.matrix('X')
        y = T.lvector('y')
        A1 = T.tanh(T.dot(X, self.W1) + self.b1)
        A2 = T.nnet.softmax(T.dot(A1, self.W2) + self.b2)
        objective = T.mean(-T.log(A2[T.arange(y.shape[0]), y]))
        dW1 = T.grad(objective, self.W1)
        dW2 = T.grad(objective, self.W2)
        db1 = T.grad(objective, self.b1)
        db2 = T.grad(objective, self.b2)
        self.output = theano.function([X], outputs=A2)
        self.get_cost = theano.function(inputs=[X, y], outputs=objective)
        self.update_iteration = theano.function([X, y], [],
                                                updates=[(self.W1, self.W1 - self.learning_rate * dW1),
                                                         (self.W2, self.W2 - self.learning_rate * dW2),
                                                         (self.b1, self.b1 - self.learning_rate * db1),
                                                         (self.b2, self.b2 - self.learning_rate * db2)])

    def train(self, X, y, valid_set, max_epoch=10000):
        per = np.random.permutation(X.shape[0])
        X, y = X[per, :], y[per] # random shuffle data
        n_batch = 2500
        mini_batch_size = X.shape[0] / n_batch
        epoch, bid = 0, 0
        while epoch < max_epoch:
            X_mini = X[mini_batch_size*bid : mini_batch_size*(bid+1), :]
            y_mini = y[mini_batch_size*bid : mini_batch_size*(bid+1)]
            self.update_iteration(X_mini, y_mini)
            epoch += 1
            bid = (bid + 1) % n_batch
            if epoch % 2500 == 0:
                print 'epoch = ', epoch, ', cost = ', self.get_cost(X, y)
                print 'validation error = ', self.test(valid_set[0], valid_set[1]) * 100

    def test(self, X, y):
        A2 = self.output(X)
        y_pred = np.argmax(A2, axis=1)
        return np.mean(y_pred != y)

    def visualize_hidden(self, filename):
        img_sz = (28, 28)
        grid_sz = (10, 10)
        W = self.W1.get_value().T
        W = W.reshape((W.shape[0], img_sz[0], img_sz[1]))
        W -= W.min()
        W *= 255 / W.max()
        img = np.zeros(((img_sz[0]+1) * grid_sz[0], (img_sz[1]+1) * grid_sz[1]))
        for i in range(grid_sz[0]):
            for j in range(grid_sz[1]):
                img[i*(img_sz[0]+1):(i+1)*(img_sz[0]+1)-1, j*(img_sz[1]+1):(j+1)*(img_sz[1]+1)-1] = W[i*grid_sz[1]+j, :, :]
        cv2.imwrite(filename, img)


def main():
    train_model = False
    train_set, valid_set, test_set = pickle.load(gzip.open('mnist.pkl.gz', 'rb'))
    if train_model:
        print '... building model'
        mlp = MLP(train_set[0].shape[1], 10)
        print '... start training'
        mlp.train(train_set[0], train_set[1], valid_set)
        pickle.dump(mlp, open('trained_mlp.pkl', 'w'))
    else:
        mlp = pickle.load(open('trained_mlp.pkl', 'r'))
    acc = mlp.test(test_set[0], test_set[1])
    print 'testing error: ', acc * 100
    mlp.visualize_hidden('mlp_feature.png')

if __name__ == '__main__':
    main()
