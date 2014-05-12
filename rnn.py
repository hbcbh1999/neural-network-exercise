'''
Simple recurrent neural network using back propagation through time (BPTT). It
tries to predict the next step along a 2D straight line. It works reasonably well 
for time sequence with less than 10 steps. For capturing longer term memory, 
Hessian-Free optimization or Echo State Network can be tried. Different 
initialization strategies and training with momentum can be tried as well.

by Joe Ng
'''
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

theano.config.exception_verbosity = 'high'
mode = theano.Mode(linker='cvm')
#mode = 'DEBUG_MODE'


class RNN:

    def __init__(self, n_in, n_out, n_hidden, learning_rate=0.001, learning_rate_decay=0.999):
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden

        self.W_in = theano.shared(np.random.uniform(-.01, .01, size=(n_in, n_hidden)), name='W_in')
        self.W_out = theano.shared(np.random.uniform(-.01, .01, size=(n_hidden, n_out)), name='W_out')
        self.W_hidden = theano.shared(np.random.uniform(-.01, .01, size=(n_hidden, n_hidden)), name='W_hidden')
        self.b = theano.shared(np.random.uniform(-.01, .01, size=(n_hidden, )), name='b')
        self.b_y = theano.shared(np.random.uniform(-.01, .01, size=(n_out, )), name='b_y')

        # initial internal state
        self.x0 = theano.shared(np.zeros(n_hidden), name='x0')
        
        # define cost function
        def rnn_one_step(u, x, W_in, W_hidden, W_out, b, b_y):
            x_t = T.tanh(T.dot(W_hidden.T, x) + T.dot(W_in.T, u) + b)
            y_t = T.dot(W_out.T, x_t) + b_y
            return x_t, y_t

        U = T.matrix('U')
        D = T.matrix('D')
        lr = T.scalar('lr')
        [x_vals, y_vals], _ = theano.scan(fn=rnn_one_step,
                                               sequences=[U],
                                               outputs_info=[self.x0, None],
                                               non_sequences=[self.W_in, self.W_hidden, self.W_out, self.b, self.b_y])
        objective = T.mean((y_vals - D) ** 2)
        self.params = [self.W_hidden, self.W_in, self.W_out, self.b, self.b_y, self.x0]
        self.gparams = [T.grad(objective, p) for p in self.params]

        updates = {}
        for p, g in zip(self.params, self.gparams):
            updates[p] = p - lr * g

        self.train_iteration = theano.function([U, D, lr], [objective], updates=updates, mode=mode)
        self.output = theano.function([U], outputs=[y_vals], mode=mode)
        self.debug_get_grad = theano.function([U, D], self.gparams, mode=mode)


    def train(self, U, D, max_epoch=200):
        epoch = 0
        while epoch < max_epoch:
            costs = []
            for i in range(len(U)):
                # first dimension should be time
                u, d = U[i], D[i]
                cost = self.train_iteration(u, d, self.learning_rate)
                costs.append(cost[0])
            print 'epoch %d:    cost: %f, lr=%f' % (epoch, np.mean(costs), self.learning_rate)
            self.learning_rate *= self.learning_rate_decay
            epoch += 1
        return self


    def predict(self, U):
        Y = []
        for i in range(len(U)):
            u = U[i]
            y = self.output(u)
            y = np.asarray(y).reshape((-1, self.n_out))
            Y.append(y)
        return Y


def main():
    # create training samples
    U, D = [], []
    n_samples = 200
    for _ in range(n_samples):
        # generate two dimension line data
        #seq_len = np.random.randint(low=30, high=50)
        seq_len = 30
        p1 = np.random.randn(2)
        p2 = np.random.randn(2)
        u = np.dot(np.arange(1, seq_len+1).reshape((-1, 1)), (p2 - p1).reshape((1,-1)))
        d = np.vstack((u[0, :], u[:-1, :]))
        U.append(u)
        D.append(d)
    U_train, D_train, U_test, D_test = U[:n_samples/2], D[:n_samples/2], U[n_samples/2:], D[n_samples/2:]

    rnn = RNN(2, 2, 40)
    rnn.train(U[:n_samples/2], D[:n_samples/2])

    Y = rnn.predict(U_train)
    print D_train[0], Y[0]
    for i in range(5):
        l1 = plt.plot(D_train[i][:,0], D_train[i][:,1], marker='o')
        l2 = plt.plot(Y[i][:,0], Y[i][:,1], color=l1[0].get_color(), marker='x', linestyle='--')
    plt.show()

    print np.mean([np.mean((Y[i] - D_train[i])**2) for i in range(len(D_train))])


if __name__ == '__main__':
    main()
