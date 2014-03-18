import pickle
import gzip
import numpy as np
import cv2

max_train_iter = 3000
gibbs_iter = 1000

rbm_path = 'trained-rbm.pkl'

do_training = False
do_sampling = True
do_testing = False

def sigmoid(a):
    return 1. / (1 + np.exp(-a))


def update_RBM_from_data(W, X, eta, bias):
    '''
    train by one step contrastive divergence (CD-1)
    '''
    num_sample = X.shape[0]
    if bias:
        X = np.hstack((X, np.ones((num_sample, 1))))
    # sample hidden layer from data
    p = sigmoid( np.dot(W.T, X.T) ) 
    h1 = np.random.binomial(1, p)
    if bias:
        h1[-1,:] = 1
    # sample visible layer
    p = sigmoid( np.dot(W, h1) )
    v = np.random.binomial(1, p)
    if bias:
        v[-1,:] = 1
    # probability of hidden layer from sampled visible layer 
    # final hidden state can use probability to avoid sampling noise
    ph = sigmoid( np.dot(W.T, v) ) 
    if bias:
        ph[-1,:] = 1
    # update weight matrix W
    W += eta * (np.dot(h1, X).T - np.dot(v, ph.T)) / num_sample
    return W


def train_RBM(data, num_hidden=128, eta=0.05, bias=True):
    num_data, num_visible = data.shape
    if bias:
        num_visible += 1
        num_hidden += 1
    W = np.random.rand(num_visible, num_hidden)
    for it in range(max_train_iter):
        print 'iteration: ', it
        d = data[np.random.random_integers(0, num_data-1, num_data / 10), :]
        W = update_RBM_from_data(W, data, eta, bias)
    return W


def sample_RBM(W, n_samples=1):
    num_visible, num_hidden = W.shape
    v = np.random.rand(num_visible)
    S = []
    for t in range(n_samples):
        for i in range(gibbs_iter):
            h = np.random.binomial(1, sigmoid( np.dot(W.T, v) )) # sample hidden layer from data
            v = np.random.binomial(1, sigmoid( np.dot(W, h) ))   # sample visible layer
        S.append(v[:-1])
    return np.array(S)


def main():
    if do_training:
        # load data
        train_set, valid_set, test_set = pickle.load(gzip.open('mnist.pkl.gz', 'rb'))

        # train RBM
        rbms = []
        for digit in range(0,10):
            X = train_set[0][(train_set[1] == digit), :]
            X_rsz = []
            for i in range(X.shape[0]):
                d = cv2.resize(X[i].reshape(28,28), (14,14))
                if i < 10:
                    cv2.imwrite('data-%02d.ppm' % i, np.array((d >= 0.5)*255, dtype=np.uint8))
                d2 = (d >= 0.5).flatten()
                X_rsz.append( d2 )
            X = np.array(X_rsz)
            W = train_RBM(X)
            rbms.append(W)
        pickle.dump(rbms, open(rbm_path, 'w'))

    if do_sampling:
        rbms = pickle.load(open(rbm_path, 'r'))
        # sample from RBM
        samples_img = None
        for digit in range(0,10):
            W = rbms[digit]
            S = sample_RBM(W, 10)
            row = None
            for t in range(S.shape[0]):
                v = S[t, :]
                img = v.reshape(14,14)
                img = np.array(img * 255, dtype=np.uint8)
                row = np.hstack((row, img)) if row is not None else img
            samples_img = np.vstack((samples_img, row)) if samples_img is not None else row
        cv2.imwrite('samples.png', samples_img)

    if do_testing:
        pass


if __name__ == '__main__':
    main()
