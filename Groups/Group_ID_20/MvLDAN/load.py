import scipy.io as sio
def load_noisyMNIST():
    mnist = sio.loadmat('noisy_MNIST.mat')
    train_mnist_view1 = mnist['X1'].astype('float32')
    train_mnist_view1_labels = mnist['trainLabel']
    train_mnist_view2 =mnist['X2'].astype('float32')
    train_mnist_view2_labels = mnist['trainLabel']


    valid_mnist_view1 = mnist['XV1'].astype('float32')
    valid_mnist_view1_labels = mnist['tuneLabel']
    valid_mnist_view2 = mnist['XV2'].astype('float32')
    valid_mnist_view2_labels = mnist['tuneLabel']


    test_mnist_view1 = mnist['XTe1'].astype('float32')
    test_mnist_view1_labels = mnist['testLabel']
    test_mnist_view2 = mnist['XTe2'].astype('float32')
    test_mnist_view2_labels = mnist['testLabel']

    ret_input_size = [train_mnist_view1.shape[1::], train_mnist_view2.shape[1::]]

    data = [[[train_mnist_view1, train_mnist_view1_labels], [valid_mnist_view1, valid_mnist_view1_labels], [test_mnist_view1, test_mnist_view1_labels]], [[train_mnist_view2, train_mnist_view2_labels], [valid_mnist_view2, valid_mnist_view2_labels], [test_mnist_view2, test_mnist_view2_labels]]]
    return data, ret_input_size
