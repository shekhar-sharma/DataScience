import numpy as np
import theano.tensor as T
import theano
import config

def MvLDAN_gneral(inputs):
    ll = len(inputs) // 2
    data = inputs[0: ll]
    labels = inputs[ll::]
    cost = th_MvLDAN_cost(data, labels)
    return cost


def th_MvLDAN_cost(data, labels):
    Sw, Sb, n_components = th_MvLDAN_Sw_Sb(data, labels)


    from theano.tensor import slinalg
    evals = slinalg.eigvalsh(Sb, Sw)

    # n_components = data[0].shape[1] - back_step
    top_evals = evals[-n_components:]
    # top_evals = evals[(evals > 0.).nonzero()]
    thresh = T.min(top_evals) + config.threshold
    cost = -T.mean(top_evals[(top_evals <= thresh).nonzero()])
    return cost

def th_MvLDAN_Sw_Sb(data, labels):
    n_view = len(data)
    dtype = 'float32'

    la = T.concatenate(labels, 0).reshape([-1])
    classes = T.extra_ops.Unique(True, False, False)(la)[0].reshape([-1])
    cnum = T.reshape(classes, [-1, 1]).shape[0]

    def loop_func(__i, __sw, __sb):

        __ni = 0
        Xgt = list(range(n_view))
        Xsum = list(range(n_view))
        for v_i in range(n_view):
            Xgt[v_i] = data[v_i][T.eq(labels[v_i], classes[__i]).nonzero(), :].reshape([-1, data[v_i].shape[1]])
            Xsum[v_i] = T.sum(Xgt[v_i], axis=0).reshape([1, -1])
            __ni += Xgt[v_i].shape[0]
        __sw_ = []
        __sb_ = []
        __ni = __ni.astype(dtype)
        for v_i in range(n_view):
            __sw__ = []
            __sb__ = []
            for v_j in range(n_view):
                tmp = T.dot(Xsum[v_i].T, Xsum[v_j]) / __ni
                sw_tmp = -tmp
                sb_tmp = tmp
                if v_i == v_j:
                    sw_tmp += T.dot(Xgt[v_i].T, Xgt[v_j])
                __sw__.append(sw_tmp)
                __sb__.append(sb_tmp)
            __sw_.append(T.concatenate(__sw__, axis=1))
            __sb_.append(T.concatenate(__sb__, axis=1))
        __sw += T.concatenate(__sw_, axis=0)
        __sb += T.concatenate(__sb_, axis=0)

        return __sw, __sb

    dim = 0
    for v in range(n_view):
        dim += data[v].shape[1]
    scan_result, scan_update = theano.scan(fn=loop_func, outputs_info=[T.zeros([dim, dim], dtype=dtype),
                                                                       T.zeros([dim, dim], dtype=dtype)],
                                           sequences=[T.arange(cnum)])

    Sw = scan_result[0][-1]
    Sb = scan_result[1][-1]

    n_all = 0
    sum_v = []
    for v_i in range(n_view):
        sum_v.append(T.sum(data[v_i], axis=0).reshape([1, -1]))
        n_all += data[v_i].shape[0]
    D_i = []
    n_all = n_all.astype(dtype=dtype)
    for v_i in range(n_view):
        D_ij = []
        for v_j in range(n_view):
            D_ij.append(T.dot(sum_v[v_i].T, sum_v[v_j]))
        D_i.append(T.concatenate(D_ij, axis=1))
    Sb -= (T.concatenate(D_i, axis=0) / n_all)
    # tmp = tf.matmul(tf.matrix_inverse(Sw + tf.eye(n_view * n, n_view * n, dtype=dtype) * 1e-3), Sb)
    Sw += T.eye(dim, dim, dtype=dtype) * config.l2_eig
    Scb = []
    Scw = []
    # for i in range(n_view):
    #     x_data[i] = x_data[i] - tf.reduce_mean(x_data[i], axis=0)
    for i in range(n_view):
        tmp1 = []
        tmp2 = []
        for j in range(n_view):
            if i != j:
                tmp1.append(T.dot(data[i].T, data[j]))
                tmp2.append(T.zeros([data[i].shape[1], data[j].shape[1]], dtype=dtype))
            else:
                tmp1.append(T.zeros([data[i].shape[1], data[j].shape[1]], dtype=dtype))
                tmp2.append(T.dot(data[i].T, data[j]))
        Scb.append(T.concatenate(tmp1, axis=1))
        Scw.append(T.concatenate(tmp2, axis=1))

    Sb += T.concatenate(Scb, axis=0) * config.lambda_cca1
    # Sw += T.concatenate(Scw, axis=0) * config.lambda_cca2
    n_components = T.min([cnum, data[0].shape[1]]) - config.back_step

    return Sw, Sb, n_components


def th_MvLDAN(data_inputs, labels):
    n_view = len(data_inputs)
    dtype = 'float32'
    mean = []
    std = []
    data = []
    for v in range(n_view):
        _data = theano.shared(data_inputs[v])
        _mean = T.mean(_data, axis=0).reshape([1, -1])
        _std = T.std(_data, axis=0).reshape([1, -1])
        _std += T.eq(_std, 0).astype(dtype)
        data.append((_data - _mean) / _std)
        mean.append(_mean)
        std.append(_std)

    Sw, Sb, _ = th_MvLDAN_Sw_Sb(data, labels)

    from theano.tensor import nlinalg
    eigvals, eigvecs = nlinalg.eig(T.dot(nlinalg.matrix_inverse(Sw), Sb))
    # evals = slinalg.eigvalsh(Sb, Sw)
    mean = list(theano.function([], mean)())
    std = list(theano.function([], std)())
    eigvals, eigvecs = theano.function([], [eigvals, eigvecs])()
    inx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[inx]
    eigvecs = eigvecs[:, inx]
    W = []
    pre = 0
    for v in range(n_view):
        W.append(eigvecs[pre: pre + mean[v].shape[1], :])
        pre += mean[v].shape[1]
    return [mean, std], W, eigvals
