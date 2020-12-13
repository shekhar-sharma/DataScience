from TensorCCA import *

U = [np.random.randint(1,4,(9,i)) for i in (20, 10, 14)]

tcca_object = TensorCCA(max_iter = 500)

fit_data = tcca_object.fit(U,reduce_to_dim = 3)

new_views = tcca_object.transform(U)

print(new_views)