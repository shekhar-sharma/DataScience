from RMvFS import RMvFS

r = RMvFS(threshold=0.3,max_iter=14)
X,y = load_iris(return_X_y=True)
X1 = [[60, 468], [40, 400], [46, 109], [50, 150], [12, 57]]     #height and width of advertisment images
X2 = [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1], [1, 1, 0, 0, 0], [1, 0, 0, 1, 1]]    # sparse matrix for various tags in the url of asvertisement
r.fit(X1,X2,Y=['ad', 'ad', 'notad', 'ad', 'notad'])

print(r.transform(X1,X2))

print(r.feature_importance)
