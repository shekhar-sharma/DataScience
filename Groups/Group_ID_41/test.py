import CRMvFE

views=8
objects=10

X = CRMvFE.get_input(views, objects)
Z = CRMvFE.crmvfe(X, views, objects)
print(Z)
print(Z.shape)