# RMvFS
> _class_ __RMvFS__( _tolerance_=0.001, _max_iter_=20,  _p_=10, _lam1_=0.01, _lam2_=0.1,  _threshold_=0.01)

&emsp; RMvFS Robust Multiview Feature Selection via View Weighted

|||
| :--- | :---|
| __Parameters:__ | __tolerance : *float, (default: 0.001)*__  <br/>&emsp; controls the conversion point of algorithm.Tolerable error amount. <br/> __max_iter: *int, (default: 20)*__ <br/>&emsp; no of iterations algorithm can take. <br/> __p: *int, (default: 10)*__ <br/> &emsp; exponent power of θv __lam1: *float, (default: 0.01)*__ <br/>&emsp; controls the robustness by tuning on each view <br/> __lam2: *float, (default: 0.1)*__ <br/>&emsp; controls the robustness by tuning on each features from each view <br/> __threshold: *float, (default: 0.01)*__ <br/>&emsp;threshold for features selection.|
| __Attributes:__ | __v : *int*__ <br/>&emsp; total number of views passed <br/> __X : *array, [v, lv, N]*__ <br> &emsp; 3d matrix contains each views’ samples <br/> __Y: *array, [ N, c ]*__ <br/>&emsp; Sparse 2d matrix, has c columns and N rows, where c is no of classes and N is sample size. Contains 1 where there is label for particular sample else 0. <br/> __N : *int*__ <br/> &emsp; Total number of samples <br/> __lv: *array, [v]*__ <br/>&emsp; array of no of features of each view <br/> __theta: *array, [v]*__ <br/>&emsp; array of θv for each view, which controls the weight of each view <br/> __c: *int*__ <br/> &emsp; number of categories or classes which are unique in labels. <br/> __W: *array, [v, lv[i], c]*__ <br/>&emsp; 3d matrix which keeps projection matrix of each view.<br/> __E: *array, [v, N, c]*__ <br/>&emsp; 3d matrix, Lagrangian Slack variable for each view.<br/> __LAMBADA : *[v, N, c]*__ <br/>&emsp; 3d matrix, Lagrangian Multiplier for each view <br/> __mu: *float*__ <br/>&emsp; a penalty parameter <br/> __zeta: *float*__ <br/> &emsp; ratio of increasing ‘mu’ with each iteration.<br/>__feature_importance : *array, [lv[i] for each i in v]*__ <br/> &emsp; array of numerical values which tells the importance of any feature represented by the index of array.|

## Note
For each view v, it finds the contribution of each view for the label given, that's here called __view weightage__. <br/><br/>
In also finds the projection matrix for each view which in turn give us __feature importance__.<br/>

## References
Jing Zhong, Ping Zhong, Yimin Xu, Lirang Yang. Robust multiview feature selection via view weighted. Multimedia Tools and Applications (2020).
## Examples
```python
>>> from RMvFS import RMvFS
>>> r = RMvFS(threshold=0.3,max_iter=14)
>>> X1 = [[60, 468], [40, 400], [46, 109], [50, 150], [12, 57]]
>>> X2 = [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1], [1, 1, 0, 0, 0], [1, 0, 0, 1, 1]]
>>> r.fit(X1,X2,Y=['ad', 'ad', 'notad', 'ad', 'notad'])
>>> print(r.feature_importance)
[330.7779217420616, 121.5821620013866, 0.20929441699046655, 0.8717475184058598, 0.04130319297080692, 0.49668218201210806, 0.4966821820121092]
>>> print(r.transform(X1,X2))
[[ 60 468   1   0   0]
 [ 40 400   1   0   0]
 [ 46 109   0   1   1]
 [ 50 150   1   0   0]
 [ 12  57   0   1   1]]
```
## Methods
|||
|:---|:---|
|[__fit( X [,*Xn], Y)__](#fit "RMvFS.fit") | fit data to model |
|[__transform( X [,*Xn] )__](#transform "RMvFS.transform")| select the features based on the parameters derived from fit method and the threshold value given during the model object initialization.| 

<br/>

> __\_\_init\_\___(_tolerance_=0.001, _max_iter_=20,  _p_=10, _lam1_=0.01, _lam2_=0.1,  _threshold_=0.01)

&emsp; Initialize self.

> <a id="fit"></a>__fit__( _X_ [_, \*Xn=None_], _Y_ )

&emsp; Fit Model to data and derives paramters/attributes like theta and W.

|||
|:---|:--|
|__Parameters:__| __X: *array, [N,lv[i]]*__ <br/> &emsp; 2d matrix, samples of all features of first view.<br/> __\*Xn: *tuple, [v-1, N, lv[i]]*__ <br/> &emsp; Array of 2d matrics which are representation of different views. Can be empty.<br/> __Y:  *array, [N,c]*__ <br/>&emsp;2d matrix, array of labels for the samples.|

> <a id="transform"></a>__transform__( _X_ [_, \*Xn=None_] )

&emsp; select the features based on the parameters derived from fit method and the threshold value given during the model object initialization.

|||
|:---|:--|
|__Parameters:__ |__X: *array, [N,lv[i]]*__ <br/> &emsp; 2d matrix, samples of all features of first view.<br/> __\*Xn: *tuple, [v-1, N, lv[i]]*__ <br/> &emsp; Array of 2d matrics which are representation of different views. Can be empty.|
|__Returns:__ | __X\_selected: *array, [v,N,lv[i]]*__ <br/>  &emsp; 3d array of selected features out of given feature matrices i.e. X and Xn.


