import torch
import numpy as np

class MVDA():
    def __init__ (self):
 
        self.multiView_X = []
        self.numViews = 0
        self.Y = []
        
# Group together the tensors of the same class or label.

    def featureGroupingByClass (self):
        uniqueY = np.unique(self.Y)
        groupedFeaturesByViews = []
        
        for view in self.multiView_X:
            groupedFeaturesByViews.append([view[np.where( self.Y == label)[0]] for label in uniqueY])
        
        return groupedFeaturesByViews
    
# Find the dimension of tensors.    
    def dimensions (self, groupedFeaturesByViews):
        
        return [len(view[0][0]) for view in groupedFeaturesByViews]
    
# Find the mean of each class of tensors.    
    
    def meanOfEachClass (self, groupedFeaturesByViews):
        meanByClass = [[torch.mean(label, dim=0) for label in view] for view in groupedFeaturesByViews]
        return meanByClass
    

    
    def numOfSamples (self, groupedFeaturesByViews):
        return [[len(view_i) for view_i in view] for view in groupedFeaturesByViews], [len([view[label] for view in groupedFeaturesByViews]) for label in range(len( groupedFeaturesByViews[0]))]

#  Calculate  with in class scatter matrix.   
    def withinClassScatterMatrix (self, groupedFeaturesByViews, meanByClass):
    
        dims = self.dimensions( groupedFeaturesByViews )
        uniqueTensorsY = torch.unique(torch.tensor( self.Y ))
        viewWiseSamples, labelWiseSamples = self.numOfSamples( groupedFeaturesByViews )

        cols = []
        for j in range (self.numViews):
            rows = []
            for r in range (self.numViews):
                s_jr = torch.zeros((dims[j], dims[r]))
                v_jr = torch.zeros((dims[j], dims[r]))
                for i in range (len(uniqueTensorsY)):
                    s_jr -= viewWiseSamples[j][i] * viewWiseSamples[r][i] / labelWiseSamples[i] * (
                            meanByClass[j][i].unsqueeze(0).t() @ meanByClass[r][i].unsqueeze(0))
                    v_jr += groupedFeaturesByViews[j][i].t() @ groupedFeaturesByViews[j][i] if j == r else 0
                if j==r:
                    rows.append(s_jr + v_jr)
                else:
                    rows.append(s_jr)

            cols.append(torch.cat(rows, dim=1))
        Sw = torch.cat(cols, dim=0)
        return Sw

#  Calculate between class scatter matrix.   
    def betweenClassScatterMatrix (self, groupedFeaturesByViews, meanByClass):

        dims = self.dimensions (groupedFeaturesByViews)
        uniqueTensorsY = torch.unique(torch.tensor( self.Y))
        viewWiseSamples, labelWiseSamples = self.numOfSamples( groupedFeaturesByViews )
        labelWiseSum = sum(labelWiseSamples)

        cols = []
        for j in range (self.numViews):
            mean_j = torch.sum(torch.cat([viewWiseSamples[j][i] * meanByClass[j][i].unsqueeze(0) for i in range(len(uniqueTensorsY))]), dim=0)
            rows = []
            for r in range (self.numViews):
                mean_r = torch.sum(torch.cat([viewWiseSamples[r][i] * meanByClass[r][i].unsqueeze(0) for i in range(len(uniqueTensorsY))]), dim=0)

                d_jr = torch.zeros((dims[j], dims[r]))
                for i in range(len(uniqueTensorsY)):
                    d_jr += viewWiseSamples[j][i] * viewWiseSamples[r][i] / labelWiseSamples[i] * (meanByClass[j][i].unsqueeze(0).t() @ meanByClass[r][i].unsqueeze(0))
                q_jr = mean_j.unsqueeze(0).t() @ mean_r.unsqueeze(0)


                s_ij = d_jr - q_jr / labelWiseSum

                rows.append(s_ij)

            cols.append(torch.cat(rows, dim=1))
        Sb = torch.cat(cols, dim=0)
        return Sb * self.numViews
    

# Calculate the eigenvectors using within class scatter matrix{ Sw } and between class scatter matrix { Sb }    
    def eigenDecomposition (self, Sw, Sb):
        
        # dividing Sb by Sw, for maximizing Sb and minimizing Sw
        W = Sw.inverse() @ Sb
        evals, evecs = torch.eig(W, eigenvectors=True)   
        evecs = evecs[:, torch.argsort(evals[:, 0].abs(), descending=True)]
        return evecs
# Using eigen decomposition find the V linear transforms .    
    def eigenprojections( self, eigenVecs, dims):
        return [eigenVecs[sum(dims[:i]):sum(dims[:i+1]), ...] for i in range(len(dims))]
    
    
    def fit_transform (self, X, Y):
        
        self.multiView_X = []
        self.numViews = len(X)
        self.Y = Y
        
        #Convert the list of features from each view into a list of pytorch tensors.
        for view in X:
            self.multiView_X.append(torch.tensor(view).float())
        
        groupedFeaturesByViews = self.featureGroupingByClass()


        dimension = self.dimensions(groupedFeaturesByViews)
        meanByClass = self.meanOfEachClass(groupedFeaturesByViews)
        Sw = self.withinClassScatterMatrix(groupedFeaturesByViews, meanByClass)
        Sb = self.betweenClassScatterMatrix(groupedFeaturesByViews, meanByClass)

        eigenVecs = self.eigenDecomposition(Sw, Sb)
        vTransforms = self.eigenprojections(eigenVecs, dimension)
        
        return vTransforms