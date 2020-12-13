import numpy as np

class GrayScaleIntensity:
    def g_feature(self,image):
        if(len(image.shape)<3):
            features = np.reshape(image, (image.size))
            return features
        else:
            if(len(image.shape)==3):
                x,y,z = image.shape
                feature_matrix = np.zeros((x,y)) 
                for i in range(0,image.shape[0]):
                    for j in range(0,image.shape[1]):
                        R = int(image[i,j,0])
                        G = int(image[i,j,1])
                        B = int(image[i,j,2])
                        Y = (0.299 * R) + (0.587 * G) + (0.114 * B);
                        U = (B - Y) * 0.565
                        V = (R - Y) * 0.713
                        UV = U + V
                        R1=R*0.299
                        R2=R*0.587
                        R3=R*0.114
                        G1=G*0.299
                        G2=G*0.587
                        G3=G*0.114
                        B1=B*0.299
                        B2=B*0.587
                        B3=B*0.114
                        R4=(R1+R2+R3)/3
                        G4=(G1+G2+G3)/3
                        B4=(B1+B2+B3)/3
                        I1=(R4+G4+B4+UV)/4
                        feature_matrix[i][j] = I1
                features = np.reshape(feature_matrix, feature_matrix.size)
                return features





