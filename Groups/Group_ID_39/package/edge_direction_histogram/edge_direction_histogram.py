import numpy as np
import cv2
import math

# defining kernals for all the 5 edges :
# horizontal, vertical, diagonal45, diagonal135, nondiagonal
kernals={}
kernals['Horizontal']=np.array([[1,1],[-1,-1]])
kernals['Vertical']=np.array([[1,-1],[1,-1]])
kernals['Diagonal45']=np.array([[1.414,0],[0,-1.414]])
kernals['Diagonals135']=np.array([[0,1.414],[-1.414,0]])
kernals['Nondiadonal']=np.array([[2,-2],[-2,2]])

# returns the Global Bins by calculating mean 
# of all the edges
def _Getmean(Bins):
  mean=np.zeros(5)
  for i in range(16):
    for j in range(5):
      mean[j]+=Bins[i][j]    
  mean[0]=mean[0]/16
  mean[1]=mean[1]/16
  mean[2]=mean[2]/16
  mean[3]=mean[3]/16
  mean[4]=mean[4]/16
  return mean

# _mull function multiple kernal['edge'] 
# with the 2*2 array of sub images of Blocks and retuens its 
# absloute sum of all element of resultant 2*2 array 
def _mull(m1,m2):
  s=0
  for i in range(2):
    for j in range(2):
      s+=(m1[i][j])*(m2[i][j])
  return abs(s)   

# getBins function will generate all the 2*2 sub images 
# of Blocks and and return Bin of that Block   
def getBins(Block):
  [r,c]=Block.shape

  # resizing the Block to multiple of 2
  r,c=2*math.ceil(r/2),2*math.ceil(c/2)
  Block=cv2.resize(Block, (r,c))

  # Bin is the return in end of fuction completion
  Bin=np.zeros(5)

  # Threshold is set for proper calculation of Bin
  Threshold=50

  nbr=int(r/2)
  nbc=int(c/2)

  # this loop will generate all the 2*2 sub images of Block 
  l=0
  for i in range(nbr):
    k=0
    for j in range(nbc):
      block=Block[k:k+2,l:l+2]
      dis={}
      
      # multiplying sub image with all the kernals
      b1=_mull(block,kernals['Horizontal'])
      b2=_mull(block,kernals['Vertical'])
      b3=_mull(block,kernals['Diagonal45'])
      b4=_mull(block,kernals['Diagonals135'])
      b5=_mull(block,kernals['Nondiadonal'])
      dis[b1]=0
      dis[b2]=1
      dis[b3]=2
      dis[b4]=3
      dis[b5]=4

      # finding the maximum amomg all the bin of sub image
      maximum=max(b1,b2,b3,b4,b5)
      index=dis[maximum]

      # checking if it is greater than threshold value
      # if true than incrementing count of that edge in Bin
      if maximum>=Threshold:
        Bin[index]+=1 
       
      k+=2
    l+=2
  
  return Bin

# edge_direction() is the Edge direction histogram
def edge_direction(image):
    
    # saving the dimension of image i.e height and width using image.shape
    (height,width)=image.shape

    # resiizing image with height and width with mutiple of 4
    M,N=4*math.ceil(height/4),4*math.ceil(width/4)
    image= cv2.resize(image, (M,N))
    #image.shape

    # initilizing Bins of 17*5 array for all the 16 Blocks of image 
    # and one for Global Bins
    # and 5 for all the diagonals
    Bins=np.zeros((17,5))

    p=0
    l=0
    # dividing image into 4*4 images called Block
    # calculating Bins for all the Blocks gives
    # 16 Bins + 1 Global_Bin i.e total 17 Bins
    # finally falttering it into 1-D array
    # gives 85 features of image
    for i in range(4):
        k=0
        for j in range(4):
            Block=image[k:k+int(M/4),l:l+int(N/4)]
    
            Bins[p]=getBins(Block)
    
            k+=int(M/4)
            p+=1
        l+=int(N/4)
    GlobalBins=_Getmean(Bins)
    Bins[16]=GlobalBins
    
    Features=Bins.flatten() 
    return Features
