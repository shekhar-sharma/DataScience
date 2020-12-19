from libs import *


def Smoothening_Transformation(training_image):
    """
    1) A smoothening operation
    """
    #Gaussian smoothing the image using Gaussian Low pass filter
    smoothing = cv2.GaussianBlur(training_image,(5,5),0)
    """
    2) A color domain transformation
    """
    # moving image in Lab mode
    from skimage import color, io
    lab=color.rgb2lab(smoothing, illuminant='D65', observer='2')
    return lab


def Normalise_ResidualCalc(lab):   
    """
    3) A normalization operation
    """
    #splitting La*b* color space
    L,A,B=cv2.split(lab)
    #cv2.imshow("L_Channel",L) 

    #Local normalization of luminance channel
    float_lumn = L.astype(np.float32) / 255.0

    blur = cv2.GaussianBlur(float_lumn, (0, 0), sigmaX=2, sigmaY=2)
    num = float_lumn - blur

    blur = cv2.GaussianBlur(num*num, (0, 0), sigmaX=20, sigmaY=20)
    den = cv2.pow(blur, 0.5)

    luminance = num / den

    cv2.normalize(luminance, dst=luminance, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

    """

    4) A spatial residual calculation

    """

    #find fourier spectrum

    dft = cv2.dft(np.float32(luminance),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    #converted to polar coordinates
    mag=cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
    ang=cv2.phase(dft_shift[:,:,0],dft_shift[:,:,1])
    #mag,ang=cv2.cartToPolar(dft_shift[:,:,0],dft_shift[:,:,1]) 

    # logarithmic image
    # Apply log transformation method 
    c = 255 / np.log(1 + np.max(mag)) 
    log_image = c * (np.log(mag + 1)) 

    # Specify the data type so that 
    # float value will be converted to int 
    log_image = np.array(log_image, dtype = np.float32)

    smooth_log_image=cv2.blur(log_image,(5,5),0)

    #Residual 
    residual=log_image-smooth_log_image

    x, y = cv2.polarToCart(residual, ang)

    rows, cols = x.shape
    crow,ccol = int(rows/2) , int(cols/2)

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.float32)
    dft = np.zeros((rows,cols,2),np.float32)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1
    dft[:,:,0]=x
    dft[:,:,1]=y

    # apply mask and inverse DFT
    fshift = dft*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    # Applying Gaussian LPF and normalize
    smooth_img_back=  cv2.GaussianBlur(img_back,(5,5),0)
    cv2.normalize(smooth_img_back,  smooth_img_back, 0.0, 1.0, cv2.NORM_MINMAX)

    # Multiplicative pixel wise pooling
    pool_image=(luminance*smooth_img_back*255.0)
    pool_image = np.array(pool_image, dtype = np.uint8)
    return pool_image