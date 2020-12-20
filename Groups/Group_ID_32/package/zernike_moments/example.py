import cv2
import matplotlib.pylab as plt
from zm import Zernikemoment
if __name__ == '__main__':
    n = 4
    m = 2
    print('Calculating Zernike moments ...')
    fig, axes = plt.subplots(2, 3)

    imgs = ['Oval_H.png', 'Oval_45.png', 'Oval_V.png']
    for i in range(3):
        src = cv2.imread(imgs[i], cv2.IMREAD_COLOR)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        Z, A, Phi = Zernikemoment(src, n, m)
        axes[0, i].imshow(plt.imread(imgs[i]))
        axes[0, i].axis('off')
        title = 'A = ' + str(round(A, 4)) + '\nPhi = ' + str(round(Phi, 4))
        axes[0, i].set_title(title)
    print('Calculation is complete')

    plt.show()
