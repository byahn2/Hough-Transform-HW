''' Consider the possible parameterizations of boundaries of interest.  In particular, consider a circle or a quadratic
curve.
1. Select either circle or quadratic curve and derive the Hough transform for that case
2. Implement your own Python routine for the boundary shape you chose and apply it to the appropriate image in shoeprint.
You may need to think behind simple Canny edge detection to provide sufficient input to the hough transform
3. Are you able to get it to work?  Discuss the benefits and drawbacks of the  Hough transform in the context of the
bounded shape you chose and the image domain you worked with.  Supplement the discussion with other images if needed.
(Max 1 page)

In the case you are implementing Hough circle transform, OpenCV-Python provides a function cv2.HoughCircles().  Please
compare your implementation to the one probided by OpenCV-Python and discuss your findings '''

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

# Canny edge detector
# filter image with derivative of Gaussian
# find magnitude and orientation of gradient
# Non-maximum suppression
# Linking and thresholding
    # define two thresholds: low and high
    # use the high threshold to start edge curves and low threshold to continue them
    # strong edges are edges, week edges are edges iff they connect to strong edges
    # look in some neighborhood (usually 8 closest)

def canny_edge_detector(img):
    # blur image
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img_bw) = cv2.threshold(gray_scale,145,255,cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(img_bw,(11,11),0)
    edges, theta = get_gradients(blur)
    max_edges = non_max_supression(edges, theta)
    thresh_edges, weak, strong = thresholding(max_edges)
    result_1 = hysteresis(thresh_edges, weak, strong)

    plt.subplot(1,4,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 4, 2), plt.imshow(gray_scale, cmap='gray')
    plt.title('Grayscale'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 4, 3), plt.imshow(img_bw, cmap='gray')
    plt.title('Binary image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,4,4),plt.imshow(blur,cmap = 'gray')
    plt.title('Gaussian blur'), plt.xticks([]), plt.yticks([])
    plt.show()
    plt.subplot(1, 2, 1), plt.imshow(edges, cmap='gray')
    plt.title('Edges'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(max_edges, cmap='gray')
    plt.title('After non-max supression'), plt.xticks([]), plt.yticks([])
    plt.show()
    plt.subplot(1, 2, 1), plt.imshow(thresh_edges, cmap='gray')
    plt.title('After thresholding'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(result_1, cmap='gray')
    plt.title('After Hysterisis'), plt.xticks([]), plt.yticks([])
    plt.show()

    result = cv2.Canny(gray_scale, 100, 200, 5, L2gradient=True)
    plt.subplot(1, 1, 1), plt.imshow(result, cmap='gray')
    plt.title('cv2_canny'), plt.xticks([]), plt.yticks([])
    plt.show()

    result[:] = result[:] - result[:] + result_1
    return result

def get_gradients(img):
    #take the gradients
    Ix = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    Iy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    edges = np.sqrt(np.square(Ix) + np.square(Iy))
    theta = np.arctan2(Iy, Ix)

    return edges, theta

def non_max_supression(img, theta):
    # Create a matrix initialized to 0 of the same size of the original gradient intensity matrix;
    # Identify the edge direction based on the angle value from the angle matrix
    # Check if the pixel in the same direction has a higher intensity than the pixel that is currently processed
    # Return the image processed with the non-max suppression algorithm

    M, N = img.shape
    max_array = np.zeros((M, N), dtype=np.int32)

    theta[img == 0] = 360
    theta[theta <= (-7. * np.pi / 8.)] = 540
    theta[theta <= (-5. * np.pi / 8.)] = 405
    theta[theta <= (-3. * np.pi / 8.)] = 450
    theta[theta <= (-1. * np.pi / 8.)] = 495
    theta[theta <= (1. * np.pi / 8.)] = 540
    theta[theta <= (3. * np.pi / 8.)] = 495
    theta[theta <= (5. * np.pi / 8.)] = 450
    theta[theta <= (7. * np.pi / 8.)] = 405
    theta[theta <= np.pi] = 540
    theta = theta - 360

    for m in range(M-1):
        for n in range(N-1):
            try:
                q = 255
                r = 255
                # if m > 0:
                #     print('m = ', str(m), ' n = ', str(n))
                #     print(theta[m, n])
                # horizontal gradients compare to space left and right
                if theta[m, n] == 180:
                    q = img[m, n + 1]
                    r = img[m, n - 1]
                    #print('theta = 180')
                # diagonal gradients compare to space down to the left and up to the right
                elif theta[m, n] == 45:
                    q = img[m + 1, n - 1]
                    r = img[m - 1, n + 1]
                    #print('theta = 45')
                # vertical gradients compare to space down one and up one
                elif theta[m,n] == 90:
                    q = img[m + 1, n]
                    r = img[m - 1, n]
                    #print('theta = 90')
                # diagonal gradients compare to space down to the right and up to the left
                elif theta[m,n] == 135:
                    q = img[m - 1, n - 1]
                    r = img[m + 1, n + 1]
                    #print('theta = 135')

                if (img[m, n] >= q) and (img[m, n] >= r):
                    #print(str(q), ',', str(r), ' < ', str(img[m,n]))
                    max_array[m, n] = img[m, n]
                else:
                    #print(str(q), ' or ', str(r), ' > ', str(img[m, n]))
                    max_array[m, n] = 0

            except IndexError as e:
                print('index error')
                pass

    return max_array

def thresholding(img):
    img = img / np.max(img) * 100.
    highThreshold = 70
    lowThreshold = 30

    M, N = img.shape
    result = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(150)
    strong = np.int32(255)

    result[img < lowThreshold] == 0
    result[np.where((img >= highThreshold) & (img > lowThreshold))] = weak
    result[img >= highThreshold] = strong

    return (result, weak, strong)


def hysteresis(img, weak, strong):
    M, N = img.shape
    for m in range(1, M-1):
        for n in range(1, N-1):
            if (img[m,n] == weak):
                try:
                    if ((img[m+1, n-1] == strong) or (img[m+1, n] == strong) or (img[m+1, n+1] == strong)
                        or (img[m, n-1] == strong) or (img[m, n+1] == strong)
                        or (img[m-1, n-1] == strong) or (img[m-1, n] == strong) or (img[m-1, n+1] == strong)):
                        img[m, n] = strong
                    else:
                        img[m, n] = 0
                except IndexError as e:
                    pass
    return img


# Hough transform
def hough_circle_detection(edges, img):
    # initialize accumulator H to all zeros
    # for each edge point (x,y)
    #   theta = the gradient orientation at (X,y)
    #   r = xcos(theta) + ysin(theta)
    #????(x,y) + r*gradient of I(x,y)
    #   H(theta, r) = H(theta, r) + 1

    M, N, D = img.shape
    max_r = int(np.floor(np.sqrt(np.square(M) + np.square((N)))))
    # H is a,b,r space. a can be at most M, b can be at most N, and r can be at most the diagonal
    H = np.zeros((N, M, max_r))
    e, theta = get_gradients(edges)
    for m in range(M):
        for n in range(N):
            if e[m, n] != 0:
                slope = np.tan(theta[m, n])
                bias = m - slope*n

                a = np.arange(N)
                b = slope*a + bias

                a_prime = np.square(-1 * a + n)
                b_prime = np.square(-1 * b + m)
                r = np.sqrt(a_prime + b_prime)

                a = np.int32(np.round(a))
                b = np.int32(np.round(b))
                r = np.int32(np.round(r))

                a_vote = a[np.where((b >= 0) & (b < M))]
                b_vote = b[np.where((b >= 0) & (b < M))]
                r_vote = r[np.where((b >= 0) & (b < M))]

                a_vote = a_vote[np.where(r_vote < max_r)]
                b_vote = b_vote[np.where(r_vote < max_r)]
                r_vote = r_vote[np.where(r_vote < max_r)]

                H[a_vote, b_vote, r_vote] += 1

    threshold = 0.82 * np.max(H)
    top_votes = np.where(H > threshold)
    a_s = top_votes[0]
    b_s = top_votes[1]
    r_s = top_votes[2]
    for c in range(len(top_votes[0])):
        a = a_s[c]
        b = b_s[c]
        r = r_s[c]
        angles = np.linspace(0., 2*np.pi, 6*r)
        for d in angles:
            i = b + np.int32(np.round(r*np.sin(d)))
            j = a + np.int32(np.round(r*np.cos(d)))
            if (i >= 0 and j >= 0 and i < M and j < N):
                img[i-1:i+1, j-1:j+1, 0] = 255

    plt.imshow(img)
    plt.show()






# turn into a binary image using a threshold and gaussian filters
# find edges using canny edge detection
#all edge points used by the circle hough transform to find underlying circle structure


def main():
    img = cv2.imread('shoeprint/circle.png')
    result = canny_edge_detector(img)
    final = hough_circle_detection(result, img)

main()