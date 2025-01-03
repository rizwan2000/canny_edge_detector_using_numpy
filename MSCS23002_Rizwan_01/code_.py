# MSCS23002
# Rizwan Ahmad
# Assignment 1


from math import log,sqrt
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
import os

scale_factor = 255

def dx(X):
    x_shape = X.shape
    dx = np.zeros(x_shape)
    for i in range(x_shape[0]):
        for j in range(1, x_shape[1]):
            # if j == 0:
                # dx[i][j] = X[i][j] - 0
                # continue
            dx[i][j] = X[i][j] - X[i][j-1]
    return dx

def convolve(a, b):
    a_shape = a.shape
    b_shape = b.shape
    c = np.zeros(a_shape)
    x_offset = b_shape[0]//2
    y_offset = b_shape[1]//2
    for i in range(x_offset, a_shape[0]-x_offset):
        for j in range(y_offset, a_shape[1]-y_offset):
            c [i,j] = np.sum(a[i-x_offset : i+x_offset+1,j-y_offset:j+y_offset+1] * b)
    return c

def rgb_to_grayscale(img):
    grayscale = np.average(img,axis=2)
    return grayscale 

def calculate_filter_size(sigma, T):
    shalf = round(sqrt(-log(T) * 2 * sigma**2))
    N = 2*shalf + 1
    return N, shalf

def calculate_gradient(shalf, sigma):
    [Y, X] = np.meshgrid(np.arange(-shalf, shalf+1), np.arange(-shalf, shalf+1))
    print(Y)
    # print(Y**2)
    print(X)
    G_xy = np.exp(-(X**2 + Y**2)/(2 * sigma**2))
    # print(G_xy)
    fx = -X*np.exp(-(X**2+Y**2)/(2*sigma**2))
    # print(fx)
    fy = -Y*np.exp(-(X**2+Y**2)/(2*sigma**2))
    # print(fy)
    return fx, fy

def generate_masks(sigma, T):
    N, shalf = calculate_filter_size(sigma, T)
    Gx, Gy = calculate_gradient(shalf, sigma)

    Gx = np.round(Gx * scale_factor)
    Gy = np.round(Gy * scale_factor)
    # print(Gx)
    # print(Gy)
    return Gx, Gy


def apply_mask_to_image(image, mask):
    masked = convolve(image, mask)/scale_factor
    # plt.imshow(masked, cmap='gray')
    # plt.show()
    return masked

def compute_magnitude(x,y):
    M = np.sqrt(x**2 + y**2)
    # plt.imshow(M, cmap='gray')
    # plt.show()
    return M

def compute_gradient_magnitude(y, x):
    phi = np.rad2deg(np.arctan2(y,x)) + 180
    # plt.imshow(phi, cmap='gray')
    # plt.show()
    return phi

def non_maximum_suppression(M, phi):
    ## quantize direction
    angle_value = (((phi %360) + 22.5) //45) % 4
    # plt.imshow(angle_value, cmap=colors.ListedColormap(['white','cyan','red','blue','yellow']))
    # plt.show()

    value_to_dxdy = {0: (0,1),1: (1,1),2: (1,0),3: (1,-1)}
    m_shape = M.shape
    nms = np.zeros(m_shape)
    for i in range(1, m_shape[0]-1):
        for j in range(1, m_shape[1]-1):
            dx, dy = value_to_dxdy[angle_value[i,j]]
            if M[i-dx,j-dy] < M[i,j] > M[i+dx,j+dy]:
                nms[i,j] = M[i,j]
    # plt.imshow(nms, cmap='gray')
    # plt.show()

    return angle_value, nms

def followEdge(th, tl, nms, HT, visited, x, y, count=0):
    values = {0: (0,1),1: (1,1),2: (1,0),3: (1,-1)}
    if count >= 10000:
        return
    
    if (x,y) in visited:
        return
    else:
        visited[(x,y)] = True

    if nms[x,y] >= tl:
        HT[x,y] = nms[x,y]
        # check neighbors recursively
        for dx,dy in values.values():
            for factor in (1,-1):
                x2, y2 = x+dx*factor, y+dy*factor
                if nms[x2,y2] >= tl:
                    followEdge(th,tl,nms,HT,visited,x2,y2,count+1)

def hysterisis_thresholding(nms, t_h=5, t_l=10):
    nms_shape = nms.shape
    temp = np.zeros(nms_shape)
    HT = np.zeros(nms_shape)
    temp[1:-1,1:-1] = nms[1:-1,1:-1]
    visited = {}
    for i in range(nms_shape[0]):
        for j in range(nms_shape[1]):
            if temp[i,j] >= t_h:
                # check neighbors recursively
                if not (i,j) in visited:
                    followEdge(t_h, t_l, temp, HT, visited, i, j)
    # plt.imshow(HT, cmap='gray')
    # plt.show()

    return HT

def canny_edge_detection(sigma=0.5, T=0.3, t_h=5, t_l=10, image_filename='./circle.jpg'):
    filename_prefix = "./Results/" + '.'.join(image_filename.split('.')[:-1])
    Gx, Gy = generate_masks(sigma,T)
    img = cv2.imread(image_filename)
    img_gs = rgb_to_grayscale(img)

    # Applying masks to images
    Mx = apply_mask_to_image(img_gs, Gx)
    plt.imsave(f"{filename_prefix}_fx_{sigma}_{t_h}_{t_l}.jpg", Mx)
    My = apply_mask_to_image(img_gs, Gy)
    plt.imsave(f"{filename_prefix}_fy_{sigma}_{t_h}_{t_l}.jpg", My)
    
    # Compute Gradient Magnitude
    M = compute_magnitude(Mx,My)
    plt.imsave(f"{filename_prefix}_magnitude_{sigma}_{t_h}_{t_l}.jpg", M)

    # Compute Gradient Direction
    phi = compute_gradient_magnitude(My,Mx)

    # Non-Maxima Suppression
    angle_value, nms = non_maximum_suppression(M,phi)
    plt.imsave(f"{filename_prefix}_quantized_{sigma}_{t_h}_{t_l}.jpg", angle_value, cmap=colors.ListedColormap(['white','cyan','red','blue','yellow']))


    ht = hysterisis_thresholding(nms, t_h, t_l)
    plt.imsave(f"{filename_prefix}_result_{sigma}_{t_h}_{t_l}.jpg", ht)


if __name__ == '__main__':
    for item in os.listdir():
        if item.endswith(('.jpeg')):
            for sigma in (1,0.5,2):
                if sigma == 1:
                    for th,tl in ((5,10), (10,20)):
                        canny_edge_detection(image_filename=item, sigma=sigma, t_h=th, t_l=tl)
                
                canny_edge_detection(image_filename=item, sigma=sigma)