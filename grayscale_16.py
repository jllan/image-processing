import numpy as np
import matplotlib.pylab as plt
import cv2


#  转换为灰度图像 Y = 0.299 R + 0.587 G + 0.114 B
def rgb2gray(img_color):
    img_gray = np.dot(img_color, [0.299, 0.587, 0.114])
    img_gray = img_gray.astype(np.uint8)
    return img_gray

# **************************************************************** #

#  降维处理，一个n×m的矩阵变成(n/2)x(m/2)
def reducing(array):
    row, column = array.shape
    result = np.ones((row//2, column//2))
    row, column = result.shape
    for r in range(0, row):
        for c in range(0, column):
            result[r, c] = round(np.sum(array[2 * r:2 * (r + 1), 2 * c:2 * (c + 1)])/4)
    return result

# **************************************************************** #

# 转换为16阶灰度图像, 最接近替换
def search_one(num):
    li = np.arange(0, 256, 16)
    return li[np.abs(li-num).argmin()]

# **************************************************************** #

# 转换为16阶灰度图像, 用四个16级灰阶像素点表示原图的一个像素点
def search_four(num):
    li = np.arange(0, 256, 16)
    l = []
    num = num*4
    for i in range(4, 0, -1):
        n = li[np.abs(li-num/i).argmin()]
        l.append(n)
        num = num - n
    l = np.array(l).reshape(2,2)
    return l

def joint(array):
    row, column = array.shape
    result = np.ones((row * 2, column * 2))
    for r in range(0, row):
        for c in range(0, column):
            result[2*r:2*(r+1), 2*c:2*(c+1)] = array[r, c]
    return result

# **************************************************************** #

# 转换为16阶灰度图像, 概率替换
def search_two(num):
    li = np.arange(0, 256, 16)
    if num in li:
        return None # 如果n在li中，则不需要替换
    else:
        min_num, max_num = li[np.abs(li-num).argsort()[:2]] # 找出与num最接近的两个数
        # min*x+max*(1-x)=num <==> max-(max-min)*x=num <==> (max-min)*x=max-num <==> x=(max-num)/(max-min)
        percent_min = (max_num-num)/(max_num-min_num) # 计算min_num与max_num的比例
        return (min_num, max_num, percent_min)

def replace(arr):
    arr_unique = np.unique(arr)
    for n in arr_unique:
        replace_num = search_two(n)
        if replace_num:
            min_num, max_num, percent_min = replace_num
            n_count = arr[arr==n].shape[0]  # 找出矩阵中所有的n
            # 对每一个n，生成一个随机数m，若m<=percent_min，择取min_num，否则取max_num
            replace_list = [min_num if np.random.random() <= percent_min else max_num for _ in range(n_count)]
            arr[arr == n] = replace_list
    return arr

# **************************************************************** #



if __name__ == '__main__':
    # img_rgb = plt.imread('img/11.jpg')
    img_rgb = cv2.imread('img/66.jpg')
    print('rgb:\n', img_rgb[:5, :5])

    img_gray = rgb2gray(img_rgb)
    print('gray:\n', img_gray[:5, :5])

    # 256阶灰度图变成16阶灰度图，维度不变，使用最接近发替换
    f_ufunc_one = np.frompyfunc(search_one, 1, 1)
    img_gray_deflation = f_ufunc_one(img_gray)
    img_gray_deflation = img_gray_deflation.astype(np.uint8)
    print('gray deflation:\n', img_gray_deflation[:5, :5])

    # 256阶灰度图变成16阶灰度图，使用4个16级灰度值替换一个256级灰度值，维度扩大一倍
    f_ufunc = np.frompyfunc(search_four, 1, 1)
    extend = f_ufunc(img_gray)
    img_gray_extension = joint(extend)
    print('gray extension:\n', img_gray_extension[:5, :5])

    # 先把256级灰度图降维，维度缩小一倍，然后256阶灰度图变成16阶灰度图，使用4个16级灰度值替换一个256级灰度值，维度扩大一倍，最终维度不变
    img_reduction = reducing(img_gray)
    f_ufunc = np.frompyfunc(search_four, 1, 1)
    extend = f_ufunc(img_reduction)
    img_gray_reduction_extension = joint(extend)
    print('gray eduction_extension:\n', img_gray_reduction_extension[:5, :5])

    # 256阶灰度图变成16阶灰度图，每个256级灰度值按照比例替换成16级灰度值，维度不变
    img_gray_ratio_replace = replace(img_gray.copy())
    print('gray ratio:\n', img_gray_ratio_replace[:5, :5])


    # 256阶rgb图，每个256级灰度值按照最接近替换成16级灰度值，维度不变
    f_ufunc_one = np.frompyfunc(search_one, 1, 1)
    img_rgb_deflation = f_ufunc_one(img_rgb)
    img_rgb_deflation = img_rgb_deflation.astype(np.uint8)
    print('rgb deflation:\n', img_rgb_deflation[:5, :5])

    # 256阶rgb图，每个256级灰度值按照概率替换成16级灰度值，维度不变
    img_rgb_ratio_replace = replace(img_rgb.copy())
    print('rgb ratio:\n', img_rgb_ratio_replace[:5, :5])

    # 输出各类灰度图
    fig1 = plt.figure()  # figsize=(10,6)
    rgb = fig1.add_subplot(221)
    gray = fig1.add_subplot(222)
    deflation = fig1.add_subplot(223)
    extension = fig1.add_subplot(224)
    reduction_extension = fig1.add_subplot(325)
    ratio = fig1.add_subplot(224)

    rgb.imshow(img_rgb)
    rgb.set_title('原图')

    gray.imshow(img_gray, cmap='gray')
    gray.set_title('灰度图')

    deflation.imshow(img_gray_deflation, cmap='gray')
    deflation.set_title('最接近替换灰度图图')

    extension.imshow(img_gray_extension, cmap='gray')
    extension.set_title('16级灰阶扩大图')

    reduction_extension.imshow(img_gray_reduction_extension, cmap='gray')
    reduction_extension.set_title('降维再16级灰阶扩大图')

    ratio.imshow(img_gray_ratio_replace, cmap='gray')
    ratio.set_title('按比例替换灰度图图')

    fig1.subplots_adjust(hspace=0.2)
    plt.show()

    # 输出各类rgb图
    plt.subplot(121); plt.imshow(img_rgb); plt.title('原图')
    # plt.show()
    plt.subplot(122); plt.imshow(img_rgb_ratio_replace); plt.title('按比例替换rgb图')
    # plt.subplot(224); plt.imshow(img_rgb_deflation); plt.title('最接近替换rgb图')

    cv2.imwrite("img/666.bmp", img_rgb_ratio_replace)
    # plt.savefig('img/3333.jpg')

    plt.show()
