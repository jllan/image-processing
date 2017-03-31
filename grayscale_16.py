import numpy as np
import matplotlib.pylab as plt


#  转换为灰度图像 Y = 0.299 R + 0.587 G + 0.114 B
def rgb2gray(img_color):
    img_gray = np.dot(img_color, [0.299, 0.587, 0.114])
    img_gray = img_gray.astype(np.uint8)
    return img_gray

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
        percent_min = (max_num-num)/(max_num-min_num)
        return (min_num, max_num, percent_min)

def replace(arr):
    arr_unique = np.unique(arr)
    for n in arr_unique:
        replace_num = search_two(n)
        if replace_num:
            min_num, max_num, percent_min = replace_num
            n_count = arr[arr==n].shape[0]
            replace_list = []
            # replace_list = [min for i in range(n_count) if np.random.random() <= replace_num[min] else max]
            for i in range(n_count):
                if np.random.random() <= percent_min:
                    replace_list.append(min_num)
                else:
                    replace_list.append(max_num)
            arr[arr == n] = replace_list
    return arr

# **************************************************************** #


if __name__ == '__main__':
    # arr = np.random.randint(0, 256, size=(60, 80))  # 随机生成60x80，值在0-255之间的矩阵
    img_rgb = plt.imread('source.jpg')

    img_gray = rgb2gray(img_rgb)
    print(img_gray[:5, :5])

    f_ufunc_one = np.frompyfunc(search_one, 1, 1)
    img_reduced = f_ufunc_one(img_gray)
    img_reduced = img_reduced.astype(np.uint8)
    print(img_reduced[:5, :5])

    f_ufunc = np.frompyfunc(search_four, 1, 1)
    extend = f_ufunc(img_gray)
    img_gray_extend = joint(extend)
    print(img_gray_extend[:5, :5])

    img_replace = replace(img_gray.copy())
    print(img_replace[:5, :5])

    # fig = plt.figure(figsize=(10,10))  # figsize=(10,6)
    # ax1 = fig.add_subplot(311)
    # ax2 = fig.add_subplot(323)
    # ax3 = fig.add_subplot(324)
    # ax4 = fig.add_subplot(325)
    # ax5 = fig.add_subplot(326)
    #
    # ax1.imshow(img_rgb)
    # ax1.set_title('原图')
    #
    # ax2.imshow(img_gray, cmap='gray')
    # ax2.set_title('灰度图')
    #
    # ax3.imshow(img_reduced, cmap='gray')
    # ax3.set_title('16级灰阶图')
    #
    # ax4.imshow(img_gray_extend, cmap='gray')
    # ax4.set_title('16级灰阶扩大图')
    #
    # ax5.imshow(img_replace, cmap='gray')
    # ax5.set_title('16级灰阶概率替换图')
    #
    # fig.subplots_adjust(hspace=0.8)

    plt.subplot(311); plt.imshow(img_rgb); plt.title('原图')
    plt.subplot(323); plt.imshow(img_gray, cmap='gray'); plt.title('灰度图')
    plt.subplot(324); plt.imshow(img_reduced, cmap='gray'); plt.title('16级灰阶图')
    plt.subplot(325); plt.imshow(img_gray_extend, cmap='gray'); plt.title('16级灰阶扩大图')
    plt.subplot(326); plt.imshow(img_gray_extend, cmap='gray'); plt.title('16级灰阶概率替换图')
    plt.show()
