# 学习图像处理

## grayscale_16
用16级灰阶[0,16,32,48,64,80...........]来显示256级灰阶图
    
### 算法
- 用一个最接近的16级灰度值来表示一个256级的灰度值
比如用一个48来表示一个45，用一个16来表示一个11

- 用四个16级灰度值表示一个256级的灰度值
比如用$ \begin{bmatrix} 32 & 48 \\ 48 & 48 \\ \end{bmatrix} $来表示一个45

- 按比例用16级灰度值表示256级灰度值
比如用[32，48]（32和48的比例为1:3）来表示四个45