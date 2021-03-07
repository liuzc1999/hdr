# coding: utf-8

import os
import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import cProfile
import constant



# 载入图像
# Code based on https://github.com/SSARCandy/HDR-imaging
def load_exposures(source_dir, channel=0):
    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, exposure, *rest) = line.split()
        filenames += [filename]
        exposure_times += [exposure]
    
    img_list = [cv2.imread(os.path.join(source_dir, f), 1) for f in filenames]
    img_list = [img[:,:,channel] for img in img_list]
    exposure_times = np.array(exposure_times, dtype=np.float32)

    return (img_list, exposure_times)



# 最小二乘法求解相机响应函数
def get_response_curve(img_list, exposure_times):
    ln_t = [np.log(e) for e in exposure_times]
    l = constant.Lamda

    # 权值
    w=np.zeros((256,), dtype=np.int)
    for z in range(0,128):
        w[z]=z
    for z in range(128,256):
        w[z]=255-z

    # 取sample：100区平均+64典型像素
    small_img = [cv2.resize(img, (10, 10)) for img in img_list]
    average_Z = [img.flatten() for img in small_img]
    I1=np.size(average_Z, 1)
    J=np.size(average_Z, 0)

    tsize=J*64
    typical_Z=np.zeros((tsize,), dtype=np.int)
    t0=0
    for img in img_list:
        height=img.shape[0]
        width=img.shape[1]
        for i in range(1,9):
            typical_Z[t0]=int(img[int((height*i)/9),int((width*i)/9)])
            t0+=1

    n = 256
    A = np.zeros(shape=(J*(I1+64)+n+1, n+I1+64), dtype=np.float32)
    b = np.zeros(shape=(np.size(A, 0), 1), dtype=np.float32)
    
    # 构造Ax=b
    # 样本约束
    k = 0
    for i in range(I1):
        for j in range(J):
                z = average_Z[j][i]
                wij = w[z]
                A[k][z] = wij
                A[k][n+i] = -wij
                b[k] = wij*ln_t[j]
                k += 1

    t=0
    for j in range(J):
        for i in range(64):
            z = typical_Z[t]
            t+=1
            wij = w[z]
            A[k][z] = wij
            A[k][n+I1+i] = -wij
            b[k] = wij*ln_t[j]
            k += 1
    
    # 中值约束
    A[k][128] = 1
    k += 1
    
    # 光滑约束
    for i in range(n-1):
        A[k][i]   =    l*w[i+1]
        A[k][i+1] = -2*l*w[i+1]
        A[k][i+2] =    l*w[i+1]
        k += 1

    # 最小二乘法求解
    x = np.linalg.lstsq(A, b)[0]
    g = x[:256]
    ln_E = x[256:]

    return g, ln_E



# 非线性加权融合
def merge(g, Z, ln_t, w):
    acc_E = np.zeros((len(Z[0]),), dtype=np.float32)
    ln_E = np.zeros((len(Z[0]),), dtype=np.float32)
    E = np.zeros((len(Z[0]),), dtype=np.float32)
    
    pixels, imgs = len(Z[0]), len(Z)
    for i in range(pixels):
        acc_w = 0
        tmp_sum=0
        for j in range(imgs):
            z = Z[j][i]
            acc_E[i] += w[z]*(g[z] - ln_t[j])
            tmp_sum+=g[z] - ln_t[j]
            acc_w += w[z]
        ln_E[i] = acc_E[i]/acc_w if acc_w > 0 else tmp_sum/imgs
        E[i]=np.exp(ln_E[i])
        acc_w = 0
    
    return E


def produce(E,pixels):
    pic = np.zeros((pixels,), dtype=np.float32)
    ca=constant.A
    cb=constant.B
    cc=constant.C
    cd=constant.D
    ce=constant.E
    cl=constant.adapted_lum
    for i in range(pixels):
        Ei=E[i]
        Ei*=cl
        Ei=(Ei * (ca * Ei + cb)) / (Ei * (cc * Ei + cd) + ce);
        pic[i]=Ei
    return pic



def preproduce(E,pixels):
    pic = np.zeros((pixels,), dtype=np.float32)
    cl=constant.adapted_lum
    for i in range(pixels):
        Ei=E[i]
        if(Ei>0):
            Ei*=cl
        pic[i]=0.9*Ei/(1+Ei)
        #Ei*=cl
        #pic[i]=Ei/(1+Ei)
    return pic



# 构建hdr图像
def construct(img_list, response_curve, exposure_times):
    img_size = img_list[0][0].shape
    
    w=np.zeros((256,), dtype=np.int)
    for z in range(0,128):
        w[z]=z
    for z in range(128,256):
        w[z]=255-z
    
    ln_t = [np.log(e) for e in exposure_times]

    hdr = np.zeros((img_size[0], img_size[1], 3), 'float32')
    #hdr2 = np.zeros((img_size[0], img_size[1], 3), 'float32')
    
    for i in range(3):
        Z = [img.flatten().tolist() for img in img_list[i]]
        E = merge(response_curve[i], Z, ln_t, w)
        #pic=preproduce(E,len(Z[0]))
        hdr[..., i] = np.reshape(E, img_size)
        #hdr2[..., i] = np.reshape(pic, img_size)
    
    return hdr



# 存储.hdr格式图像
# Code based on https://gist.github.com/edouardp/3089602
def save_hdr(hdr, filename):
    image = np.zeros((hdr.shape[0], hdr.shape[1], 3), 'float32')
    image[..., 0] = hdr[..., 2]
    image[..., 1] = hdr[..., 1]
    image[..., 2] = hdr[..., 0]

    f = open(filename, 'wb')
    f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
    header = '-Y {0} +X {1}\n'.format(image.shape[0], image.shape[1]) 
    f.write(bytes(header, encoding='utf-8'))

    brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
    mantissa = np.zeros_like(brightest)
    exponent = np.zeros_like(brightest)
    np.frexp(brightest, mantissa, exponent)
    scaled_mantissa = mantissa * 256.0 / brightest
    rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
    rgbe[...,3] = np.around(exponent + 128)

    rgbe.flatten().tofile(f)
    f.close()



def norm(hdr):
    brightest = np.max(hdr)
    for k in range(3):
        for i in range(hdr.shape[0]):
            for j in range(hdr.shape[1]):
                hdr[i][j][k]/=brightest
    return hdr



# tone mapping
def tone_mapping(hdr,fname):
    g=constant.gamma
    i=constant.intensity
    l=constant.light_adapt
    c=constant.color_adapt
    tonemapReinhard = cv2.createTonemapReinhard(g, i,l,c)
    ldrReinhard = tonemapReinhard.process(hdr)
    cv2.imwrite(fname, ldrReinhard * 255)


# main
if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('[Usage] python script <input img dir> <curve name> <rmap name> <output .hdr name> <output2 .jpg name>')
        sys.exit(0)
 
    img_dir, cname, mname, output_hdr_filename, fname = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

    # 载入
    print('Reading input images .... ')
    img_list_b, exposure_times = load_exposures(img_dir, 0)
    img_list_g, exposure_times = load_exposures(img_dir, 1)
    img_list_r, exposure_times = load_exposures(img_dir, 2)

    # 求解响应函数
    print('Solving response curves .... ')
    gb, _ = get_response_curve(img_list_b, exposure_times)
    gg, _ = get_response_curve(img_list_g, exposure_times)
    gr, _ = get_response_curve(img_list_r, exposure_times)
    print('Saving response curves plot .... ')
    plt.figure(figsize=(10, 10))
    plt.plot(gr, range(256), 'rx')
    plt.plot(gg, range(256), 'gx')
    plt.plot(gb, range(256), 'bx')
    plt.ylabel('pixel value Z')
    plt.xlabel('log exposure X')
    plt.savefig(cname)

    # 融合
    print('Constructing HDR image .... ')
    hdr = construct([img_list_b, img_list_g, img_list_r], [gb, gg, gr], exposure_times)
    print('Saving pseudo-color radiance map .... ')
    plt.figure(figsize=(12,8))
    plt.imshow(np.log2(cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)), cmap='jet')
    plt.colorbar()
    plt.savefig(mname)
    print('Saving HDR image .... ')
    save_hdr(hdr, output_hdr_filename)
    #save_hdr(hdr2, output_hdr_filename2)
    tone_mapping(hdr,fname)

    hdr=norm(hdr)
    plt.figure(figsize=(12,8))
    plt.imshow(np.log2(cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)), cmap='jet')
    plt.colorbar()
    plt.savefig('uni_'+mname)
    save_hdr(hdr, 'uni_'+output_hdr_filename)

    print('done')




