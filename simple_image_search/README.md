# simple image sarch
# 简单图片搜索服务器


----------
> 该项目是借助于日本的一位博士的项目，在此基础上修改了一些代码。原本的代码是机器上运行时错误的，到现在我也没有找到原因，但是在keras作者的帮助下，使用了tensorflow的一些特性。

## 环境安装

    $ pip install numpy Pillow h5py tensorflow Keras Flask

## 如果你要在你的本机上运行的话：
> 将该项目拷贝下来或者git clone下来。

    python offline.py #提取特征并保存
    python server.py  #运行服务端

> tips:当你首次运行的时候，keras会从github上下载vgg16模型，为了节约时间建议您从百度网盘上下载vgg16模型，分别是这两种模型：
    
    vgg16_weights_tf_dim_ordering_tf_kernels.h5
    vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
 
> 下载完尽情的玩吧。^-^
    