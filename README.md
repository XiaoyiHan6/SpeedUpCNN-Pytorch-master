# SpeedUpCNN-Pytorch-master
This Code describes how to seed up the training of CNN (Convolutional Neural Network), and also shows whether the results change. there are some my Chinese communication websites such as CSDN and Quora (Chinese)-Zhihu where I explain this code.

[CSDN](https://blog.csdn.net/XiaoyYidiaodiao/article/details/124854343?spm=1001.2014.3001.5501)

[Quora (Chinese)-Zhihu](https://zhuanlan.zhihu.com/p/516996892)

GPU is RTX 2060 (6G), So the experiment didn't work well. (No Money)

This is just a Demo of how to speed up training on a single and multi-GPU (simulation).

**AlexNet** was taken as the network architecture(**image size** required for input was **227x227x3**) as the datset, **Adamw** as the gradient descent function, and learning rate mechanism as **ReduceLROnPlateau** for example.

***

1.The results of the model without DP or DDP

Run Training Code:   `python Normal/train_XXX.py or train.py`

Run Evaluation Code: `python Normal/eval.py`

|   |Original Model|autocast|autocast+GradScaler|
|:-:|:-:|:-:|:-:|
|AP|81.91|84.03|82.20|
|traing time|22m22s|21m21s|27m27s|

***

2.The results of the model with DP 

Run Training Code:   `python DP/train_XXX.py or train.py` 

Run Evaluation Code: `python DP/eval.py`

|   |Original Model|autocast|autocast+GradScaler|
|:-:|:-:|:-:|:-:|
|AP|82.16|81.89|84.09|
|traing time|22m22s|21m21s|27m27s|

***

3.The results of the model with DDP

Run Training Code:   `python DDP/train_XXX.py or train.py`

Run Evaluation Code: `python DDP/eval.py`

|   |Original Model|autocast|autocast+GradScaler|
|:-:|:-:|:-:|:-:|
|AP|82.25|81.63|82.52|
|traing time|21m21s|20m20s|20m20s|

***
