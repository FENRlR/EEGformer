# EEGformer: A transformer–based brain activity classification method using EEG signal
### Zhijiang Wan, Manyu Li, Shichang Liu, Jiajin Huang, Hai Tan, Wenfeng Duan
Unofficial implementation of [EEGformer](https://doi.org/10.3389/fnins.2023.1148855).

![Alt text](resources/fnins-17-1148855-g002.jpg)

## Test outputs
Simple test for binary classification was held using dataset from [harunshimanto/epileptic-seizure-recognition](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition).
```
>>> b 1 -> loss : 0.02516809105873108
>>> b 2 -> loss : 0.02175748720765114
>>> b 3 -> loss : 0.021974051371216774
>>> b 4 -> loss : 0.025791224092245102
>>> b 5 -> loss : 0.04797504469752312
>>> b 6 -> loss : 0.023308182135224342
>>> b 7 -> loss : 0.013465813361108303
>>> b 8 -> loss : 0.01862935908138752
>>> b 9 -> loss : 0.04344731196761131
>>> b 10 -> loss : 0.025865040719509125
>>> b 11 -> loss : 0.043744731694459915
>>> b 12 -> loss : 0.015098555013537407
>>> b 13 -> loss : 0.019317815080285072
>>> b 14 -> loss : 0.02460692636668682
acc = 0.979
sen = 0.945
spe = 0.9875
>>> epoch 30 -> tp : 189, fp : 10, tn : 790, fn : 11
```
![Alt text](resources/Figure_1.png)