# <center>LSM</center>

![Alt text](./pic/LSM.png)
### Welcome to the official repository of: [New Perspectives on Multivariate Time Series Forecasting: Lightweight Networks Combined with Multi-Scale Hybrid State Space Models]. 

## Usage

1. Install requirements. ```pip install -r requirements.txt```

2. Navigate through our example scripts located at ```./scripts/TimeMachine```. You'll find the core of TimeMachine in ```models/LSM.py```. For example, to get the multivariate forecasting results for weather dataset, just run the following command, and you can open ```./result.txt``` to see the results once the training is completed. Moreover, the results will also be available at ```csv_results```, which can be utilized to make queries in the dataframe:
```
sh ./scripts/LSM/ETTh1.sh
```

Hyper-paramters can be tuned based upon needs (e.g. different look-back windows and prediction lengths). TimeMachine is built on the popular [PatchTST](https://github.com/yuqinie98/PatchTST) framework.


## Acknowledgement

We are deeply grateful for the valuable code and efforts contributed by the following GitHub repositories. Their contributions have been immensely beneficial to our work.
- Mamba (https://github.com/state-spaces/mamba)
- PatchTST (https://github.com/yuqinie98/PatchTST)
- iTransformer (https://github.com/thuml/iTransformer)
- RevIN (https://github.com/ts-kim/RevIN)
- Reformer (https://github.com/lucidrains/reformer-pytorch)
- Informer (https://github.com/zhouhaoyi/Informer2020)
- FlashAttention (https://github.com/shreyansh26/FlashAttention-PyTorch)
- Autoformer (https://github.com/thuml/Autoformer)
- Stationary (https://github.com/thuml/Nonstationary_Transformers)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)

