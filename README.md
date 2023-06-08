# Confidence-Aware Bidirectional Offline Model-Based Imagination (CABI)

Code for Double Check Your State Before Trusting It: Confidence-Aware Bidirectional Offline Model-Based Imagination (NeurIPS 2022). The code is highly based on the [offlineRL](https://agit.ai/Polixir/OfflineRL) repository.

### NeoRL

```shell
git clone https://agit.ai/Polixir/neorl.git
cd neorl
pip install -e .
```

### D4RL
```shell
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```

For more details on use, please see [d4rl](https://github.com/rail-berkeley/d4rl).

## Install offlinerl

```shell
pip install -e .
```

## Example

```python
CUDA_VISIBLE_DEVICES=5 nohup python examples/train_d4rl.py --algo_name=td3_bc --task d4rl-halfcheetah-medium-v0 --seed 45 --real-data-ratio 0.7 --horizon 3 --log-dir ./logs/td3-bc-cabi/halfcheetah-medium-v0/r45 > out.log 2>&1 &
```

## Results on MuJoCo-v2 Environments
In this part, we present the results of CABI+TD3_BC on MuJoCo "-v2" datasets. The results are averaged over 4 random seeds.

| Task Name | TD3_BC | TD3_BC+CABI |
|-----------|--------|-------------|
|halfcheetah-random-v2| 11.0 $\pm$ 1.1 | **14.3 $\pm$ 0.4** |
|hopper-random-v2| 8.5 $\pm$ 0.6 | **15.7 $\pm$ 11.1** |
|walker2d-random-v2| 1.6 $\pm$ 1.7 | **6.0 $\pm$ 0.3** |
|halfcheetah-medium-v2| 48.3 $\pm$ 0.3 | **50.1 $\pm$ 0.2** |
|hopper-medium-v2| 59.3 $\pm$ 4.2 | **60.1 $\pm$ 0.8** |
|walker2d-medium-v2| 83.7 $\pm$ 2.1 | **84.3 $\pm$ 0.2** |
|halfcheetah-medium-replay-v2| 44.6 $\pm$ 0.5 |**46.2 $\pm$ 0.3** |
|hopper-medium-replay-v2| 60.9 $\pm$ 18.8 | **70.1 $\pm$ 13.0** |
|walker2d-medium-replay-v2| 81.8 $\pm$ 5.5 | **83.5 $\pm$ 3.2** |
|halfcheetah-medium-expert-v2| 90.7 $\pm$ 4.3 | **94.8 $\pm$ 1.0** |
|hopper-medium-expert-v2| 98.0 $\pm$ 9.4 | **111.4 $\pm$ 0.3** |
|walker2d-medium-expert-v2| 110.1 $\pm$ 0.5 | **110.5 $\pm$ 0.6** |
|halfcheetah-expert-v2| 96.7 $\pm$ 1.1 | **96.9 $\pm$ 1.4** |
|hopper-expert-v2| 107.8 $\pm$ 7 | **109.3 $\pm$ 1.2** |
|walker2d-expert-v2| 110.2 $\pm$ 0.3 | **110.5 $\pm$ 0.3** |

## Citation

If you use our method or code in your research, please consider citing the paper as follows:
```
@inproceedings{lyu2022doublecheck,
 title={Double Check Your State Before Trusting It: Confidence-Aware Bidirectional Offline Model-Based Imagination},
 author={Jiafei Lyu and Xiu Li and Zongqing Lu},
 booktitle={Thirty-sixth Conference on Neural Information Processing Systems},
 year={2022}
}
```

## Update Log

I do have noticed that the reported results in our original paper maybe out-of-date. I am now working on the experimental results on MuJoCo-v2 datasets, and Adroit-v1 datasets. I will append the results in this repo when the experiments are done.

2023/6/8, update results on mujoco-v2 datasets
