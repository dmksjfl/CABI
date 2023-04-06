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
