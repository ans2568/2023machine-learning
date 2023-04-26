# 2023machine-learning
기계학습 수업 중 사용한 모델

### 사용한 Docker Image : nvcr.io/nvidia/pytorch:21.08-py3

```bash
docker run -it --privileged --network=host --name=machine_learning --env=NVIDIA_VISIBLE_DEVICES=all --runtime=nvidia --env=NVIDIA_DRIVER_CAPABILITIES=all --gpus=all nvcr.io/nvidia/pytorch:21.08-py3
```

### efficient net Github : https://github.com/katsura-jp/efficientnet-pytorch

### efficient net 을 제외한 network Github : https://github.com/weiaicunzai/pytorch-cifar100
