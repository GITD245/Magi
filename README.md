### Docker版本安装

使用docker_manager安装 nvcr.io/nvidia/pytorch:24.03-py3

```baso
git clone https://github.com/GITD245/Magi.git
git clone https://github.com/GITD245/megatron-3.0.2
```

Magi 初始化

```bash
cd /workspace/Magi
pip install -r requirements.txt
# 编译 推荐第二个
python setup.py install
pip install -e .
bear -- pip install -e . # 使用clangd
```

启动megatron训练

```bash
cd /workspace/megatron-3.0.2
# 加快编译速度 export MAX_JOBS=36
bash examples/pretrain_gpt_distributed_magi.sh
```

### virtualenv安装

创建虚拟环境

```
virtualenv -p /usr/bin/python3.10 magi_env
source magi_env/bin/activate
git clone https://github.com/GITD245/Magi.git
git clone https://github.com/GITD245/megatron-3.0.2
```

Magi初始化

```bash
cd /workspace/Magi
pip install -r requirements.txt
# 最后的cu根据实际版本切换
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# virtualenv中无法使用pip install -e .
python setup.py install
```

安装apex

```bash
git clone https://github.com/NVIDIA/apex.git
cd /workspace/apex
# 可能需要注释掉setup.py中一段代码
python setup.py install --cuda_ext
```

安装megatron

可能会遇到这个问题：https://github.com/NVIDIA/Megatron-LM/issues/143
将python3-config替换为绝对路径可解决

第一次运行自动开始编译 编译卡住时删除 megatron-3.0.2/megatron/fused_kernels/build 文件夹重新编译

目前遇到过因NCCL原因编译hang住问题，尝试重启容器/开发机解决

```bash
cd /workspace/megatron-3.0.2
# 加快编译速度 export MAX_JOBS=36
bash examples/pretrain_gpt_distributed_magi.sh
```

### Other

FasterMoE
```bash
git checkout origin-fastmoe
FMOE_FASTER_SCHEDULE_ENABLE=1 FMOE_FASTER_SHADOW_ENABLE=1 bash examples/pretrain_gpt_distributed_faster.sh
```

nsys

```bash
nsys profile --output=my_report --stats=true --trace=cuda,cublas,cudnn examples/pretrain_gpt_distributed_magi.sh
```