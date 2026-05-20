#!/bin/bash
# 保存为 install_vllm_master.sh
# chmod +x install_vllm_master.sh
# ./install_vllm_master.sh

set -e  # 遇到错误立即退出

echo "========================================="
echo "   vLLM Master分支一键安装脚本"
echo "   8x RTX 3090 | 256GB RAM | 96核"
echo "========================================="

# ===== 1. 创建新环境 =====
echo "[1/10] 创建conda环境..."
conda create -n vllm-master python=3.10 -y

# 激活环境（使用完整的conda路径）
CONDA_BASE=$(conda info --base)
conda activate vllm-master

# ===== 2. 安装CUDA 12.1工具包 =====
echo "[2/10] 安装CUDA 12.1工具包..."
conda install -c nvidia cuda=12.1.0 -y

# ===== 3. 安装GCC 11.4（CUDA 12.1的最佳伴侣）=====
echo "[3/10] 安装GCC 11.4编译器..."
conda install -c conda-forge gxx_linux-64=11.4.0 gcc_linux-64=11.4.0 -y

# 创建软链接
cd $CONDA_PREFIX/bin
ln -sf x86_64-conda-linux-gnu-g++ g++ 2>/dev/null || true
ln -sf x86_64-conda-linux-gnu-gcc gcc 2>/dev/null || true
cd - > /dev/null

# ===== 4. 设置环境变量 =====
echo "[4/10] 设置环境变量..."
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export CUDAHOSTCXX=$CXX
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ===== 5. 安装PyTorch 2.5.1（最新版，支持CUDA 12.1）=====
echo "[5/10] 安装PyTorch 2.5.1..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# ===== 6. 安装编译依赖 =====
echo "[6/10] 安装编译依赖..."
pip install ninja==1.11.1.1
pip install packaging setuptools-scm wheel
pip install cmake==3.29.6
pip install numpy pybind11

# ===== 7. 获取最新的master分支 =====
echo "[7/10] 获取vLLM最新master分支..."
cd ~
if [ -d "vllm" ]; then
    cd vllm
    git fetch origin
    git reset --hard origin/main
    git pull
else
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
fi

# 记录当前commit
echo "当前commit: $(git rev-parse HEAD)"

# ===== 8. 设置编译参数（针对8x RTX 3090优化）=====
echo "[8/10] 设置编译参数..."
export MAX_JOBS=32  # 96核的一半，避免资源竞争
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 3090的计算能力
export VLLM_CUDA_ARCH="8.6"
export CMAKE_BUILD_TYPE=Release
export VERBOSE=1

# 为了解决cumem_allocator.cpp中的fabric API问题
# 最新master可能已经修复，如果还报错，启用下面这行
export CXXFLAGS="-D__CUDA_API_VERSION_INTERNAL"

# ===== 9. 编译安装 =====
echo "[9/10] 开始编译安装（这可能需要15-30分钟）..."
echo "编译日志将保存到 vllm_build.log"

# 清理旧的构建文件
rm -rf build/ dist/ .eggs/ *.egg-info .deps/

# 开始编译并保存日志
time pip install -e . -v 2>&1 | tee vllm_build.log

# ===== 10. 验证安装 =====
echo "[10/10] 验证安装..."
python -c "
import torch
import vllm
print('✅ PyTorch版本:', torch.__version__)
print('✅ CUDA可用:', torch.cuda.is_available())
print('✅ GPU数量:', torch.cuda.device_count())
print('✅ vLLM版本:', vllm.__version__)
print('✅ vLLM路径:', vllm.__file__)
print('\n🎉 安装成功！')
"

echo "========================================="
echo "   安装完成！"
echo "   环境名称: vllm-master"
echo "   激活命令: conda activate vllm-master"
echo "========================================="
