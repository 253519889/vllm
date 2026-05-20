#!/bin/bash
# 保存为 clean_env.sh
# chmod +x clean_env.sh
# ./clean_env.sh

echo "=== 开始清理conda环境 ==="

# 退出所有环境
conda deactivate
conda deactivate 2>/dev/null
conda deactivate 2>/dev/null

# 删除旧的vllm环境
conda remove -n vllm --all -y 2>/dev/null
conda remove -n vllm-perfect --all -y 2>/dev/null
conda remove -n vllm-stable --all -y 2>/dev/null

# 清理pip和conda缓存
conda clean --all -y
pip cache purge 2>/dev/null

# 清理vLLM构建文件
cd ~/vllm
rm -rf build/ dist/ .eggs/ *.egg-info .deps/ CMakeCache.txt CMakeFiles/
rm -rf ~/.cache/vllm 2>/dev/null
rm -rf ~/.cache/torch_extensions 2>/dev/null
rm -rf ~/.nv/ComputeCache/* 2>/dev/null

echo "=== 清理完成 ==="
echo "现在可以运行安装脚本了"
