#!/bin/bash
echo "=== CUDA环境诊断 ==="

echo -e "\n1. nvcc版本："
which nvcc
nvcc --version

echo -e "\n2. CUDA_HOME设置："
echo "CUDA_HOME: $CUDA_HOME"

echo -e "\n3. 查找libcudart.so："
find $CONDA_PREFIX -name "libcudart.so*" 2>/dev/null | head -5

echo -e "\n4. 库文件是否在LD_LIBRARY_PATH中："
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -E "(cuda|nvidia)" || echo "未找到CUDA路径"

echo -e "\n5. CMake版本（建议<3.30）："
cmake --version

echo -e "\n6. 建议修复："
echo "export CUDA_HOME=$CONDA_PREFIX"
echo "export CUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX"
echo "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"

