# FlashAttention NPU Standalone

这个目录提供一个**独立的 FA NPU 最小工程**，包含：

- NPU C++实现（来自 `csrc/flash_attn_npu`）
- 独立编译脚本（`scripts/build.sh`）
- 独立测试脚本（`scripts/test.sh`）
- Python 接口（`flash_attn_npu.flash_attn_with_kvcache`）

## 目录结构

```text
fa_npu_standalone/
├── csrc/flash_attn_npu/      # NPU kernel 实现
├── flash_attn_npu/           # Python 最小封装
├── tests/                    # NPU 测试
├── scripts/build.sh          # 编译脚本
├── scripts/test.sh           # 测试脚本
└── setup.py                  # 独立构建入口
```

## 依赖

- PyTorch
- torch_npu
- Ascend Toolkit（含 `bisheng`）
- pytest

并确保：

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend
```

## 编译

```bash
cd fa_npu_standalone
./scripts/build.sh
```

## 运行测试

```bash
cd fa_npu_standalone
./scripts/test.sh
```

## 说明

- 构建逻辑参考根目录 `setup.py` 的 NPU 分支。
- 测试逻辑参考 `tests/test_flash_attn_npu.py`，并以最小形式保留核心校验。
- `setup.py` 默认包含 `../csrc/catlass/include` 头文件路径（与主仓一致）。
