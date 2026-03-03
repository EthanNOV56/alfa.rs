# 构建与安装指南

## 系统要求

- **Rust**: 1.70+ (安装: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- **Python**: 3.8+
- **pip**: 最新版
- **C编译器**: gcc/clang (Rust 需要)

## 快速安装

```bash
# 1. 进入项目目录
cd /root/.openclaw/workspace/alpha-expr

# 2. 安装 Python 依赖
pip install -e .[dev]

# 3. 构建 Rust 扩展
maturin develop --release

# 4. 验证安装
python -c "import alpha_expr; print(alpha_expr.__version__)"
```

## 开发环境设置

```bash
# 安装所有开发工具
pip install -e .[dev]
pip install maturin

# 构建并安装开发版本
maturin develop

# 运行测试
pytest tests/

# 运行示例
python examples/basic_usage.py
```

## 项目结构

```
alpha-expr/
├── Cargo.toml              # Rust 配置
├── pyproject.toml          # Python 配置
├── src/lib.rs              # Rust 核心实现
├── alpha_expr/             # Python 包
│   ├── __init__.py         # 主模块
│   ├── _fallback.py        # Python 回退实现
│   └── _core.*.so          # Rust 扩展 (构建后生成)
├── examples/               # 使用示例
├── tests/                  # 单元测试
├── README.md              # 项目文档
└── BUILD.md              # 本文件
```

## Rust 扩展构建细节

### 手动构建

```bash
# 仅构建 Rust 库
cargo build --release

# 生成 Python 扩展
maturin build --release

# 直接安装
maturin install
```

### 调试构建

```bash
# 调试模式（更快构建）
maturin develop

# 发布模式（优化性能）
maturin develop --release
```

## 常见问题

### 1. 导入错误：找不到模块 `_core`

```
ImportError: cannot import name '_core' from 'alpha_expr'
```

**解决方案**：
```bash
# 确保已构建 Rust 扩展
maturin develop --release

# 或者使用纯 Python 回退模式
# 修改 alpha_expr/__init__.py 中的 HAS_RUST_EXT = True 为 False
```

### 2. Rust 编译错误

**解决方案**：
```bash
# 更新 Rust 工具链
rustup update

# 清理并重新构建
cargo clean
maturin develop --release
```

### 3. Python 版本不兼容

**解决方案**：
```bash
# 使用正确的 Python 版本
python3.8 -m pip install -e .[dev]

# 或者创建虚拟环境
python -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

### 4. 缺少系统依赖

**Ubuntu/Debian**:
```bash
sudo apt-get install python3-dev build-essential
```

**macOS**:
```bash
xcode-select --install
brew install python3
```

## 性能优化

### 构建选项

```bash
# 最大优化（推荐生产环境）
RUSTFLAGS="-C target-cpu=native" maturin develop --release

# 启用并行计算
export RAYON_NUM_THREADS=4  # 使用4个线程
```

### 内存优化

Rust 实现已针对内存效率进行优化：
- 使用 `ndarray` 进行零拷贝操作
- 避免不必要的内存分配
- 并行处理每日数据

## 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_basic.py::test_quantile_backtest_basic

# 带覆盖率报告
pytest --cov=alpha_expr tests/
```

## 发布

### 构建分发包

```bash
# 构建 wheel 包
maturin build --release

# 构建源分发
python -m build
```

### PyPI 发布

```bash
# 构建包
maturin build --release

# 上传到 PyPI
twine upload target/wheels/*
```

## 许可证

MIT License