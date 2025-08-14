```markdown
# CTC Decoder Speed Benchmark

这是一个用于比较 Python 与 C++ 实现的 CTC（Connectionist Temporal Classification）解码器性能差异的示例。

```
## 快速开始

### 1. 安装构建工具

```bash
pip install build
```

### 2. 构建 wheel 包

在项目根目录执行：

```bash
python -m build --wheel
```

构建完成后，会在 `dist/` 目录下生成类似如下的 wheel 文件：

```
dist/ctc_decoder-0.2.0-cp310-cp310-win_amd64.whl
```

> 实际文件名可能根据你的 Python 版本和操作系统有所不同。

### 3. 安装 wheel 包

```bash
pip install dist/ctc_decoder-0.2.0-cp310-cp310-win_amd64.whl
```

### 4. 运行性能测试

```bash
python main.py
```

运行后将会输出 Python 和 C++ 实现的 CTC 解码器在相同输入下的运行时间对比。

## 注意事项

- 本项目目前在Windows11、Python3.10环境下进行了测试，可以正常运行。
- 如需在其他平台使用，请在对应平台上重新构建 wheel 包。
