# 🎯 安全帽检测系统 (Safety Helmet Detection System)

基于先进的YOLOv8模型构建的安全帽检测系统，提供高精度的实时检测能力和友好的Web交互界面。

## 获取地址
[https://mbd.pub/o/bread/YZWVmZhrbQ==](https://mbd.pub/o/bread/YZWVmZhrbQ==)
## 系统展示
[https://player.bilibili.com/player.html?aid=114880946380173]


## 🌟 系统特性

- **高精度检测**: 基于YOLOv8模型，mAP50达到96.53%
- **多模式检测**: 支持单图检测、批量检测、实时摄像头检测
- **友好界面**: 现代化的Web界面，支持拖拽上传
- **GPU加速**: 支持CUDA加速训练和推理
- **完整流程**: 从数据集处理到模型训练再到系统部署的全流程解决方案
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5f9ddf3f2b4a45ae9259d69d5dab2697.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4e5725c49fff4fa58c9f45cb0d7ab52a.png#pic_center)![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/73699eb5162243aebdb1e440bc09b6c1.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5116f8a95d784a50b19dd501cb5cc8fa.png#pic_center)


## 📋 目录结构

```
HSRS/
├── VOC2028/                    # 原始VOC数据集
│   ├── Annotations/           # XML标注文件
│   ├── JPEGImages/           # 图片文件
│   └── ImageSets/            # 数据集分割信息
├── yolo_dataset/              # 转换后的YOLO格式数据集
│   ├── images/
│   │   ├── train/            # 训练图片
│   │   └── val/              # 验证图片
│   ├── labels/
│   │   ├── train/            # 训练标签
│   │   └── val/              # 验证标签
│   └── dataset.yaml          # 数据集配置文件
├── hat_detection/             # 训练输出目录
│   └── yolov8_hat_model_gpu/ # 训练好的模型
│       └── weights/
│           ├── best.pt       # 最佳模型
│           └── last.pt       # 最新模型
├── dataset_manager.py         # 统一数据集管理工具
├── train_gpu.py              # GPU训练脚本
├── detection_model.py         # 检测模型封装
├── app.py                    # Flask Web应用
├── run.py                    # 一键启动脚本
└── requirements.txt          # 依赖包列表
```

## 🚀 快速开始

### 1. 环境准备

#### 系统要求
- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8+
- **GPU**: NVIDIA GPU (推荐，支持CUDA)
- **内存**: 8GB+ RAM
- **存储**: 10GB+ 可用空间



# 安装依赖
pip install -r requirements.txt
```

#### GPU环境配置（可选但推荐）

```bash
# 检查CUDA版本
nvidia-smi

# 安装CUDA版本的PyTorch
python install_gpu_pytorch.py
```

### 2. 数据集处理

#### 自动处理（推荐）
```bash
# 一键处理数据集
python dataset_manager.py
```


### 3. 模型训练

#### GPU训练（推荐）
```bash
# 使用GPU训练
python train_gpu.py
```

#### CPU训练
```bash
# 使用CPU训练
python train_model.py
```

#### 训练参数说明
- **epochs**: 训练轮数（默认100）
- **batch_size**: 批次大小（GPU自动调整）
- **img_size**: 图片尺寸（默认640）
- **device**: 训练设备（自动检测GPU/CPU）

### 4. 系统部署

#### 一键启动
```bash
# 完整流程：数据集处理 + 模型训练 + Web服务
python run.py
```

#### 分步启动
```bash
# 仅启动Web服务
python app.py
```

#### 访问系统
打开浏览器访问: `http://localhost:5000`

## 📊 模型性能

### 训练结果
- **mAP50**: 96.53%
- **mAP50-95**: 80.01%
- **Precision**: 95.14%
- **Recall**: 93.63%

### 检测能力
- **检测类别**: 安全帽 (hat)
- **置信度阈值**: 0.5
- **推理速度**: ~20ms/帧 (GPU)
- **支持格式**: JPG, PNG, BMP

## 🎮 使用指南

### Web界面功能

#### 1. 单图检测
- 点击"选择图片"或拖拽图片到上传区域
- 系统自动检测并显示结果
- 支持结果图片下载

#### 2. 批量检测
- 选择包含多张图片的文件夹
- 系统批量处理并生成报告
- 支持结果打包下载

#### 3. 实时检测
- 点击"开启摄像头"
- 实时显示检测结果
- 支持截图保存

### API接口

#### 单图检测
```bash
curl -X POST http://localhost:5000/api/detect_image \
  -F "image=@test.jpg"
```

#### 批量检测
```bash
curl -X POST http://localhost:5000/api/detect_batch \
  -F "folder=@test_folder"
```

## 🔧 故障排除

### 常见问题

#### 1. 数据集问题
```bash
# 检查数据集完整性
python dataset_manager.py --action check

# 修复数据集
python dataset_manager.py --action fix
```

#### 2. GPU相关问题
```bash
# 检查GPU环境
python -c "import torch; print(torch.cuda.is_available())"

# 重新安装GPU PyTorch
python install_gpu_pytorch.py
```

#### 3. 模型加载问题
```bash
# 检查模型文件
ls hat_detection/yolov8_hat_model_gpu/weights/

# 重新训练模型
python train_gpu.py
```

#### 4. Web服务问题
```bash
# 检查端口占用
netstat -an | grep 5000

# 更换端口
python app.py --port 5001
```

### 错误代码说明

| 错误代码 | 说明 | 解决方案 |
|---------|------|----------|
| `MODEL_NOT_FOUND` | 模型文件未找到 | 检查模型路径或重新训练 |
| `DATASET_ERROR` | 数据集错误 | 运行数据集修复脚本 |
| `GPU_UNAVAILABLE` | GPU不可用 | 检查CUDA安装或使用CPU |
| `PORT_OCCUPIED` | 端口被占用 | 更换端口或关闭占用进程 |

## 📈 性能优化

### 训练优化
1. **数据增强**: 启用更多数据增强策略
2. **学习率调整**: 使用更激进的学习率衰减
3. **模型集成**: 训练多个模型进行集成
4. **超参数调优**: 使用网格搜索优化参数

### 推理优化
1. **模型量化**: 使用INT8量化减少模型大小
2. **TensorRT加速**: 使用TensorRT进行推理加速
3. **批处理**: 批量处理提高吞吐量
4. **缓存机制**: 缓存检测结果减少重复计算

## 🔒 安全考虑

### 数据安全
- 上传的图片仅用于检测，不会永久存储
- 支持图片自动清理机制
- 可配置数据保留时间

### 系统安全
- 支持HTTPS部署
- 可配置访问控制
- 支持API密钥认证

