import torch

# ✅ 加载超分模型
def load_sr_model(model_name="FSRCNN",upscalefactor=4, device="cpu"):
    """
    加载指定的超分辨率模型

    :param model_name: 选择的超分模型，例如 "FSRCNN", "EDSR", "RCAN"
    :param device: 运行设备 ("cpu" 或 "cuda")
    :return: 加载好的模型
    """

    # ✅ 预定义模型权重路径
    model_paths = {
        "FSRCNN": "./vsrmodel/model_zoo/fsrcnn_x4-T91-97a30bfb.pth.tar",
        # "EDSR": "./vsrmodel/model_zoo/edsr_baseline.pth",
        # "RCAN": "./vsrmodel/model_zoo/rcan.pth"
    }

    if model_name not in model_paths:
        raise ValueError(f"❌ 不支持的模型: {model_name}")

    weight_path = model_paths[model_name]

    # ✅ 选择模型
    model = None
    if model_name == "FSRCNN":
        from vsrmodel.models.fsrcnn import FSRCNN
        model = FSRCNN(upscale_factor= upscalefactor).to(device)
    # elif model_name == "EDSR":
    #     from vsrmodel.models.edsr import EDSR
    #     model = EDSR().to(device)
    # elif model_name == "RCAN":
    #     from vsrmodel.models.rcan import RCAN
    #     model = RCAN().to(device)

    # ✅ 加载预训练权重
    weight = torch.load(weight_path, map_location=device)
    if "state_dict" in weight:
        model.load_state_dict(weight["state_dict"])
    else:
        model.load_state_dict(weight)

    model.eval()  # 设为推理模式
    print(f"✅ 成功加载模型: {model_name} ({weight_path}) 到 {device}")

    return model