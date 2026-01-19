import os
import json
import numpy as np
import torch

def export_model_parameters(artifacts_dir="./artifacts", output_dir="../c_workspace"):
    """
    导出模型参数到C头文件和JSON文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载PI参数
    with open(os.path.join(artifacts_dir, "pi_params.json"), "r", encoding="utf-8") as f:
        pi_params = json.load(f)
    
    # 2. 加载特征归一化参数
    with open(os.path.join(artifacts_dir, "feature_norm.json"), "r", encoding="utf-8") as f:
        norm_params = json.load(f)
    
    # 注意: 间隙补偿参数已移除，相关功能在gap_detection模块中
    # 如需使用间隙补偿，请单独导入gap_detection.c模块
    
    # 3. 加载PyTorch模型权重
    model_path = os.path.join(artifacts_dir, "model_residual.pth")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    
    # 提取GRU参数
    # GRU权重结构: weight_ih_l0 [3*hidden, input], weight_hh_l0 [3*hidden, hidden]
    # 偏置: bias_ih_l0 [3*hidden], bias_hh_l0 [3*hidden]
    # 全连接层: fc.weight [1, hidden], fc.bias [1]
    
    weight_ih = state_dict['gru.weight_ih_l0'].numpy()  # [3*H, in_dim]
    weight_hh = state_dict['gru.weight_hh_l0'].numpy()  # [3*H, H]
    bias_ih = state_dict['gru.bias_ih_l0'].numpy()      # [3*H]
    bias_hh = state_dict['gru.bias_hh_l0'].numpy()      # [3*H]
    fc_weight = state_dict['fc.weight'].numpy()          # [1, H]
    fc_bias = state_dict['fc.bias'].numpy()              # [1]
    
    hidden_size = weight_hh.shape[1]
    input_size = weight_ih.shape[1]
    
    # GRU包含3个门: reset gate (r), update gate (z), new gate (n)
    # weight_ih = [W_ir | W_iz | W_in]  (按行堆叠)
    # weight_hh = [W_hr | W_hz | W_hn]  (按行堆叠)
    
    # 合并参数到一个字典
    all_params = {
        "model_type": "PI_GRU",
        "pi_params": pi_params,
        "norm_params": norm_params,
        "gru_params": {
            "input_size": int(input_size),
            "hidden_size": int(hidden_size),
            "weight_ih": weight_ih.tolist(),  # [3*H, in_dim]
            "weight_hh": weight_hh.tolist(),  # [3*H, H]
            "bias_ih": bias_ih.tolist(),      # [3*H]
            "bias_hh": bias_hh.tolist(),      # [3*H]
            "fc_weight": fc_weight.tolist(),  # [1, H]
            "fc_bias": fc_bias.tolist()       # [1]
        },
        "checkpoint_params": {
            "r_mean": float(checkpoint.get("r_mean", 0.0)),
            "r_std": float(checkpoint.get("r_std", 1.0)),
            "seq_len": int(checkpoint.get("seq_len", 32))
        }
    }
    
    # 保存为JSON文件（仅保存到artifacts目录）
    json_path_artifacts = os.path.join(artifacts_dir, "model_params.json")
    with open(json_path_artifacts, "w", encoding="utf-8") as f:
        json.dump(all_params, f, indent=2)
    print(f"已保存JSON参数文件: {json_path_artifacts}")
    
    # 生成C头文件
    generate_c_header(all_params, output_dir)
    
    return all_params

def generate_c_header(params, output_dir):
    """
    生成C语言头文件
    """
    header_path = os.path.join(output_dir, "model_params.h")
    
    pi = params["pi_params"]
    norm = params["norm_params"]
    gru = params["gru_params"]
    ckpt = params["checkpoint_params"]
    
    H = gru["hidden_size"]
    In = gru["input_size"]
    M = pi["M"]
    
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("/*\n")
        f.write(" * 自动生成的神经网络模型参数头文件\n")
        f.write(" * 模型: PI算子 + GRU残差网络\n")
        f.write(" */\n\n")
        f.write("#ifndef MODEL_PARAMS_H\n")
        f.write("#define MODEL_PARAMS_H\n\n")
        
        # 定义常量
        f.write("/* 模型维度参数 */\n")
        f.write(f"#define INPUT_SIZE {In}\n")
        f.write(f"#define HIDDEN_SIZE {H}\n")
        f.write(f"#define NUM_PLAY_OPERATORS {M}\n")
        f.write(f"#define SEQ_LEN {ckpt['seq_len']}\n\n")
        
        # PI参数
        f.write("/* PI算子参数 */\n")
        f.write(f"#define DT {pi['dt']}\n")
        f.write(f"#define LAG_SAMPLES {pi['lag_samples_x_to_F']}\n")
        f.write(f"#define BETA_DX {pi['beta_dx']}\n")
        f.write(f"#define ALPHA_IQ {pi['alpha_iq']}\n")
        f.write(f"#define BETA_DIQ {pi['beta_diq']}\n")
        f.write(f"#define PI_BIAS {pi['bias']}\n")
        f.write(f"#define USE_DIQ {1 if pi['use_diq'] else 0}\n")
        f.write(f"#define IQ_SCALE {pi['iq_scale']}\n\n")
        
        # 残差归一化参数
        f.write("/* 残差归一化参数 */\n")
        f.write(f"#define RESIDUAL_MEAN {ckpt['r_mean']}\n")
        f.write(f"#define RESIDUAL_STD {ckpt['r_std']}\n\n")
        
        # PI算子的r值数组
        f.write("/* PI算子的r值数组 */\n")
        f.write(f"static const float r_list[{M}] = {{\n")
        for i, r in enumerate(pi['r_list']):
            if i % 4 == 0:
                f.write("    ")
            f.write(f"{r:.10f}f")
            if i < M - 1:
                f.write(", ")
            if (i + 1) % 4 == 0 or i == M - 1:
                f.write("\n")
        f.write("};\n\n")
        
        # PI算子权重
        f.write("/* PI算子权重 w_P */\n")
        f.write(f"static const float w_P[{M}] = {{\n")
        for i, w in enumerate(pi['w_P']):
            if i % 4 == 0:
                f.write("    ")
            f.write(f"{w:.10f}f")
            if i < M - 1:
                f.write(", ")
            if (i + 1) % 4 == 0 or i == M - 1:
                f.write("\n")
        f.write("};\n\n")
        
        # 特征归一化参数
        f.write("/* 特征归一化参数 */\n")
        f.write(f"static const float f_min[{In}] = {{\n")
        for i, v in enumerate(norm['f_min']):
            if i % 4 == 0:
                f.write("    ")
            f.write(f"{v:.10f}f")
            if i < In - 1:
                f.write(", ")
            if (i + 1) % 4 == 0 or i == In - 1:
                f.write("\n")
        f.write("};\n\n")
        
        f.write(f"static const float f_max[{In}] = {{\n")
        for i, v in enumerate(norm['f_max']):
            if i % 4 == 0:
                f.write("    ")
            f.write(f"{v:.10f}f")
            if i < In - 1:
                f.write(", ")
            if (i + 1) % 4 == 0 or i == In - 1:
                f.write("\n")
        f.write("};\n\n")
        
        # GRU权重矩阵
        write_matrix(f, "weight_ih_l0", gru['weight_ih'], [3*H, In])
        write_matrix(f, "weight_hh_l0", gru['weight_hh'], [3*H, H])
        write_vector(f, "bias_ih_l0", gru['bias_ih'], 3*H)
        write_vector(f, "bias_hh_l0", gru['bias_hh'], 3*H)
        
        # 全连接层权重
        write_vector(f, "fc_weight", gru['fc_weight'][0], H)  # [1, H] -> [H]
        f.write(f"static const float fc_bias = {gru['fc_bias'][0]:.10f}f;\n\n")
        
        f.write("#endif /* MODEL_PARAMS_H */\n")
    
    print(f"已保存C头文件: {header_path}")

def write_matrix(f, name, data, shape):
    """写入矩阵到C头文件"""
    rows, cols = shape
    f.write(f"/* {name}: [{rows} x {cols}] */\n")
    f.write(f"static const float {name}[{rows}][{cols}] = {{\n")
    for i, row in enumerate(data):
        f.write("    {")
        for j, val in enumerate(row):
            f.write(f"{val:.10f}f")
            if j < len(row) - 1:
                f.write(", ")
        f.write("}")
        if i < len(data) - 1:
            f.write(",")
        f.write("\n")
    f.write("};\n\n")

def write_vector(f, name, data, size):
    """写入向量到C头文件"""
    f.write(f"/* {name}: [{size}] */\n")
    f.write(f"static const float {name}[{size}] = {{\n")
    for i, val in enumerate(data):
        if i % 4 == 0:
            f.write("    ")
        f.write(f"{val:.10f}f")
        if i < size - 1:
            f.write(", ")
        if (i + 1) % 4 == 0 or i == size - 1:
            f.write("\n")
    f.write("};\n\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="导出模型参数到C头文件")
    parser.add_argument("--artifacts", type=str, default="./artifacts", help="模型参数所在目录")
    parser.add_argument("--output", type=str, default="../c_workspace", help="输出目录")
    args = parser.parse_args()
    
    print("开始导出模型参数...")
    params = export_model_parameters(args.artifacts, args.output)
    print(f"\n导出完成!")
    print(f"  - JSON文件: {os.path.join(args.artifacts, 'model_params.json')}")
    print(f"  - C头文件: {os.path.join(args.output, 'model_params.h')}")
    print(f"\n模型信息:")
    print(f"  - 输入维度: {params['gru_params']['input_size']}")
    print(f"  - 隐藏层大小: {params['gru_params']['hidden_size']}")
    print(f"  - PI算子数量: {params['pi_params']['M']}")
    print(f"  - 序列长度: {params['checkpoint_params']['seq_len']}")
