"""
完整自动化工作流脚本
功能：
1. 训练Python神经网络模型
2. 导出模型参数到C语言头文件
3. 转换Excel数据为CSV
4. 编译C程序
5. 运行C程序进行验证
6. 绘制C程序预测结果与真实值的对比图表

使用方法：
    python auto_workflow.py

可选参数：
    --epochs N          训练轮数 (默认: 20)
    --hidden N          GRU隐藏层大小 (默认: 16)
    --test-data FILE    测试数据文件 (默认: data/1.xlsx)
    --skip-train        跳过训练步骤（使用已有模型）
    --skip-compile      跳过编译步骤（使用已有可执行文件）
"""

import os
import sys
import subprocess
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# 导入excel_to_csv模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from excel_to_csv import excel_to_csv

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Colors:
    """终端颜色"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_step(step_num, description):
    """打印步骤标题"""
    print(f"\n{'='*70}")
    print(f"  步骤 {step_num}: {description}")
    print(f"{'='*70}")

def run_command(cmd, description, cwd=None, shell=True, realtime=False):
    """运行命令并打印输出"""
    print(f"\n> {description}")
    print(f"  命令: {cmd}\n")
    
    try:
        if realtime:
            # 实时输出模式 - 用于训练等长时间运行的命令
            process = subprocess.Popen(
                cmd,
                shell=shell,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='gbk',  # Windows中文系统使用gbk编码
                errors='replace',  # 替换无法解码的字符
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时打印输出
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
            except UnicodeDecodeError as e:
                print(f"\n[警告] 编码警告: {e}")
            
            process.wait()
            
            if process.returncode != 0:
                print(f"\n[错误] 命令执行失败 (返回码: {process.returncode})")
                return False
        else:
            # 缓冲输出模式 - 用于快速命令
            result = subprocess.run(
                cmd, 
                shell=shell, 
                cwd=cwd,
                capture_output=True, 
                text=True, 
                encoding='gbk',  # Windows中文系统使用gbk编码
                errors='replace'  # 替换无法解码的字符
            )
            
            if result.stdout:
                print(result.stdout)
            
            if result.returncode != 0:
                print(f"\n[错误] 命令执行失败 (返回码: {result.returncode})")
                if result.stderr:
                    print(f"错误信息:\n{result.stderr}")
                return False
        
        print(f"[OK] {description} - 完成")
        return True
    
    except Exception as e:
        print(f"\n[错误] 异常: {str(e)}")
        return False

def check_files_exist(files, description):
    """检查文件是否存在"""
    print(f"\n> 检查{description}...")
    all_exist = True
    for file in files:
        if os.path.exists(file):
            print(f"  [OK] {file}")
        else:
            print(f"  [X] {file} - 不存在")
            all_exist = False
    return all_exist

def plot_comparison(csv_file, output_dir):
    """
    读取C程序验证结果，绘制预测值与真实值的对比图
    """
    print_step("最终", "绘制验证结果对比图表")
    
    if not os.path.exists(csv_file):
        print(f"[错误] 结果文件不存在: {csv_file}")
        return False
    
    try:
        # 读取CSV数据
        print(f"\n> 读取验证结果: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"  数据点数量: {len(df)}")
        
        # 提取数据
        time = df['Time'].values
        true_values = df['true_value'].values
        pred_values = df['predicted_value'].values
        errors = df['error'].values
        
        # 计算统计指标
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(np.abs(errors))
        
        # 计算R²
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((true_values - np.mean(true_values))**2) + 1e-12
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"\n> 验证指标:")
        print(f"  MAE  (平均绝对误差):     {mae:.3f} N")
        print(f"  RMSE (均方根误差):       {rmse:.3f} N")
        print(f"  最大误差:                {max_error:.3f} N")
        print(f"  R2 系数:                 {r2:.6f}")
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # ========== 图1: 预测值 vs 真实值对比 ==========
        ax1 = axes[0]
        
        # 限制显示点数（如果数据太多）
        max_points = 5000
        if len(time) > max_points:
            step = len(time) // max_points
            time_plot = time[::step]
            true_plot = true_values[::step]
            pred_plot = pred_values[::step]
            print(f"  数据点过多，采样显示 {len(time_plot)} 个点")
        else:
            time_plot = time
            true_plot = true_values
            pred_plot = pred_values
        
        ax1.plot(time_plot, true_plot, 'b-', linewidth=1.5, label='真实值 (外置力传感器)', alpha=0.8)
        ax1.plot(time_plot, pred_plot, 'r--', linewidth=1.2, label='预测值 (C语言神经网络)', alpha=0.9)
        
        ax1.set_xlabel('时间 (秒)', fontsize=12)
        ax1.set_ylabel('夹紧力 (N)', fontsize=12)
        ax1.set_title('C语言神经网络模型验证结果 - 预测值与真实值对比', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 添加统计信息文本框
        textstr = f'MAE = {mae:.2f} N\nRMSE = {rmse:.2f} N\nR² = {r2:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        # ========== 图2: 误差分布 ==========
        ax2 = axes[1]
        
        if len(time) > max_points:
            errors_plot = errors[::step]
            time_error_plot = time_plot
        else:
            errors_plot = errors
            time_error_plot = time
        
        ax2.plot(time_error_plot, errors_plot, 'g-', linewidth=1, alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.axhline(y=mae, color='r', linestyle='--', linewidth=1, label=f'MAE = ±{mae:.2f} N')
        ax2.axhline(y=-mae, color='r', linestyle='--', linewidth=1)
        
        ax2.fill_between(time_error_plot, -mae, mae, alpha=0.2, color='green', label='MAE范围')
        
        ax2.set_xlabel('时间 (秒)', fontsize=12)
        ax2.set_ylabel('预测误差 (N)', fontsize=12)
        ax2.set_title('预测误差随时间变化', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(output_dir, f'c_validation_result_{timestamp}.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n[OK] 图表已保存: {plot_file}")
        
        # 同时保存一个汇总JSON文件
        summary = {
            "timestamp": timestamp,
            "data_points": len(df),
            "metrics": {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2),
                "Max_Error": float(max_error)
            },
            "data_range": {
                "true_min": float(np.min(true_values)),
                "true_max": float(np.max(true_values)),
                "pred_min": float(np.min(pred_values)),
                "pred_max": float(np.max(pred_values))
            }
        }
        
        summary_file = os.path.join(output_dir, f'c_validation_summary_{timestamp}.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] 汇总数据已保存: {summary_file}")
        
        return True
    
    except Exception as e:
        print(f"\n[错误] 绘图失败 - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='自动化神经网络模型工作流')
    parser.add_argument('--epochs', type=int, default=0, help='训练轮数 (0=交互式输入)')
    parser.add_argument('--hidden', type=int, default=16, help='GRU隐藏层大小')
    parser.add_argument('--test-data', type=str, default='data/1.xlsx', help='测试数据文件')
    parser.add_argument('--skip-train', action='store_true', help='跳过训练步骤')
    parser.add_argument('--skip-compile', action='store_true', help='跳过编译步骤')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  神经网络模型自动化工作流")
    print("  Python训练 -> C语言实现 -> 验证 -> 可视化")
    print("="*70)
    
    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_workspace = script_dir
    c_workspace = os.path.join(os.path.dirname(script_dir), "c_workspace")
    artifacts_dir = os.path.join(python_workspace, "artifacts")
    results_dir = os.path.join(python_workspace, "results")
    
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # ==================== 步骤1: 训练模型 ====================
    if not args.skip_train:
        print_step(1, "训练Python神经网络模型")
        
        # 如果epochs为0或未指定，请求用户输入
        epochs = args.epochs
        if epochs <= 0:
            print("\n请输入训练轮数（建议值：5-50）")
            print("  5-10   : 快速测试（约2-5分钟）")
            print("  10-20  : 中等训练（约5-10分钟）")
            print("  20-50  : 完整训练（约10-25分钟）")
            while True:
                try:
                    user_input = input("\n请输入训练轮数 [默认: 10]: ").strip()
                    if user_input == '':
                        epochs = 10
                        break
                    epochs = int(user_input)
                    if epochs > 0:
                        break
                    else:
                        print("[错误] 训练轮数必须大于0，请重新输入")
                except ValueError:
                    print("[错误] 请输入有效的数字")
            
            print(f"\n已设置训练轮数: {epochs}")
        
        train_cmd = (
            f'python train_schemeC.py '
            f'--inputs "data/*.xlsx" '
            f'--epochs {epochs} '
            f'--hidden {args.hidden} '
            f'--apply_physical_constraint'
        )
        
        # 使用实时输出模式，这样可以看到训练进度
        if not run_command(train_cmd, "训练神经网络模型", cwd=python_workspace, realtime=True):
            print("\n[错误] 训练失败！")
            return 1
        
        # 检查训练输出
        required_files = [
            os.path.join(artifacts_dir, "pi_params.json"),
            os.path.join(artifacts_dir, "feature_norm.json"),
            os.path.join(artifacts_dir, "model_residual.pth")
        ]
        
        if not check_files_exist(required_files, "训练输出文件"):
            print("\n[错误] 训练文件不完整！")
            return 1
    else:
        print_step(1, "跳过训练步骤（使用已有模型）")
    
    # ==================== 步骤2: 导出模型参数 ====================
    print_step(2, "导出模型参数到C语言")
    
    export_cmd = (
        f'python export_model_to_c.py '
        f'--artifacts "{artifacts_dir}" '
        f'--output "{c_workspace}"'
    )
    
    if not run_command(export_cmd, "导出模型参数", cwd=python_workspace):
        print("\n[错误] 导出失败！")
        return 1
    
    # 检查导出文件（JSON在artifacts，头文件在c_workspace）
    export_files = [
        os.path.join(artifacts_dir, "model_params.json"),
        os.path.join(c_workspace, "model_params.h")
    ]
    
    if not check_files_exist(export_files, "导出文件"):
        print("\n[错误] 导出文件不完整！")
        return 1
    
    # ==================== 步骤3: 转换测试数据 ====================
    print_step(3, "转换测试数据为CSV格式")
    
    data_folder = os.path.join(python_workspace, "data")
    data_csv_folder = os.path.join(python_workspace, "data_csv")
    
    # 创建data_csv文件夹
    os.makedirs(data_csv_folder, exist_ok=True)
    
    # 查找所有Excel文件
    excel_files = []
    if os.path.exists(data_folder):
        for file in os.listdir(data_folder):
            if file.endswith(('.xlsx', '.xls')):
                excel_files.append(file)
    
    if not excel_files:
        print(f"\n[错误] 在 {data_folder} 中未找到Excel文件")
        return 1
    
    print(f"\n> 找到 {len(excel_files)} 个Excel文件")
    
    # 转换所有Excel文件（跳过已存在的CSV文件）
    csv_files = []
    converted_count = 0
    skipped_count = 0
    
    for excel_file in excel_files:
        excel_path = os.path.join(data_folder, excel_file)
        csv_filename = os.path.splitext(excel_file)[0] + '.csv'
        csv_path = os.path.join(data_csv_folder, csv_filename)
        
        # 检查CSV文件是否已存在
        if os.path.exists(csv_path):
            print(f"  [跳过] {csv_filename} 已存在")
            csv_files.append(csv_filename)
            skipped_count += 1
            continue
        
        # 转换Excel到CSV
        print(f"  [转换] {excel_file} -> {csv_filename}")
        result = excel_to_csv(excel_path, csv_path, skip_existing=False)
        
        if result and os.path.exists(csv_path):
            csv_files.append(csv_filename)
            converted_count += 1
        else:
            print(f"  [警告] 转换 {excel_file} 失败，跳过")
    
    if not csv_files:
        print("\n[错误] 没有成功转换任何CSV文件")
        return 1
    
    print(f"\n[OK] CSV文件准备完成: {len(csv_files)} 个文件")
    print(f"  - 新转换: {converted_count} 个")
    print(f"  - 已存在跳过: {skipped_count} 个")
    
    # ==================== 步骤4: 编译C程序 ====================
    if not args.skip_compile:
        print_step(4, "编译C语言程序")
        
        # 检查是否有GCC
        try:
            subprocess.run(['gcc', '--version'], capture_output=True, check=True)
            has_gcc = True
        except:
            has_gcc = False
        
        if not has_gcc:
            print("\n[错误] 未找到GCC编译器")
            print("请安装MinGW-w64或其他C编译器")
            return 1
        
        compile_cmd = (
            'gcc -std=c99 -O2 -Wall -Wextra '
            '-o force_predictor.exe main.c neural_network.c -lm'
        )
        
        if not run_command(compile_cmd, "编译C程序", cwd=c_workspace):
            print("\n[错误] 编译失败！")
            return 1
        
        exe_file = os.path.join(c_workspace, "force_predictor.exe")
        if not os.path.exists(exe_file):
            print(f"\n[错误] 可执行文件生成失败: {exe_file}")
            return 1
    else:
        print_step(4, "跳过编译步骤（使用已有可执行文件）")
    
    # ==================== 步骤5: 运行C程序验证 ====================
    print_step(5, "运行C程序进行验证")
    
    # 列出所有可用的CSV文件
    print(f"\n可用的验证数据集（位于 data_csv/ 文件夹）:")
    for i, csv_file in enumerate(csv_files, 1):
        csv_path = os.path.join(data_csv_folder, csv_file)
        if os.path.exists(csv_path):
            # 获取文件大小
            file_size = os.path.getsize(csv_path)
            size_kb = file_size / 1024
            print(f"  [{i}] {csv_file:<20} ({size_kb:.1f} KB)")
    
    # 请求用户选择
    selected_csv = None
    while True:
        try:
            user_choice = input(f"\n请选择验证数据集 [1-{len(csv_files)}，默认: 1]: ").strip()
            if user_choice == '':
                selected_index = 0
                break
            choice_num = int(user_choice)
            if 1 <= choice_num <= len(csv_files):
                selected_index = choice_num - 1
                break
            else:
                print(f"[错误] 请输入 1 到 {len(csv_files)} 之间的数字")
        except ValueError:
            print("[错误] 请输入有效的数字")
    
    selected_csv = csv_files[selected_index]
    selected_csv_path = os.path.join(data_csv_folder, selected_csv)
    
    print(f"\n已选择数据集: {selected_csv}")
    
    # 准备输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_basename = os.path.splitext(selected_csv)[0]
    c_results_csv = os.path.join(results_dir, f"c_validation_{csv_basename}_{timestamp}.csv")
    
    run_cmd = f'force_predictor.exe "{selected_csv_path}" "{c_results_csv}"'
    
    # 使用实时输出，可以看到验证进度
    if not run_command(run_cmd, "C程序验证", cwd=c_workspace, realtime=True):
        print("\n[错误] 验证失败！")
        return 1
    
    if not os.path.exists(c_results_csv):
        print(f"\n[错误] 结果文件生成失败: {c_results_csv}")
        return 1
    
    # ==================== 步骤6: 绘制对比图表 ====================
    if not plot_comparison(c_results_csv, results_dir):
        print("\n[错误] 绘图失败！")
        return 1
    
    # ==================== 完成 ====================
    print("\n" + "="*70)
    print("  [OK] 所有步骤完成！")
    print("="*70)
    print(f"\n生成的文件:")
    print(f"  1. 模型参数:   {os.path.join(c_workspace, 'model_params.h')}")
    print(f"  2. JSON参数:   {os.path.join(c_workspace, 'model_params.json')}")
    print(f"  3. CSV数据集:  {data_csv_folder}/ ({len(csv_files)} 个文件)")
    print(f"  4. 验证数据:   {selected_csv}")
    print(f"  5. 验证结果:   {c_results_csv}")
    print(f"  6. 对比图表:   {results_dir}/*.png")
    print(f"  7. 汇总数据:   {results_dir}/*.json")
    
    print(f"\n查看结果:")
    print(f"  - 图表保存在: {results_dir}")
    print(f"  - 可以打开PNG文件查看可视化结果")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[警告] 用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[错误] 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
