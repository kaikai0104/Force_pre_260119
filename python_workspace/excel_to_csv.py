"""
将Excel文件转换为CSV格式，用于C程序读取
"""
import pandas as pd
import os
import sys

def excel_to_csv(excel_path, csv_path=None, skip_existing=True):
    """
    将Excel文件转换为CSV格式
    
    参数:
        excel_path: Excel文件路径
        csv_path: 输出CSV文件路径（可选）
        skip_existing: 如果CSV文件已存在，是否跳过转换（默认True）
    """
    if csv_path is None:
        csv_path = excel_path.replace('.xlsx', '.csv').replace('.xls', '.csv')
    
    # 检查CSV文件是否已存在
    if skip_existing and os.path.exists(csv_path):
        print(f"[跳过] {os.path.basename(csv_path)} 已存在")
        return csv_path
    
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path, header=0, engine='openpyxl')
        
        # 只保留前3列
        df = df.iloc[:, :3].copy()
        df.columns = ['x', 'iq', 'F']
        
        # 删除无效数据
        df = df.replace([float('inf'), float('-inf')], float('nan')).dropna()
        
        # 保存为CSV
        df.to_csv(csv_path, index=False)
        
        print(f"[转换完成] {os.path.basename(excel_path)} -> {os.path.basename(csv_path)}")
        print(f"  样本数: {len(df)}")
        
        return csv_path
    
    except Exception as e:
        print(f"[错误] 转换失败 {excel_path}: {str(e)}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_to_csv.py <excel_file> [output_csv]")
        print("\nExample:")
        print("  python excel_to_csv.py data/1.xlsx")
        print("  python excel_to_csv.py data/1.xlsx output.csv")
        return
    
    excel_path = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) >= 3 else None
    
    if not os.path.exists(excel_path):
        print(f"Error: File {excel_path} not found")
        return
    
    excel_to_csv(excel_path, csv_path)

if __name__ == "__main__":
    main()
