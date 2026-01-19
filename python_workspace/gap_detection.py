"""
制动间隙点检测模块

此模块提供制动间隙点识别功能，可独立启用或禁用。
制动间隙点是指电机电流发生突变的位置，通常对应于制动系统开始接触的时刻。

主要功能:
1. detect_gap_point: 检测电流序列中的突变点
2. compensate_displacement_gap: 将检测到的间隙点作为位移零点进行补偿
"""

import numpy as np


def moving_average(data, window):
    """移动平均滤波"""
    if window <= 1:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')


def detect_gap_point(iq_series, threshold_factor=2.0, window=5):
    """
    检测制动间隙点（电流突变点）
    
    参数:
        iq_series: 电流序列
        threshold_factor: 突变检测阈值倍数（相对于初始电流标准差）
        window: 滑动窗口大小
    
    返回:
        gap_index: 间隙点索引（电流第一次突变的位置）
    """
    # 计算电流的绝对值
    iq_abs = np.abs(iq_series)
    
    # 计算初始段的基线（前10%的数据）
    baseline_len = max(int(len(iq_abs) * 0.1), 50)
    baseline = iq_abs[:baseline_len]
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    
    # 如果标准差太小，使用固定阈值
    if baseline_std < 0.1:
        baseline_std = 0.1
    
    # 计算滑动窗口内的电流变化率
    iq_smooth = moving_average(iq_abs, window)
    delta_iq = np.diff(iq_smooth)
    
    # 寻找第一个显著突变点
    threshold = baseline_mean + threshold_factor * baseline_std
    
    # 从baseline之后开始检测
    search_start = baseline_len
    for i in range(search_start, len(iq_smooth) - 1):
        # 检测电流是否超过阈值，且有明显上升趋势
        if iq_smooth[i] > threshold and delta_iq[i-1] > 0:
            # 向前回溯找到实际起跳点
            for j in range(i, max(search_start - 10, 0), -1):
                if iq_smooth[j] <= baseline_mean + 0.5 * baseline_std:
                    return j
            return i
    
    # 如果没有检测到突变点，返回0（无间隙）
    return 0


def compensate_displacement_gap(x_series, iq_series, gap_detection_params=None):
    """
    补偿制动间隙：将电流突变点作为位移零点
    
    参数:
        x_series: 原始位移序列
        iq_series: 电流序列
        gap_detection_params: 间隙检测参数字典
            - threshold_factor: 阈值倍数（默认2.0）
            - window: 滑动窗口大小（默认5）
    
    返回:
        x_compensated: 补偿后的位移
        gap_index: 检测到的间隙点索引
        gap_value: 间隙大小（位移值）
    """
    if gap_detection_params is None:
        gap_detection_params = {'threshold_factor': 2.0, 'window': 5}
    
    # 检测间隙点
    gap_index = detect_gap_point(iq_series, **gap_detection_params)
    
    # 获取间隙大小（起始位移值）
    gap_value = x_series[gap_index] if gap_index > 0 else 0.0
    
    # 补偿位移：将间隙点作为零点
    x_compensated = x_series - gap_value
    
    # 确保位移非负
    x_compensated = np.maximum(x_compensated, 0.0)
    
    return x_compensated, gap_index, gap_value
