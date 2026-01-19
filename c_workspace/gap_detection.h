/*
 * 制动间隙点检测模块
 * 
 * 此模块提供制动间隙点识别功能，可独立启用或禁用。
 * 制动间隙点是指电机电流发生突变的位置，通常对应于制动系统开始接触的时刻。
 * 
 * 主要功能:
 * 1. detect_gap_point: 检测电流序列中的突变点
 * 2. compensate_displacement_gap: 将检测到的间隙点作为位移零点进行补偿
 */

#ifndef GAP_DETECTION_H
#define GAP_DETECTION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * 检测制动间隙点（电流突变点）
 * @param iq_series 电流序列
 * @param len 序列长度
 * @param threshold_factor 阈值倍数
 * @param window 滑动窗口大小
 * @return 间隙点索引
 */
int detect_gap_point(const float *iq_series, int len, float threshold_factor, int window);

/**
 * 补偿制动间隙
 * @param x_series 原始位移序列
 * @param iq_series 电流序列
 * @param len 序列长度
 * @param threshold_factor 阈值倍数
 * @param window 滑动窗口大小
 * @param gap_value 输出：检测到的间隙大小
 * @return 间隙点索引
 */
int compensate_displacement_gap(float *x_series, const float *iq_series, int len,
                                float threshold_factor, int window, float *gap_value);

#endif /* GAP_DETECTION_H */
