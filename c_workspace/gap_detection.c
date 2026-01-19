/*
 * 制动间隙点检测模块实现
 */

#include "gap_detection.h"

/* 移动平均滤波 - 本地实现 */
static void gap_moving_average(const float *input, float *output, int len, int window) {
    if (window <= 1) {
        for (int i = 0; i < len; i++) {
            output[i] = input[i];
        }
        return;
    }
    
    int half_win = window / 2;
    for (int i = 0; i < len; i++) {
        int start = (i - half_win < 0) ? 0 : i - half_win;
        int end = (i + half_win >= len) ? len - 1 : i + half_win;
        
        float sum = 0.0f;
        int count = 0;
        for (int j = start; j <= end; j++) {
            sum += input[j];
            count++;
        }
        output[i] = sum / count;
    }
}

/* 检测制动间隙点 */
int detect_gap_point(const float *iq_series, int len, float threshold_factor, int window) {
    if (len < 100) return 0;  // 数据太少，无法检测
    
    // 计算电流绝对值
    float *iq_abs = (float *)malloc(len * sizeof(float));
    for (int i = 0; i < len; i++) {
        iq_abs[i] = fabsf(iq_series[i]);
    }
    
    // 计算基线（前10%的数据）
    int baseline_len = len / 10;
    if (baseline_len < 50) baseline_len = 50;
    if (baseline_len > len / 2) baseline_len = len / 2;
    
    float baseline_sum = 0.0f;
    for (int i = 0; i < baseline_len; i++) {
        baseline_sum += iq_abs[i];
    }
    float baseline_mean = baseline_sum / baseline_len;
    
    // 计算基线标准差
    float baseline_var = 0.0f;
    for (int i = 0; i < baseline_len; i++) {
        float diff = iq_abs[i] - baseline_mean;
        baseline_var += diff * diff;
    }
    float baseline_std = sqrtf(baseline_var / baseline_len);
    
    // 最小标准差
    if (baseline_std < 0.1f) {
        baseline_std = 0.1f;
    }
    
    // 应用移动平均滤波
    float *iq_smooth = (float *)malloc(len * sizeof(float));
    gap_moving_average(iq_abs, iq_smooth, len, window);
    
    // 计算阈值
    float threshold = baseline_mean + threshold_factor * baseline_std;
    
    // 从baseline之后开始检测
    int gap_index = 0;
    for (int i = baseline_len; i < len - 1; i++) {
        // 检测电流超过阈值且有上升趋势
        float delta = iq_smooth[i] - iq_smooth[i - 1];
        if (iq_smooth[i] > threshold && delta > 0) {
            // 向前回溯找实际起跳点
            for (int j = i; j >= baseline_len - 10 && j >= 0; j--) {
                if (iq_smooth[j] <= baseline_mean + 0.5f * baseline_std) {
                    gap_index = j;
                    break;
                }
            }
            if (gap_index == 0) gap_index = i;
            break;
        }
    }
    
    free(iq_abs);
    free(iq_smooth);
    
    return gap_index;
}

/* 补偿制动间隙 */
int compensate_displacement_gap(float *x_series, const float *iq_series, int len,
                                float threshold_factor, int window, float *gap_value) {
    // 检测间隙点
    int gap_index = detect_gap_point(iq_series, len, threshold_factor, window);
    
    // 获取间隙大小
    *gap_value = (gap_index > 0) ? x_series[gap_index] : 0.0f;
    
    // 补偿位移：减去间隙值
    for (int i = 0; i < len; i++) {
        x_series[i] -= *gap_value;
        // 确保非负
        if (x_series[i] < 0.0f) {
            x_series[i] = 0.0f;
        }
    }
    
    return gap_index;
}
