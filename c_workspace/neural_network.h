/*
 * 神经网络模型C语言实现
 * 模型: PI算子 + GRU残差网络
 */

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "model_params.h"

/* 数据结构定义 */
typedef struct {
    float x;      // 位移
    float iq;     // 电机q轴电流
    float F;      // 夹紧力 (真实值,用于验证)
} SampleData;

typedef struct {
    float data[SEQ_LEN][INPUT_SIZE];  // 输入序列缓冲区
    int len;                           // 当前序列长度
} SequenceBuffer;

typedef struct {
    float hidden[HIDDEN_SIZE];  // GRU隐藏状态
} GRUState;

/* 函数声明 */

/**
 * 移动平均滤波
 * @param input 输入数组
 * @param output 输出数组
 * @param len 数组长度
 * @param window 窗口大小
 */
void moving_average(const float *input, float *output, int len, int window);

/**
 * 计算数值梯度
 * @param input 输入数组
 * @param output 输出数组
 * @param len 数组长度
 * @param dt 时间步长
 */
void compute_gradient(const float *input, float *output, int len, float dt);

/**
 * Play算子 (Prandtl-Ishlinskii模型的基础单元)
 * @param x_series 输入位移序列
 * @param y_series 输出序列
 * @param len 序列长度
 * @param r 阈值参数
 */
void play_operator(const float *x_series, float *y_series, int len, float r);

/**
 * 物理约束: 限制夹紧力变化率和非负性
 * @param force 夹紧力序列
 * @param len 序列长度
 * @param dt 采样周期
 * @param max_rate 最大变化率 (N/s)
 * @param min_force 最小夹紧力 (N)
 */
void physical_force_constraint(float *force, int len, float dt, float max_rate, float min_force);

/**
 * Sigmoid激活函数
 */
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * Tanh激活函数
 */
static inline float tanh_activation(float x) {
    return tanhf(x);
}

/**
 * 计算PI基线预测
 * @param x_smooth 平滑后的位移序列
 * @param dx 位移导数
 * @param iq_smooth 平滑后的电流序列
 * @param diq 电流导数
 * @param output 输出的基线预测
 * @param len 序列长度
 */
void compute_pi_baseline(const float *x_smooth, const float *dx, 
                         const float *iq_smooth, const float *diq,
                         float *output, int len);

/**
 * 特征归一化
 * @param features 特征数组 [INPUT_SIZE]
 * @param normalized 归一化后的特征
 */
void normalize_features(const float *features, float *normalized);

/**
 * GRU前向传播 (单步)
 * @param input 当前时刻输入 [INPUT_SIZE]
 * @param state GRU状态
 * @return 输出值
 */
float gru_forward_step(const float *input, GRUState *state);

/**
 * GRU前向传播 (序列)
 * @param input_seq 输入序列 [SEQ_LEN][INPUT_SIZE]
 * @param seq_len 序列长度
 * @return 输出值 (最后一个时刻的输出)
 */
float gru_forward_sequence(const float input_seq[][INPUT_SIZE], int seq_len);

/**
 * 完整的模型推理
 * @param samples 输入样本数组
 * @param num_samples 样本数量
 * @param predictions 输出预测数组
 * @param window_size 移动平均窗口大小
 */
void model_inference(const SampleData *samples, int num_samples, 
                     float *predictions, int window_size);

/**
 * 计算评估指标
 * @param predictions 预测值数组
 * @param ground_truth 真实值数组
 * @param len 数组长度
 * @param mae 输出: 平均绝对误差
 * @param rmse 输出: 均方根误差
 * @param r2 输出: R²系数
 */
void compute_metrics(const float *predictions, const float *ground_truth, int len,
                     float *mae, float *rmse, float *r2);

#endif /* NEURAL_NETWORK_H */
