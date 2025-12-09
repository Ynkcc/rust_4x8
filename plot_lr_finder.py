#!/usr/bin/env python3
# plot_lr_finder.py - 学习率扫描结果可视化脚本
#
# 使用方法:
#   python3 plot_lr_finder.py
#
# 功能:
# 1. 读取 lr_finder_results.csv 文件
# 2. 绘制学习率-损失曲线
# 3. 标注重要的点（最小损失、最陡下降）
# 4. 给出学习率选择建议

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def find_steepest_descent(lr, loss):
    """找到损失下降最陡的区间"""
    log_lr = np.log(lr)
    gradients = np.gradient(loss, log_lr)
    steepest_idx = np.argmin(gradients)
    return steepest_idx, gradients[steepest_idx]

def main():
    # 检查文件是否存在
    csv_file = 'lr_finder_results.csv'
    if not os.path.exists(csv_file):
        print(f"❌ 错误: 找不到文件 '{csv_file}'")
        print("请先运行学习率扫描器:")
        print("  cargo run --bin banqi-lr-finder")
        sys.exit(1)
    
    # 读取数据
    print(f"读取数据: {csv_file}")
    df = pd.read_csv(csv_file)
    
    if len(df) == 0:
        print("❌ 错误: CSV文件为空")
        sys.exit(1)
    
    print(f"✓ 成功读取 {len(df)} 个数据点")
    
    # 提取数据
    lr = df['learning_rate'].values
    loss = df['loss'].values
    policy_loss = df['policy_loss'].values
    value_loss = df['value_loss'].values
    
    # 找到关键点
    min_loss_idx = np.argmin(loss)
    min_loss_lr = lr[min_loss_idx]
    min_loss = loss[min_loss_idx]
    
    steepest_idx, steepest_gradient = find_steepest_descent(lr, loss)
    steepest_lr = lr[steepest_idx]
    steepest_loss = loss[steepest_idx]
    
    # 推荐学习率
    suggested_min_lr = steepest_lr
    suggested_max_lr = min_loss_lr / 3.0
    suggested_initial_lr = np.sqrt(suggested_min_lr * suggested_max_lr)
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 第一个图：总损失
    ax1 = axes[0]
    ax1.plot(lr, loss, 'b-', linewidth=2, label='Total Loss')
    ax1.axvline(min_loss_lr, color='g', linestyle='--', alpha=0.7, 
                label=f'Min Loss (LR={min_loss_lr:.2e})')
    ax1.axvline(steepest_lr, color='orange', linestyle='--', alpha=0.7,
                label=f'Steepest Descent (LR={steepest_lr:.2e})')
    ax1.axvline(suggested_initial_lr, color='r', linestyle='--', alpha=0.7,
                label=f'Suggested LR={suggested_initial_lr:.2e}')
    
    # 标注关键点
    ax1.scatter([min_loss_lr], [min_loss], color='g', s=100, zorder=5)
    ax1.scatter([steepest_lr], [steepest_loss], color='orange', s=100, zorder=5)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Learning Rate Finder - Total Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # 第二个图：策略损失和价值损失
    ax2 = axes[1]
    ax2.plot(lr, policy_loss, 'r-', linewidth=2, label='Policy Loss', alpha=0.7)
    ax2.plot(lr, value_loss, 'b-', linewidth=2, label='Value Loss', alpha=0.7)
    ax2.axvline(suggested_initial_lr, color='k', linestyle='--', alpha=0.5,
                label=f'Suggested LR={suggested_initial_lr:.2e}')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Learning Rate', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Learning Rate Finder - Policy vs Value Loss', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    plt.tight_layout()
    
    # 保存图片
    output_file = 'lr_finder_plot.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {output_file}")
    
    # 打印分析结果
    print("\n" + "="*60)
    print("学习率扫描分析结果")
    print("="*60)
    
    print("\n📊 关键点:")
    print(f"  最小损失点: LR = {min_loss_lr:.2e}, Loss = {min_loss:.4f}")
    print(f"  最陡下降点: LR = {steepest_lr:.2e}, Loss = {steepest_loss:.4f}")
    print(f"              梯度 = {steepest_gradient:.4f}")
    
    print("\n💡 推荐学习率:")
    print(f"  初始学习率: {suggested_initial_lr:.2e}")
    print(f"  最小学习率: {suggested_min_lr:.2e} (用于学习率调度)")
    print(f"  最大学习率: {suggested_max_lr:.2e} (用于循环学习率)")
    
    print("\n📈 使用建议:")
    print(f"  1. 固定学习率训练:")
    print(f"     learning_rate = {suggested_initial_lr:.2e}")
    print(f"  ")
    print(f"  2. 指数衰减:")
    print(f"     initial_lr = {suggested_initial_lr:.2e}")
    print(f"     decay_rate = 0.95  # 每轮衰减5%")
    print(f"  ")
    print(f"  3. 余弦退火:")
    print(f"     max_lr = {suggested_initial_lr:.2e}")
    print(f"     min_lr = {suggested_min_lr:.2e}")
    print(f"  ")
    print(f"  4. 循环学习率 (CLR):")
    print(f"     base_lr = {suggested_min_lr:.2e}")
    print(f"     max_lr = {suggested_max_lr:.2e}")
    
    print("\n⚠️ 注意事项:")
    print("  - 这些是建议值，实际训练时需要根据验证集表现调整")
    print("  - 如果训练不稳定（损失爆炸），降低学习率（除以2-10）")
    print("  - 如果收敛太慢，可以尝试稍微增大学习率")
    print("  - Adam优化器通常对学习率不太敏感，可以从建议值开始")
    
    print("\n" + "="*60)
    
    # 显示图表
    plt.show()

if __name__ == '__main__':
    main()
