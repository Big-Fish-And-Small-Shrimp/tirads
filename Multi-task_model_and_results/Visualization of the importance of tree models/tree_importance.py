import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from other_code.UMAP import clf, bst, gbdt


# ====================== 特征重要性提取函数 ======================
def extract_importances(models, feature_names):
    """
    models: 包含三个训练好的树模型(xgb, lgb, gbdt)的字典
    feature_names: 特征名称列表（对应8个软标签）
    返回：包含归一化特征重要性的DataFrame
    """
    importance_dict = {}

    # XGBoost特征重要性（基于F-score）
    xgb_imp = models['xgb'].feature_importances_
    importance_dict['XGBoost'] = xgb_imp / xgb_imp.sum()

    # LightGBM特征重要性（基于分裂次数）
    lgb_imp = models['lgb'].feature_importance(importance_type='split')
    importance_dict['LightGBM'] = lgb_imp / lgb_imp.sum()

    # GradientBoosting特征重要性（基于平均增益）
    gbdt_imp = models['gbdt'].feature_importances_
    importance_dict['GradientBoosting'] = gbdt_imp / gbdt_imp.sum()

    return pd.DataFrame(importance_dict, index=feature_names)


# ====================== 可视化函数 ======================
def plot_importances(importance_df, title="Feature Importance Comparison"):
    """
    绘制三个模型的特征重要性对比条形图
    """
    ax = importance_df.plot.bar(
        figsize=(12, 6),
        width=0.8,
        color=['#1f77b4', '#ff7f0e', '#2ca02c']
    )

    plt.title(title, fontsize=14)
    plt.ylabel('Normalized Importance', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1))

    # 添加数值标签
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 9),
            textcoords='offset points',
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig("feature_importance_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


# ====================== 主执行流程 ======================
if __name__ == "__main__":
    # 特征名称映射（根据您的label编号定义）
    feature_names = [
        'Label0 (3-class)',
        'Label1 (5-class)',
        'Label2 (binary)',
        'Label3 (4-class)',
        'Label4 (binary)',
        'Label5 (binary)',
        'Label6 (binary)',
        'Label7 (binary)'
    ]

    # 创建模型字典（确保模型已训练）
    trained_models = {
        'xgb': clf,
        'lgb': bst,
        'gbdt': gbdt
    }

    # 提取并归一化特征重要性
    importance_df = extract_importances(trained_models, feature_names)

    # 输出重要性表格
    print("=== 特征重要性量化结果 ===")
    print(importance_df.round(4))

    # 绘制对比图
    plot_importances(importance_df,
                     title="Feature Importance Comparison Across Tree Models")

    # 保存为CSV文件
    importance_df.to_csv("feature_importance_results.csv", float_format='%.4f')