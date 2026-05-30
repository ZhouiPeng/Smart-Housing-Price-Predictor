# 成员C：模型构建、评估与解释

## 1. 工作输入

成员C直接使用成员B输出的处理后数据：

- `data/processed/X_train.csv`、`X_test.csv`
- `data/processed/y_train.csv`、`y_test.csv`
- 预测目标使用 `LogSalePrice`，同时在结果中还原为美元口径 RMSE，便于解释真实房价误差。

## 2. 模型方案

本次训练并比较了线性回归、Ridge、Lasso、随机森林和梯度提升树。Ridge、Lasso、随机森林、梯度提升树均通过 3 折交叉验证选择参数。本次运行已包含 XGBoost 模型。

## 3. 模型性能结论

最优模型为 **Ridge**。

- 测试集 log RMSE：0.1246
- 测试集 R2：0.9167
- 测试集还原房价 RMSE：约 $22,730
- 泛化差距 log RMSE：0.0235

完整模型对比表见 `results/memberC/model_performance.csv`，调参结果见 `results/memberC/tuning_results.json`。

## 4. 最优模型参数

```json
{
  "model__alpha": "100.0"
}
```

## 5. 模型解释

最优模型解释部分使用模型系数/特征重要性作为可复现解释图；XGBoost 另行使用 SHAP 做专门解释。

已额外使用 XGBoost 模型生成 SHAP 全局解释图与单样本解释图，可直接放入汇报材料。

重要特征前 8 位为：GrLivArea, OverallQual, RoofMatl_ClyTile, BsmtFinSF1, GarageCars, OverallCond, HouseAge, Condition2_PosN。

这些变量整体符合房地产定价直觉：房屋整体质量、居住面积、车库容量、地下室面积、浴室数量和建造/翻新时间等变量通常直接影响购房者对房屋功能性和品质的判断。

## 6. 误差分析

误差诊断图见 `figures/memberC/residual_diagnostics.png`，最大误差样本见 `results/memberC/largest_prediction_errors.csv`。从残差图可以重点检查高价房样本是否存在系统性低估，因为高端房屋往往受到位置、装修、稀缺设施等非线性因素影响，单纯依靠结构化变量仍可能不够。

## 7. 可交付文件清单

- `scripts/memberC_modeling.py`：成员C建模与解释脚本
- `results/memberC/model_performance.csv`：模型性能对比表
- `results/memberC/tuning_results.json`：交叉验证调参结果
- `results/memberC/feature_importance.csv`：特征重要性结果
- `results/memberC/xgboost_shap_importance.csv`：XGBoost SHAP 平均绝对贡献排序
- `results/memberC/largest_prediction_errors.csv`：高误差样本
- `figures/memberC/model_comparison_rmse.png`
- `figures/memberC/actual_vs_predicted.png`
- `figures/memberC/residual_diagnostics.png`
- `figures/memberC/feature_importance_top20.png`
- `figures/memberC/shap_summary_top20.png`
- `figures/memberC/xgboost_shap_summary_top20.png`
- `figures/memberC/xgboost_shap_single_sample.png`
