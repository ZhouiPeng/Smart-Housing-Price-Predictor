from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "results" / "memberC"
FIGURE_DIR = ROOT / "figures" / "memberC"
MODEL_DIR = ROOT / "models"
DOC_DIR = ROOT / "docs"
RANDOM_STATE = 42
CV_FOLDS = 3


def ensure_dirs() -> None:
    for directory in [RESULT_DIR, FIGURE_DIR, MODEL_DIR, DOC_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def dollar_rmse(log_true: np.ndarray, log_pred: np.ndarray) -> float:
    return rmse(np.expm1(log_true), np.expm1(log_pred))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train_raw = pd.read_csv(DATA_DIR / "y_train.csv")
    y_test_raw = pd.read_csv(DATA_DIR / "y_test.csv")

    y_train = y_train_raw["LogSalePrice"]
    y_test = y_test_raw["LogSalePrice"]
    price_train = y_train_raw["SalePrice"]
    price_test = y_test_raw["SalePrice"]
    return X_train, X_test, y_train, y_test, price_train, price_test


def optional_xgboost_model() -> tuple[str, Any, dict[str, list[Any]]] | None:
    try:
        from xgboost import XGBRegressor
    except Exception:
        return None

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_estimators=400,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        n_jobs=1,
    )
    grid = {
        "max_depth": [2, 3],
        "learning_rate": [0.03, 0.05],
        "n_estimators": [300, 500],
    }
    return "XGBoost", model, grid


def build_model_specs() -> list[tuple[str, Any, dict[str, list[Any]] | None]]:
    specs: list[tuple[str, Any, dict[str, list[Any]] | None]] = [
        (
            "Linear Regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ]
            ),
            None,
        ),
        (
            "Ridge",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge(random_state=RANDOM_STATE)),
                ]
            ),
            {"model__alpha": [1.0, 10.0, 30.0, 100.0]},
        ),
        (
            "Lasso",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Lasso(random_state=RANDOM_STATE, max_iter=20000)),
                ]
            ),
            {"model__alpha": [0.001, 0.003, 0.01, 0.03]},
        ),
        (
            "Random Forest",
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            {
                "n_estimators": [120],
                "max_depth": [None, 16],
                "min_samples_leaf": [1, 2],
            },
        ),
        (
            "Gradient Boosting",
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [160, 260],
                "learning_rate": [0.04, 0.07],
                "max_depth": [2, 3],
            },
        ),
    ]

    xgb = optional_xgboost_model()
    if xgb is not None:
        specs.append(xgb)
    return specs


def fit_or_tune(name: str, estimator: Any, grid: dict[str, list[Any]] | None, X: pd.DataFrame, y: pd.Series) -> tuple[Any, dict[str, Any], float]:
    if grid is None:
        scores = cross_val_score(estimator, X, y, cv=CV_FOLDS, scoring="neg_root_mean_squared_error", n_jobs=1)
        estimator.fit(X, y)
        return estimator, {}, float(-scores.mean())

    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring="neg_root_mean_squared_error",
        cv=CV_FOLDS,
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y)
    best_params = {k: str(v) for k, v in search.best_params_.items()}
    return search.best_estimator_, best_params, float(-search.best_score_)


def evaluate_model(name: str, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return {
        "model": name,
        "train_rmse_log": rmse(y_train, train_pred),
        "test_rmse_log": rmse(y_test, test_pred),
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "test_rmse_dollar": dollar_rmse(y_test.to_numpy(), test_pred),
        "generalization_gap_log_rmse": rmse(y_test, test_pred) - rmse(y_train, train_pred),
    }


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    ordered = metrics.sort_values("test_rmse_log")
    sns.barplot(data=ordered, x="test_rmse_log", y="model", color="#4C78A8")
    plt.xlabel("Test RMSE on log(SalePrice)")
    plt.ylabel("")
    plt.title("Model comparison")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison_rmse.png", dpi=180)
    plt.close()


def plot_predictions(best_name: str, y_test: pd.Series, pred: np.ndarray) -> None:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=pred, s=28, alpha=0.72)
    low = min(float(y_test.min()), float(pred.min()))
    high = max(float(y_test.max()), float(pred.max()))
    plt.plot([low, high], [low, high], color="#D62728", linewidth=1.8)
    plt.xlabel("Actual log(SalePrice)")
    plt.ylabel("Predicted log(SalePrice)")
    plt.title(f"Actual vs predicted: {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close()


def plot_residuals(best_name: str, pred: np.ndarray, residuals: np.ndarray) -> None:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=pred, y=residuals, s=28, alpha=0.72, color="#59A14F")
    plt.axhline(0, color="#D62728", linewidth=1.5)
    plt.xlabel("Predicted log(SalePrice)")
    plt.ylabel("Residual: actual - predicted")
    plt.title(f"Residual diagnostics: {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "residual_diagnostics.png", dpi=180)
    plt.close()


def extract_tree_importance(model: Any, feature_names: list[str]) -> pd.DataFrame | None:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif isinstance(model, Pipeline) and hasattr(model[-1], "coef_"):
        importances = np.abs(model[-1].coef_)
    else:
        return None
    return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)


def plot_importance(importance: pd.DataFrame, path: Path, title: str) -> None:
    top = importance.head(20).iloc[::-1]
    plt.figure(figsize=(8, 7))
    sns.barplot(data=top, x="importance", y="feature", color="#F58518")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_interpretation(best_model: Any, X_test: pd.DataFrame, y_test: pd.Series, pred: np.ndarray) -> tuple[pd.DataFrame, str]:
    feature_names = X_test.columns.tolist()
    importance = extract_tree_importance(best_model, feature_names)
    method = "model_importance"

    if importance is None:
        perm = permutation_importance(
            best_model,
            X_test,
            y_test,
            scoring="neg_root_mean_squared_error",
            n_repeats=12,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance", ascending=False)
        method = "permutation_importance"

    importance.to_csv(RESULT_DIR / "feature_importance.csv", index=False)
    plot_importance(importance, FIGURE_DIR / "feature_importance_top20.png", "Top feature importance")

    try:
        import shap

        sample = X_test.sample(min(250, len(X_test)), random_state=RANDOM_STATE)
        explainer = shap.Explainer(best_model.predict, sample)
        shap_values = explainer(sample)
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "shap_summary_top20.png", dpi=180, bbox_inches="tight")
        plt.close()
        method = "shap"
    except Exception:
        plot_importance(
            importance,
            FIGURE_DIR / "shap_summary_top20.png",
            "SHAP fallback: feature importance",
        )

    residuals = y_test.to_numpy() - pred
    errors = pd.DataFrame(
        {
            "actual_log_price": y_test,
            "predicted_log_price": pred,
            "residual_log": residuals,
            "absolute_error_log": np.abs(residuals),
            "actual_price": np.expm1(y_test),
            "predicted_price": np.expm1(pred),
        },
        index=X_test.index,
    )
    errors = errors.sort_values("absolute_error_log", ascending=False)
    errors.head(15).to_csv(RESULT_DIR / "largest_prediction_errors.csv", index_label="row_index")
    return importance, method


def make_xgboost_shap(xgb_model: Any | None, X_test: pd.DataFrame) -> bool:
    if xgb_model is None:
        return False

    try:
        import shap

        sample = X_test.sample(min(250, len(X_test)), random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(sample)

        mean_abs = np.abs(shap_values.values).mean(axis=0)
        shap_rank = pd.DataFrame(
            {"feature": sample.columns, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        shap_rank.to_csv(RESULT_DIR / "xgboost_shap_importance.csv", index=False)

        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "xgboost_shap_summary_top20.png", dpi=180, bbox_inches="tight")
        plt.close()

        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "xgboost_shap_single_sample.png", dpi=180, bbox_inches="tight")
        plt.close()
        return True
    except Exception as exc:
        (RESULT_DIR / "xgboost_shap_error.txt").write_text(str(exc), encoding="utf-8")
        return False


def write_report(
    metrics: pd.DataFrame,
    best_name: str,
    best_params: dict[str, Any],
    importance: pd.DataFrame,
    explain_method: str,
    xgb_shap_done: bool,
) -> None:
    best_row = metrics.loc[metrics["model"] == best_name].iloc[0]
    top_features = importance.head(8)["feature"].tolist()
    xgb_note = (
        "本环境未安装 xgboost，因此脚本使用 sklearn 的 GradientBoostingRegressor 作为梯度提升树基线；"
        "若安装 xgboost，脚本会自动加入 XGBoost 模型并重新比较。"
        if "XGBoost" not in set(metrics["model"])
        else "本次运行已包含 XGBoost 模型。"
    )
    shap_note = (
        "最优模型解释图使用 SHAP 生成。"
        if explain_method == "shap"
        else "最优模型解释部分使用模型系数/特征重要性作为可复现解释图；XGBoost 另行使用 SHAP 做专门解释。"
    )
    xgb_shap_note = (
        "已额外使用 XGBoost 模型生成 SHAP 全局解释图与单样本解释图，可直接放入汇报材料。"
        if xgb_shap_done
        else "未能生成 XGBoost 专属 SHAP 图，详情可查看 `results/memberC/xgboost_shap_error.txt`。"
    )

    report = f"""# 成员C：模型构建、评估与解释

## 1. 工作输入

成员C直接使用成员B输出的处理后数据：

- `data/processed/X_train.csv`、`X_test.csv`
- `data/processed/y_train.csv`、`y_test.csv`
- 预测目标使用 `LogSalePrice`，同时在结果中还原为美元口径 RMSE，便于解释真实房价误差。

## 2. 模型方案

本次训练并比较了线性回归、Ridge、Lasso、随机森林和梯度提升树。Ridge、Lasso、随机森林、梯度提升树均通过 {CV_FOLDS} 折交叉验证选择参数。{xgb_note}

## 3. 模型性能结论

最优模型为 **{best_name}**。

- 测试集 log RMSE：{best_row["test_rmse_log"]:.4f}
- 测试集 R2：{best_row["test_r2"]:.4f}
- 测试集还原房价 RMSE：约 ${best_row["test_rmse_dollar"]:,.0f}
- 泛化差距 log RMSE：{best_row["generalization_gap_log_rmse"]:.4f}

完整模型对比表见 `results/memberC/model_performance.csv`，调参结果见 `results/memberC/tuning_results.json`。

## 4. 最优模型参数

```json
{json.dumps(best_params, ensure_ascii=False, indent=2)}
```

## 5. 模型解释

{shap_note}

{xgb_shap_note}

重要特征前 8 位为：{", ".join(top_features)}。

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
"""
    (DOC_DIR / "memberC_model_analysis.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    X_train, X_test, y_train, y_test, _, _ = load_data()

    metrics: list[dict[str, Any]] = []
    tuning: dict[str, Any] = {}
    models: dict[str, Any] = {}

    for name, estimator, grid in build_model_specs():
        model, params, cv_rmse = fit_or_tune(name, estimator, grid, X_train, y_train)
        row = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        row["cv_rmse_log"] = cv_rmse
        metrics.append(row)
        tuning[name] = {"best_params": params, "best_cv_rmse_log": cv_rmse}
        models[name] = model

    metrics_df = pd.DataFrame(metrics).sort_values("test_rmse_log")
    metrics_df.to_csv(RESULT_DIR / "model_performance.csv", index=False)
    (RESULT_DIR / "tuning_results.json").write_text(
        json.dumps(tuning, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_name = str(metrics_df.iloc[0]["model"])
    best_model = models[best_name]
    best_pred = best_model.predict(X_test)
    joblib.dump(best_model, MODEL_DIR / "memberC_best_model.joblib")
    if "XGBoost" in models:
        joblib.dump(models["XGBoost"], MODEL_DIR / "memberC_xgboost_model.joblib")

    plot_model_comparison(metrics_df)
    plot_predictions(best_name, y_test, best_pred)
    plot_residuals(best_name, best_pred, y_test.to_numpy() - best_pred)
    importance, explain_method = make_interpretation(best_model, X_test, y_test, best_pred)
    xgb_shap_done = make_xgboost_shap(models.get("XGBoost"), X_test)
    write_report(metrics_df, best_name, tuning[best_name]["best_params"], importance, explain_method, xgb_shap_done)

    print(metrics_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Report: {DOC_DIR / 'memberC_model_analysis.md'}")


if __name__ == "__main__":
    main()
