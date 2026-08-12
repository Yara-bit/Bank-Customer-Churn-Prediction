# 银行客户流失预警与精准营销系统 (Bank Customer Churn Prediction)

基于 **GBDT 梯度提升决策树** 与 **Streamlit** 打造的端到端银行客户流失预警与数字化营销工作台。系统不仅具备客户流失概率预测功能，还能结合业务规则自动输出客户经理干预 SOP 建议。

## 🌟 项目亮点

* **端到端机器学习 Pipeline**：整合数据清洗、数据泄漏（Leakage）特征剔除、独热编码（One-Hot Encoding）与衍生特征工程。
* **业务驱动的最佳阈值**：打破传统 0.5 默认分类阈值，基于成本-收益权衡寻优得出 **0.3659 最佳决策阈值**，大幅提升高危流失客户的召回率（Recall）。
* **自动化营销 SOP 建议**：针对预测高风险客户，自动识别“产品结构风险”、“中年高危期”、“高资沉睡”等具体场景并推导干预策略。
* **交互式 Web 工作台**：使用 Streamlit 搭建轻量级决策界面，支持多维度特征实时调参与风险毫秒级预测。
* **跨环境工程兼容排查**：内置 Scikit-Learn 跨版本（CyHalfBinomialLoss）反序列化与 Pickle 作用域动态补丁，确保生产与云端环境稳定运行。

## 📂 项目结构说明

```text
├── Customer-Churn-Records.csv   # 银行客户基础数据集(来自Kaggle数据集https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)
├── bank.ipynb                   # 数据探索 (EDA)、特征工程与模型建模 Notebook
├── churn_pipeline_aligned.pkl   # 导出的完整机器学习 Pipeline
├── gb_model.pkl                 # GBDT 模型权重备份
├── app.py                       # Streamlit Web 应用服务主程序
├── requirements.txt             # 项目环境依赖包列表
└── README.md                    # 项目说明文档
