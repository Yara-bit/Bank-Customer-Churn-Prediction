# app.py - 银行客户流失预警工作台
import sys
import __main__
import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------------------------------------------------------------
# 1. 修复模块映射 (解决 joblib 反序列化 ModuleNotFoundError: No module named '_loss')
# ------------------------------------------------------------------------------
try:
    import sklearn._loss
    sys.modules['_loss'] = sklearn._loss
except ImportError:
    try:
        import sklearn.ensemble._gb_losses as _loss
        sys.modules['_loss'] = _loss
    except ImportError:
        pass

# ------------------------------------------------------------------------------
# 2. 定义自定义转换器并挂载至 __main__ 作用域
# ------------------------------------------------------------------------------
class BankPipelineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_columns=None):
        self.target_columns = target_columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = X.copy()
        drop_leakage_cols = ['Complain', 'Satisfaction Score']
        X_out = X_out.drop(columns=[c for c in drop_leakage_cols if c in X_out.columns], errors='ignore')
        
        X_out['Balance_Salary_Ratio'] = X_out['Balance'] / (X_out['EstimatedSalary'] + 1e-5)
        X_out['Balance_per_Product'] = X_out['Balance'] / X_out['NumOfProducts']
        X_out['Is_Zero_Balance'] = (X_out['Balance'] == 0).astype(int)
        
        X_out['Age_Tenure_Ratio'] = X_out['Tenure'] / (X_out['Age'] + 1e-5)
        X_out['Is_High_Risk_Age'] = ((X_out['Age'] >= 40) & (X_out['Age'] <= 65)).astype(int)
        
        X_out['Is_Optimal_Products'] = (X_out['NumOfProducts'] == 2).astype(int)
        X_out['Is_Excess_Products'] = (X_out['NumOfProducts'] >= 3).astype(int)
        
        cat_cols = ['Geography', 'Gender', 'Card Type']
        X_encoded = pd.get_dummies(X_out, columns=cat_cols, drop_first=True)
        
        if self.target_columns is not None:
            for col in self.target_columns:
                if col not in X_encoded.columns:
                    X_encoded[col] = 0
            X_encoded = X_encoded[self.target_columns]
            
        return X_encoded

# 注册到当前运行环境的 __main__ 模块下
setattr(__main__, 'BankPipelineTransformer', BankPipelineTransformer)
sys.modules['__main__'].BankPipelineTransformer = BankPipelineTransformer

# ------------------------------------------------------------------------------
# 3. 页面配置与模型加载
# ------------------------------------------------------------------------------
st.set_page_config(page_title="银行客户流失预警系统", layout="wide")
st.title("🏦 银行客户流失预警与精准营销系统")

@st.cache_resource
def load_pipeline():
    return joblib.load('churn_pipeline_aligned.pkl')

payload = load_pipeline()
pipeline = payload['pipeline']
optimal_thresh = payload['optimal_threshold']

# ------------------------------------------------------------------------------
# 4. 侧边栏：客户特征输入栏
# ------------------------------------------------------------------------------
st.sidebar.header("📋 客户基础特征输入")
credit_score = st.sidebar.number_input("信用评分 (CreditScore)", 300, 850, 650)
geography = st.sidebar.selectbox("所在国家 (Geography)", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("性别 (Gender)", ["Male", "Female"])
age = st.sidebar.slider("年龄 (Age)", 18, 90, 45)
tenure = st.sidebar.slider("开户网龄/年 (Tenure)", 0, 10, 3)
balance = st.sidebar.number_input("账户余额/€ (Balance)", 0.0, 250000.0, 75000.0)
num_of_products = st.sidebar.selectbox("持有产品数量 (NumOfProducts)", [1, 2, 3, 4], index=1)
has_cr_card = st.sidebar.selectbox("是否持有信用卡", [1, 0], format_func=lambda x: "是" if x==1 else "否")
is_active_member = st.sidebar.selectbox("是否活跃会员", [1, 0], format_func=lambda x: "活跃" if x==1 else "不活跃/沉睡")
salary = st.sidebar.number_input("预估年薪/€ (EstimatedSalary)", 0.0, 200000.0, 80000.0)
card_type = st.sidebar.selectbox("卡片等级 (Card Type)", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"])
points = st.sidebar.number_input("客户积分 (Point Earned)", 0, 1000, 450)

# ------------------------------------------------------------------------------
# 5. 评估逻辑与结果展示
# ------------------------------------------------------------------------------
if st.button("🚀 评估该客户流失风险"):
    input_data = {
        'CreditScore': credit_score, 'Geography': geography, 'Gender': gender,
        'Age': age, 'Tenure': tenure, 'Balance': balance,
        'NumOfProducts': num_of_products, 'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member, 'EstimatedSalary': salary,
        'Card Type': card_type, 'Point Earned': points
    }
    input_df = pd.DataFrame([input_data])
    prob = pipeline.predict_proba(input_df)[0, 1]
    is_high_risk = prob >= optimal_thresh
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("预测流失概率", f"{prob:.1%}")
    with col2:
        st.metric("最佳决策阈值", f"{optimal_thresh:.4f}")
    with col3:
        if is_high_risk:
            st.error("🚨 判定结果：高危流失客户")
        else:
            st.success("✅ 判定结果：低风险/留存客户")
            
    st.markdown("### 💡 客户经理干预 SOP 建议")
    if is_high_risk:
        if num_of_products >= 3:
            st.warning("⚠️ **产品结构风险**：持有产品过多（>=3个），建议联系客户精简服务套餐。")
        if age >= 40 and age <= 65:
            st.warning("⚠️ **年龄区间风险**：处于中年高危期，建议主动推送稳健型中长期理财或养老规划。")
        if is_active_member == 0 and balance > 50000:
            st.warning("⚠️ **高资沉睡风险**：存款较高但账户不活跃，建议 3 个工作日内进行电话关怀。")
    else:
        st.info("该客户目前处于低风险区间，维持常规客户维系即可。")
