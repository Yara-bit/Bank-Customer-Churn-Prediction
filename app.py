# app.py - 银行客户流失预警工作台
import streamlit as st
import pandas as pd
import joblib

# 页面基础配置
st.set_page_config(page_title="银行客户流失预警系统", layout="wide")
st.title("🏦 银行客户流失预警与精准营销系统")
st.markdown("基于 GBDT 算法与端到端 Pipeline 的实时流失风险评估")

# 1. 加载打包好的 Pipeline 组合包
@st.cache_resource
def load_pipeline():
    return joblib.load('churn_pipeline_aligned.pkl')

payload = load_pipeline()
pipeline = payload['pipeline']
optimal_thresh = payload['optimal_threshold']

# 2. 侧边栏：客户特征输入栏
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

# 3. 实时推理计算
if st.button("🚀 评估该客户流失风险"):
    input_data = {
        'CreditScore': credit_score, 'Geography': geography, 'Gender': gender,
        'Age': age, 'Tenure': tenure, 'Balance': balance,
        'NumOfProducts': num_of_products, 'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member, 'EstimatedSalary': salary,
        'Card Type': card_type, 'Point Earned': points
    }
    input_df = pd.DataFrame([input_data])
    
    # 预测概率
    prob = pipeline.predict_proba(input_df)[0, 1]
    is_high_risk = prob >= optimal_thresh
    
    # 展示预测结果
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
            
    # 业务建议
    st.markdown("### 💡 客户经理干预 SOP 建议")
    if is_high_risk:
        if num_of_products >= 3:
            st.warning("⚠️ **产品结构风险**：该客户持有产品过多（>=3个），触发交叉营销惩罚。建议协助其进行产品瘦身，梳理核心服务。")
        if age >= 40 and age <= 65:
            st.warning("⚠️ **年龄区间风险**：处于 40-65 岁中年高危流失期。建议主动推送稳健型中长期理财或养老投资计划。")
        if is_active_member == 0 and balance > 50000:
            st.warning("⚠️ **高资沉睡风险**：存款较高但长期不活跃。建议客户经理在 3 个工作日内电话拜访，重新激活关系。")
    else:
        st.info("该客户属于低风险群体，维持常规服务即可，无需额外投入挽留资源。")
