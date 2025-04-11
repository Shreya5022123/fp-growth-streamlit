# Install these once before running:
# pip install streamlit pandas mlxtend

import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title("📈 Association Rule Mining App (FP-Growth)")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("Upload CSV File (Transaction List)", type=["csv"])

if uploaded_file:
    # Read file as plain text and parse transactions
    content = uploaded_file.read().decode("utf-8")
    lines = content.strip().split('\n')
    transactions = [line.strip().split(',') for line in lines]

    st.subheader("✅ Sample Transactions:")
    st.write(transactions[:5])

    # Step 2: Encode transactions using TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    # Step 3: Get min support and confidence from user
    min_support = st.slider("Minimum Support", 0.01, 1.0, 0.05)
    min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5)

    # Step 4: Apply FP-Growth algorithm
    frequent_itemsets = fpgrowth(df_trans, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    st.subheader("🔗 Association Rules")
    if not rules.empty:
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

        # Step 5: Optional - Visualize top 10 rules by lift
        st.subheader("📊 Top 10 Rules by Lift")
        top_rules = rules.nlargest(10, 'lift')[['antecedents', 'consequents', 'lift']]
        st.dataframe(top_rules)
        st.bar_chart(top_rules.set_index('antecedents')['lift'])
    else:
        st.warning("No rules found with the selected support/confidence thresholds.")
