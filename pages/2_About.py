# pages/2_About.py
import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title("About the Advanced Quantitative Retirement Planner")
st.markdown("---")

st.header("The Mission")
st.write("""
For decades, ultra-wealthy families and institutional endowments have relied on sophisticated Monte Carlo simulations and dynamic spending algorithms to manage their wealth. Meanwhile, DIY investors have been forced to rely on rudimentary calculators that output dangerous, straight-line "averages."

**I built this platform to democratize institution-grade financial modeling.**

The Advanced Quantitative Retirement Planner was designed to bridge the gap between basic retirement calculators and expensive, gatekept professional financial software. It evaluates the raw, mathematical truth of your retirement survivability.
""")

st.header("Specialized for Federal Employees")
st.write("""
While this simulator is highly effective for any private-sector retiree, it features a specialized logic engine built specifically to handle the unique nuances of United States Federal Employees and Military Retirees.
""")
st.markdown("""
- Native integration for the Thrift Savings Plan (TSP).
- Actuarial comparisons between Medicare Part B / IRMAA and FEHB (FEPBlue, GEHA) / TRICARE for Life.
- Phased Retirement modeling and FERS Pension COLA calculations.
""")

st.header("Why 'CASAM'?")
st.write("""
This tool relies on the **Constant Amortization Spending Model (CASAM)**. Instead of using rigid rules like the "4% Rule," CASAM looks at your actual portfolio balance, guaranteed income streams (Social Security, Pensions), and specific tax liabilities every single year, dynamically adjusting your safe spending limits to ensure your money outlives you.
""")
