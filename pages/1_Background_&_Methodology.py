# pages/1_Background_&_Methodology.py
import streamlit as st

st.set_page_config(page_title="Methodology", page_icon="⚙️", layout="wide")

st.title("Under the Hood: The Quantitative Methodology")
st.markdown("---")

st.write("""
The Advanced Retirement Simulator is not a traditional "straight-line" calculator. Traditional calculators assume your portfolio grows by a flat 7% every year and inflation is a flat 3%. In the real world, average returns don't matter as much as the sequence of those returns.

To evaluate your retirement survivability, I built an **Institution-Grade Stochastic Engine** that tests your financial profile against 10,000 parallel realities. Here is exactly how the mathematical models work:
""")

st.header("1. Stochastic Market & Inflation Modeling")
st.write("Instead of linear math, the engine uses advanced statistical modeling to simulate 10,000 potential future timelines.")
st.markdown("""
- **Fat-Tailed Market Shocks (Student's t-Distribution):** Stock market returns are not perfectly a "bell curve." The real market experiences extreme crashes (like 2008 or 2020) more often than standard math predicts. Our engine uses a Student's t-distribution (Degrees of Freedom = 5) to inject realistic "fat-tail" black swan events into your simulations.
- **Correlated Asset Classes (Cholesky Decomposition):** If the stock market crashes, bonds and cash usually behave differently. The engine applies a Cholesky Matrix to maintain historically accurate mathematical correlations between your TSP, IRAs, Taxable accounts, and Cash.
- **Mean-Reverting Inflation with Stagflation Jumps:** Inflation isn't static. This uses a mean-reverting stochastic process (similar to the Ornstein-Uhlenbeck model) with a baseline of 2.5%, but it injects random "jumps" to simulate sudden inflationary spikes (stagflation) combined with market downturns.
""")

st.header("2. The Withdrawal Optimization Algorithm (Brent's Method)")
st.write("**How does it find your perfect 'Optimized Initial Withdrawal Rate'?**")
st.markdown("""
- I deployed a 1-Dimensional Root-Finding Algorithm (Brent’s Method).
- The engine runs your 10,000 lifetimes at a random withdrawal rate, calculates the median ending wealth at your life expectancy, and compares it to your declared "Target Legacy Floor."
- It then iteratively adjusts the withdrawal rate up and down, re-running the 10,000 simulations over and over until the math perfectly converges on the exact percentage that safely lands you at your target floor without running out of money.
""")

st.header("3. Adaptive Guardrails & The Retirement Smile")
st.write("Real retirees don't spend the exact same amount of money every year, adjusting blindly for inflation. They adjust based on the market and aging.")
st.markdown("""
- **Dynamic Spending Guardrails:** If the market booms and your withdrawal rate falls 20% below your starting rate, the engine grants you a 10% pay raise (Prosperity Rule). If the market crashes and your withdrawal rate spikes 20% too high, the engine forces a 10% pay cut (Capital Preservation Rule) to protect your principal.
- **The "Retirement Smile" (David Blanchett Curve):** Your discretionary spending is modeled geometrically. Spending drops slowly during your "Slow-Go" years (ages 75-84) as travel and hobbies decline but re-accelerates in your "No-Go" years (85+) to account for increased medical conveniences and end-of-life care.
""")

st.header("4. Dynamic Liquidation Hierarchy & Sequence Risk Mitigation")
st.write("The engine actively manages where you pull money from year by year based on what the simulated market is doing.")
st.markdown("""
- **Normal Years:** Lifestyle is funded by Tax-Deferred accounts (TSP/Trad IRA), allowing your Tax-Free (Roth) and Taxable accounts to compound.
- **Market Crash Years (Down >10%):** The engine triggers an emergency **Sequence of Return Risk (SORR)** protocol. It immediately halts the sale of equities in your TSP/IRA to avoid locking in losses. It seamlessly pivots to burning down your Cash Buffer, followed by Taxable and Roth accounts, until the market recovers.
""")

st.header("5. Federal Tax Code & Roth Optimization Engine")
st.write("The model contains a highly detailed US Tax logic tree.")
st.markdown("""
- It tracks Standard Deductions, Ordinary Brackets, Long-Term Capital Gains (LTCG), Net Investment Income Tax (NIIT), and State/Local taxes.
- **Roth Optimizer:** In the background, the engine actually runs your entire lifetime 5 separate times using different Roth Conversion strategies (Baseline, Filling your current bracket, Targeting IRMAA cliffs, and Max Bracket limits). It compares the "Terminal Legacy" of all 5 runs and surfaces the mathematical winner to you, alongside a step-by-step conversion schedule.
""")
