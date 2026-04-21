# Ensure all required tabs + new advanced actuarial tabs are available
    t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11 = st.tabs([
        "📊 Projections", "💵 Cash Flow", "📉 Guardrails", "📈 Net Worth", "🏛️ Taxes", 
        "🏛️ Legacy/Estate", "💡 Coach Alerts", "🔄 Roth Optimizer", "🦅 Social Sec", "🏥 Medicare", "💾 Exports"
    ])

    with t1:
        st.subheader("Lifetime Projections & Monte Carlo Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Probability of Success", f"{prob_success:.1f}%")
        col2.metric("Median Terminal Wealth", f"${median_paths[-1]:,.0f}")
        col3.metric("10th Percentile Wealth", f"${np.percentile(history['total_bal'], 10, axis=0)[-1]:,.0f}")
        st.plotly_chart(plot_wealth_trajectory(history, inputs['target_floor'], years_arr), use_container_width=True)

    with t2:
        st.subheader("Cash Flow Forecast & Real Net Spendable Income")
        st.plotly_chart(plot_net_spendable(history, years_arr), use_container_width=True)

    with t3: # NEW TAB
        st.subheader("Income Volatility & Dynamic Guardrails")
        st.write("This chart illustrates the impact of Guyton-Klinger rules. In severe market downturns, your scheduled spending may be reduced by up to 10% to protect portfolio longevity. In strong markets, spending receives a prosperity bump.")
        # Import and call new function
        from visuals import plot_income_volatility
        st.plotly_chart(plot_income_volatility(history, years_arr), use_container_width=True)

    with t4:
        st.subheader("Net Worth Forecast & Asset Liquidity")
        st.plotly_chart(plot_liquidity_timeline(history, years_arr), use_container_width=True)

    with t5:
        st.subheader("Taxes, Withdrawals, and RMDs")
        st.plotly_chart(plot_taxes_and_rmds(history, years_arr), use_container_width=True)

    with t6: # NEW TAB
        st.subheader("After-Tax Legacy & Estate Breakdown")
        st.write("Not all terminal wealth is created equal. Inherited TSP/401(k) assets are heavily taxed, whereas Roth and Taxable assets pass highly efficiently.")
        # Import and call new function
        from visuals import plot_legacy_breakdown
        st.plotly_chart(plot_legacy_breakdown(history), use_container_width=True)
        
        # Calculate rough after-tax estate value
        med_tsp = np.median(history['tsp_bal'][:, -1])
        med_roth = np.median(history['roth_bal'][:, -1])
        med_taxable = np.median(history['taxable_bal'][:, -1]) + np.median(history['cash_bal'][:, -1])
        
        # Heirs pay ordinary income on TSP (assume 24% blended tax), step-up basis on taxable, 0% on roth
        net_to_heirs = (med_tsp * 0.76) + med_taxable + med_roth
        st.metric("Estimated Net After-Tax Value to Heirs", f"${net_to_heirs:,.0f}", delta=f"Lost to IRD Taxes: -${med_tsp * 0.24:,.0f}", delta_color="inverse")

    with t7:
        st.subheader("PlannerPlus Coach Alerts & Actionable To-Do List")
        # (Keep your existing coach alert code here)
        med_taxes = np.median(history['taxes_fed'], axis=0)
        if med_taxes[-1] > med_taxes[0] * 2.5:
            st.warning("⚠️ **RMD Tax Spike Alert**: Your projected tax liability more than doubles after age 75. Consider Roth Conversions.")
        if prob_success >= 85:
            st.success("✅ **Plan is on Track**: Highly secure probability of meeting your terminal floor.")

    with t8:
        st.subheader("Roth Conversion Optimizer")
        # (Keep existing Roth code)
        st.markdown(f"**Baseline Median Terminal Wealth:** ${median_paths[-1]:,.0f}")

    with t9:
        st.subheader("Social Security Claiming Strategy")
        # (Keep existing SS code)
        st.plotly_chart(plot_social_security_analysis(prob_success), use_container_width=True)

    with t10:
        st.subheader("Medicare Part B & IRMAA vs. Retiree Coverage")
        # (Keep existing Medicare code)
        total_medicare_cost = np.sum(np.median(history['medicare_cost'], axis=0))
        st.metric("Total Projected Lifetime Medicare Part B + IRMAA Costs", f"${total_medicare_cost:,.0f}")

    with t11:
        st.subheader("Strict-Format CSV Data Exports")
        # (Keep existing CSV code)