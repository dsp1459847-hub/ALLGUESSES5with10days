import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter

st.set_page_config(page_title="MAYA AI - Auto Backtest", layout="wide")

st.title("MAYA AI: 10-Day Auto-Backtest & Live Predictor")

# --- 1. Sidebar Controls ---
st.sidebar.header("📁 Data & Date Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel File", type=['csv', 'xlsx'])

# Date Selection Control
selected_end_date = st.sidebar.date_input("Calculation End Date", datetime(2026, 4, 16))

st.sidebar.markdown("---")
st.sidebar.header("🎯 Target Selection")
shift_names = ["DS", "FD", "GD", "GL", "DB", "SG", "ZA"]
target_shift_name = st.sidebar.selectbox("Predict for Which Shift?", shift_names)

max_repeat_limit = st.sidebar.slider("Max Repeat Limit", 2, 5, 4)

if uploaded_file is not None:
    try:
        # --- 2. Data Loading & Cleaning ---
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        
        # User ki selected date tak ka data filter karna
        filtered_df = df[df['DATE'].dt.date <= selected_end_date].copy()
        
        # 'XX' aur blanks hata kar clean karna
        filtered_df[target_shift_name] = pd.to_numeric(filtered_df[target_shift_name], errors='coerce')
        filtered_df = filtered_df.dropna(subset=['DATE', target_shift_name])
        filtered_df[target_shift_name] = filtered_df[target_shift_name].astype(int)
        
        # Sort values properly
        filtered_df = filtered_df.sort_values(by='DATE').reset_index(drop=True)
        
        if len(filtered_df) == 0:
            st.warning(f"Selected date ({selected_end_date}) tak koi valid data nahi mila.")
            st.stop()

        # --- 3. Core Engine Function ---
        def run_elimination(shift_list, limit):
            eliminated = set()
            scores = Counter()
            for days in range(1, 31):
                if len(shift_list) < days: continue
                sheet = shift_list[-days:]
                counts = Counter(sheet)
                
                # Zero-Repeat Block Elimination
                if len(counts) == len(sheet) and len(sheet) > 1:
                    eliminated.update(sheet)
                
                # Max Hit Elimination
                for num, freq in counts.items():
                    if freq >= limit: eliminated.add(num)
                    else: scores[num] += 1
            return eliminated, scores

        # --- 4. AUTO-BACKTEST (Last 10 Days Table) ---
        st.markdown("---")
        st.write(f"### 📅 10-Day Auto-Backtest for **{target_shift_name}**")
        st.write("*(Yahan aap dekh sakte hain ki pichle 10 dinon mein AI ki prediction kitni sahi thi)*")

        # Pichle 10 valid records nikalna
        last_10_records = filtered_df.tail(10)
        
        backtest_results = []
        tier_overall_hits = {"High Tier": 0, "Medium Tier": 0, "Low Tier": 0, "Failed (Eliminated)": 0}
        
        with st.spinner("Calculating 10-day historical predictions..."):
            for index, row in last_10_records.iterrows():
                test_date = row['DATE']
                actual_num = row[target_shift_name]
                
                # Is din se pehle ka data lena (Prediction hamesha past data se hoti hai)
                past_df = filtered_df[filtered_df['DATE'] < test_date]
                past_list = past_df[target_shift_name].tolist()
                
                if len(past_list) > 0:
                    elim_p, scores_p = run_elimination(past_list, max_repeat_limit)
                    safe_p = sorted([n for n in range(100) if n not in elim_p], key=lambda x: scores_p[x], reverse=True)
                    
                    status = "Failed (Eliminated)"
                    if safe_p:
                        n_p = len(safe_p)
                        high_t = safe_p[:int(n_p*0.33)]
                        med_t = safe_p[int(n_p*0.33):int(n_p*0.66)]
                        low_t = safe_p[int(n_p*0.66):]
                        
                        if actual_num in high_t: status = "High Tier"
                        elif actual_num in med_t: status = "Medium Tier"
                        elif actual_num in low_t: status = "Low Tier"
                    
                    tier_overall_hits[status] += 1
                    
                    backtest_results.append({
                        "Date": test_date.strftime('%d %B %Y'),
                        "Actual Number": f"{actual_num:02d}",
                        "Prediction Result": status
                    })

        if backtest_results:
            result_df = pd.DataFrame(backtest_results)
            # Table coloring for better visibility
            def highlight_status(val):
                if val == 'High Tier': return 'background-color: #004d00; color: white'
                elif val == 'Medium Tier': return 'background-color: #b38f00; color: white'
                elif val == 'Low Tier': return 'background-color: #003366; color: white'
                else: return 'background-color: #4d0000; color: white'

            st.table(result_df.style.map(highlight_status, subset=['Prediction Result']))
            
        # --- 5. LIVE PREDICTION FOR NEXT DAY ---
        st.markdown("---")
        target_prediction_date = filtered_df['DATE'].iloc[-1] + timedelta(days=1)
        st.header(f"🎯 Live Prediction for: {target_prediction_date.strftime('%d %B %Y')}")
        
        target_data_list = filtered_df[target_shift_name].tolist()
        elim_final, scores_final = run_elimination(target_data_list, max_repeat_limit)
        safe_pool = sorted([n for n in range(100) if n not in elim_final], key=lambda x: scores_final[x], reverse=True)
        
        # Recommendation logic based on backtest
        valid_hits = {k: v for k, v in tier_overall_hits.items() if k != "Failed (Eliminated)"}
        best_tier = max(valid_hits, key=valid_hits.get) if sum(valid_hits.values()) > 0 else "None"

        if best_tier != "None":
            st.success(f"**AI Recommendation: Agli shift ke liye aapko [{best_tier.upper()}] par focus karna chahiye.**")
            st.write(f"*(Kyonki pichle 10 din mein sabse zyada results isi category se paas hue hain)*")

        if safe_pool:
            n_s = len(safe_pool)
            high_tier = safe_pool[:int(n_s*0.33)]
            med_tier = safe_pool[int(n_s*0.33):int(n_s*0.66)]
            low_tier = safe_pool[int(n_s*0.66):]
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("#### 🔥 High Tier")
                st.write(", ".join([f"{x:02d}" for x in high_tier]))
            with c2:
                st.markdown("#### ⚡ Medium Tier")
                st.write(", ".join([f"{x:02d}" for x in med_tier]))
            with c3:
                st.markdown("#### ❄️ Low Tier")
                st.write(", ".join([f"{x:02d}" for x in low_tier]))
                
        st.info(f"Total Eliminated Numbers: {len(elim_final)} | Safe (Playable) Numbers: {len(safe_pool)}")

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("👈 Please upload your Excel/CSV file to begin.")
              
