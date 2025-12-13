import streamlit as st
from statsbombpy import sb
import pandas as pd
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# --- App Configuration ---
st.set_page_config(page_title="MLS 2025 Dashboard", layout="wide")
st.title("🇺🇸 MLS 2025 Tactical Dashboard")

# --- 🔒 AUTHENTICATION SETUP ---
# Securely fetch keys from Streamlit Secrets
try:
    AUTH_CREDS = st.secrets["statsbomb"]
except FileNotFoundError:
    st.error("Secrets not found. Please set up your secrets.toml or Streamlit Cloud Secrets.")
    st.stop()
    
# --- Helper Function for Downloads ---
def convert_fig_to_png(fig):
    """Converts a matplotlib figure to a PNG image in memory."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=300, facecolor='#22312b')
    buf.seek(0)
    return buf

# --- Fetch Match Schedule ---
@st.cache_data
def get_mls_matches():
    # 1. Fetch competitions
    comps = sb.competitions(creds=AUTH_CREDS)
    
    # 2. Filter for MLS 2025
    # Note: Ensure the string matches exactly what StatsBomb returns for the season name
    mls_2025 = comps[
        (comps['competition_name'] == 'Major League Soccer') & 
        (comps['season_name'] == '2025')
    ]
    
    if mls_2025.empty:
        st.error("Could not find MLS 2025 data. Check API permissions or Season Name.")
        st.stop()
        
    comp_id = mls_2025.iloc[0]['competition_id']
    season_id = mls_2025.iloc[0]['season_id']
    
    # 3. Fetch Matches
    df = sb.matches(competition_id=comp_id, season_id=season_id, creds=AUTH_CREDS)
    df = df.sort_values('match_date')
    df['match_label'] = df['home_team'] + " vs " + df['away_team'] + " (" + df['match_date'] + ")"
    return df

try:
    matches = get_mls_matches()
except Exception as e:
    st.error(f"Authentication Error: Please check your API keys. Details: {e}")
    st.stop()

# --- Sidebar: Match Selection ---
st.sidebar.header("Filter Options")

# 1. Select Team
unique_teams = sorted(list(set(matches['home_team'].unique()) | set(matches['away_team'].unique())))
selected_team = st.sidebar.selectbox("Select Team", unique_teams)

# 2. Select Match
team_matches = matches[(matches['home_team'] == selected_team) | (matches['away_team'] == selected_team)]
match_label = st.sidebar.selectbox("Select Match", team_matches['match_label'].tolist())

match_data = team_matches[team_matches['match_label'] == match_label].iloc[0]
match_id = match_data['match_id']
opponent_team = match_data['away_team'] if selected_team == match_data['home_team'] else match_data['home_team']

# --- Fetch Events (SAFE MODE) ---
@st.cache_data
def get_events(m_id):
    try:
        # Attempt to fetch events
        ev = sb.events(match_id=m_id, creds=AUTH_CREDS)
        return ev
    except ValueError:
        # Catches "No objects to concatenate" (Empty Data)
        return None
    except Exception:
        # Catches other API issues
        return None

events = get_events(match_id)

# --- 🛑 CRITICAL SAFETY CHECK ---
if events is None or events.empty:
    st.warning(f"⚠️ Event data is not available for **{match_label}** yet.")
    st.info("This matches appears in the schedule, but StatsBomb has not uploaded the detailed event data yet. This is common for future matches or games played in the last 24 hours.")
    st.stop()  # Stop the script here so it doesn't crash below

# --- 4. PLAYER SELECTION ---
# We use the full match data to ensure the player list persists across period changes
team_events_all = events[events['team'] == selected_team]
players = sorted(team_events_all['player'].dropna().unique().tolist())
players.insert(0, "All Players")

selected_player = st.sidebar.selectbox("Filter by Player", players)

# --- 5. PERIOD SELECTION ---
period_mapping = {'All': [1, 2, 3, 4, 5], '1st Half': [1], '2nd Half': [2], 'Extra Time': [3, 4], 'Penalties': [5]}
selected_period = st.sidebar.radio("Select Period", list(period_mapping.keys()))
period_filter = period_mapping[selected_period]

# --- 6. APPLY FILTERS ---
viz_events = team_events_all[team_events_all['period'].isin(period_filter)]

if selected_player != "All Players":
    viz_events = viz_events[viz_events['player'] == selected_player]

# --- SCOREBOARD METRICS ---
period_events_both_teams = events[events['period'].isin(period_filter)]
team_goals = len(period_events_both_teams[(period_events_both_teams['team'] == selected_team) & (period_events_both_teams['shot_outcome'] == 'Goal')])
opp_goals = len(period_events_both_teams[(period_events_both_teams['team'] == opponent_team) & (period_events_both_teams['shot_outcome'] == 'Goal')])

# Safe xG calculation (checks if column exists)
if 'shot_statsbomb_xg' in period_events_both_teams.columns:
    team_xg = period_events_both_teams[(period_events_both_teams['team'] == selected_team) & (period_events_both_teams['type'] == 'Shot')]['shot_statsbomb_xg'].sum()
    opp_xg = period_events_both_teams[(period_events_both_teams['team'] == opponent_team) & (period_events_both_teams['type'] == 'Shot')]['shot_statsbomb_xg'].sum()
else:
    team_xg = 0.0
    opp_xg = 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Team", selected_team)
col2.metric(f"Score ({selected_period})", f"{team_goals} - {opp_goals}")
col3.metric(f"{selected_team} xG", f"{team_xg:.2f}")
col4.metric(f"{opponent_team} xG", f"{opp_xg:.2f}")

st.divider()

# --- TABS INTERFACE ---
tab1, tab2, tab3 = st.tabs(["🎯 Shot Map", "yPass Map", "🔥 Heatmap"])

# === TAB 1: SHOT MAP ===
with tab1:
    shots = viz_events[viz_events['type'] == 'Shot'].copy()
    
    if not shots.empty:
        shots = shots.sort_values('minute').reset_index(drop=True)
        
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
        fig, ax = pitch.draw(figsize=(10, 7))
        
        shots['x'] = shots['location'].apply(lambda x: x[0])
        shots['y'] = shots['location'].apply(lambda x: x[1])
        
        # Check Selection
        selected_row_index = None
        if "shot_table" in st.session_state:
            selection = st.session_state["shot_table"].get("selection", {})
            if selection.get("rows"):
                selected_row_index = selection["rows"][0]

        for i, shot in shots.iterrows():
            if i == selected_row_index:
                color = 'yellow'
                alpha = 1.0
                zorder = 3
            else:
                color = 'green' if shot['shot_outcome'] == 'Goal' else 'red'
                alpha = 0.6
                zorder = 2

            marker = 'football' if shot['shot_outcome'] == 'Goal' else 'o'
            size = 200 if shot['shot_outcome'] == 'Goal' else 100
            
            pitch.scatter(shot['x'], shot['y'], ax=ax, c=color, marker=marker, 
                          s=size, alpha=alpha, zorder=zorder, edgecolors='white')

        ax.scatter([], [], c='green', marker='o', s=100, label='Goal')
        ax.scatter([], [], c='red', marker='o', s=100, alpha=0.6, label='No Goal')
        ax.scatter([], [], c='yellow', marker='o', s=100, edgecolors='white', label='Selected')
        ax.legend(loc='upper left', labelspacing=1, fontsize=10)

        st.pyplot(fig)
        
        # Download Button
        clean_player_name = selected_player.replace(" ", "")
        fn = f"ShotMap_{selected_team}_{clean_player_name}_{selected_period}.png"
        st.download_button(
            label="⬇️ Download Shot Map as PNG",
            data=convert_fig_to_png(fig),
            file_name=fn,
            mime="image/png"
        )
        
        if 'shot_statsbomb_xg' in shots.columns:
            st.caption(f"Showing {len(shots)} shots for {selected_player} during {selected_period} | Total xG: {shots['shot_statsbomb_xg'].sum():.2f}")
        
        with st.expander("View Detailed Shot Data (Click row to highlight)", expanded=False):
            available_cols = ['minute', 'player', 'shot_outcome', 'shot_statsbomb_xg', 'shot_type']
            display_cols = [c for c in available_cols if c in shots.columns]
            st.dataframe(shots[display_cols], use_container_width=True, on_select="rerun", selection_mode="single-row", key="shot_table")
            
    else:
        st.warning(f"No shots found for {selected_player} in {selected_period}.")

# === TAB 2: PASS MAP ===
with tab2:
    st.header(f"Passing Network ({selected_period})")
    passes = viz_events[viz_events['type'] == 'Pass'].copy()
    
    if not passes.empty:
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
        fig, ax = pitch.draw(figsize=(10, 7))
        
        passes['x'] = passes['location'].apply(lambda x: x[0])
        passes['y'] = passes['location'].apply(lambda x: x[1])
        passes['end_x'] = passes['pass_end_location'].apply(lambda x: x[0])
        passes['end_y'] = passes['pass_end_location'].apply(lambda x: x[1])
        
        completed = passes[passes['pass_outcome'].isna()]
        
        pitch.arrows(completed.x, completed.y, completed.end_x, completed.end_y, 
                     ax=ax, width=2, headwidth=3, color='#adff2f', alpha=0.5, label='Completed')
        
        st.pyplot(fig)
        
        # Download Button
        clean_player_name = selected_player.replace(" ", "")
        fn = f"PassMap_{selected_team}_{clean_player_name}_{selected_period}.png"
        st.download_button(
            label="⬇️ Download Pass Map as PNG",
            data=convert_fig_to_png(fig),
            file_name=fn,
            mime="image/png"
        )
        
        st.write(f"**Passes displayed:** {len(passes)}")
    else:
        st.warning(f"No passes found for {selected_player} in {selected_period}.")

# === TAB 3: HEATMAP ===
with tab3:
    st.header(f"Heatmap ({selected_period})")
    heatmap_data = viz_events.dropna(subset=['location']).copy()
    
    if not heatmap_data.empty:
        heatmap_data['x'] = heatmap_data['location'].apply(lambda x: x[0])
        heatmap_data['y'] = heatmap_data['location'].apply(lambda x: x[1])
        
        pitch = Pitch(pitch_type='statsbomb', line_zorder=2, pitch_color='#22312b', line_color='#c7d5cc')
        fig, ax = pitch.draw(figsize=(10, 7))
        
        bin_statistic = pitch.bin_statistic(heatmap_data.x, heatmap_data.y, statistic='count', bins=(25, 25))
        pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
        
        st.pyplot(fig)
        
        # Download Button
        clean_player_name = selected_player.replace(" ", "")
        fn = f"HeatMap_{selected_team}_{clean_player_name}_{selected_period}.png"
        st.download_button(
            label="⬇️ Download Heatmap as PNG",
            data=convert_fig_to_png(fig),
            file_name=fn,
            mime="image/png"
        )

    else:
        st.warning("Not enough data for heatmap.")