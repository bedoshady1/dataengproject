"""
NYC Motor Vehicle Crashes Dashboard - ADVANCED VERSION (Pro UI)
================================================================
Professional, creative UI with dark mode, KPI cards, and live refresh.

Author: Advanced UI v3.0
Date: November 2025
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
import logging
from functools import lru_cache

# ============================================================================
# CONFIGURATION & LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_PATH = "cleaned_integrated_nyc_crashes.zip"

COLORS_LIGHT = {
    'bg': '#f5f7fb',
    'surface': '#ffffff',
    'surface_alt': '#f8fafc',
    'border': '#e2e8f0',
    'primary': '#2563eb',
    'primary_soft': 'rgba(37, 99, 235, 0.12)',
    'accent': '#ec4899',
    'accent_soft': 'rgba(236, 72, 153, 0.12)',
    'success': '#16a34a',
    'warning': '#f97316',
    'danger': '#ef4444',
    'muted': '#6b7280',
    'text': '#0f172a',
    'text_soft': '#64748b',
    'gradient_hero': 'linear-gradient(135deg, #2563eb 0%, #4f46e5 40%, #ec4899 100%)',
}

COLORS_DARK = {
    'bg': '#020617',
    'surface': '#020617',
    'surface_alt': '#020617',
    'border': '#1e293b',
    'primary': '#60a5fa',
    'primary_soft': 'rgba(96, 165, 250, 0.25)',
    'accent': '#fb7185',
    'accent_soft': 'rgba(251, 113, 133, 0.25)',
    'success': '#22c55e',
    'warning': '#fb923c',
    'danger': '#f97373',
    'muted': '#9ca3af',
    'text': '#e5e7eb',
    'text_soft': '#9ca3af',
    'gradient_hero': 'linear-gradient(135deg, #1d4ed8 0%, #4f46e5 40%, #e11d48 100%)',
}

# ============================================================================
# DATA LOADING WITH CACHING
# ============================================================================

@lru_cache(maxsize=1)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from local file: {path}")
        # IMPORTANT: tell pandas it's a ZIP
        df = pd.read_csv(path, compression="zip", low_memory=False)

        df.columns = [c.strip() for c in df.columns]
        logger.info(f"Raw columns from source: {list(df.columns)}")
        logger.info(f"Loaded {len(df):,} records successfully")

        possible_date_cols = ["CRASH DATE", "Crash Date", "CRASH_DATE", "crash_date"]
        date_col = next((c for c in possible_date_cols if c in df.columns), None)
        if not date_col:
            raise ValueError(
                f"No crash date column found. Available columns: {list(df.columns)}"
            )
        if date_col != "CRASH DATE":
            df.rename(columns={date_col: "CRASH DATE"}, inplace=True)
        df["CRASH DATE"] = pd.to_datetime(df["CRASH DATE"], errors="coerce")

        if "CRASH_YEAR" not in df.columns:
            df["CRASH_YEAR"] = df["CRASH DATE"].dt.year
        if "CRASH_MONTH" not in df.columns:
            df["CRASH_MONTH"] = df["CRASH DATE"].dt.month
        if "CRASH_DOW" not in df.columns:
            df["CRASH_DOW"] = df["CRASH DATE"].dt.day_name()
        if "CRASH_HOUR" not in df.columns:
            df["CRASH_HOUR"] = df["CRASH DATE"].dt.hour
        if "CRASH_MONTH_NAME" not in df.columns:
            df["CRASH_MONTH_NAME"] = df["CRASH DATE"].dt.month_name()

        df["TIME_OF_DAY"] = pd.cut(
            df["CRASH_HOUR"],
            bins=[0, 6, 12, 18, 24],
            labels=[
                "Night (12AM-6AM)",
                "Morning (6AM-12PM)",
                "Afternoon (12PM-6PM)",
                "Evening (6PM-12AM)",
            ],
            include_lowest=True,
        )

        if {"CRASH DATE", "LATITUDE", "LONGITUDE"}.issubset(df.columns):
            df = df.drop_duplicates(
                subset=["CRASH DATE", "LATITUDE", "LONGITUDE"],
                keep="first",
            )

        if "BOROUGH" in df.columns:
            df["BOROUGH"] = df["BOROUGH"].fillna("Unknown")

        for col in ["NUMBER OF PERSONS KILLED", "NUMBER OF PERSONS INJURED"]:
            if col not in df.columns:
                df[col] = 0
        df["SEVERITY_SCORE"] = (
            df["NUMBER OF PERSONS KILLED"] * 10
            + df["NUMBER OF PERSONS INJURED"]
        )

        logger.info("Data preprocessing complete")
        return df

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

df = load_data()
# ============================================================================
# DROPDOWN OPTIONS & HELPERS
# ============================================================================

def sorted_unique(col):
    if col not in df.columns:
        return ['All']
    vals = df[col].dropna().unique().tolist()
    vals = sorted([v for v in vals if str(v).strip()])
    return ['All'] + vals

BOROUGHS = sorted_unique('BOROUGH')
YEARS = ['All'] + sorted(df['CRASH_YEAR'].astype(str).unique().tolist())
MONTHS = ['All'] + sorted(
    df['CRASH_MONTH_NAME'].dropna().unique().tolist(),
    key=lambda x: [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ].index(x)
)
VEHICLES = sorted_unique('VEHICLE TYPE CODE 1')
FACTORS = sorted_unique('CONTRIBUTING FACTOR VEHICLE 1')
TIME_OF_DAY = [
    'All',
    'Night (12AM-6AM)',
    'Morning (6AM-12PM)',
    'Afternoon (12PM-6PM)',
    'Evening (6PM-12AM)'
]
INJURIES = [
    'All',
    'Persons Injured',
    'Persons Killed',
    'Pedestrians Injured',
    'Pedestrians Killed',
    'Cyclists Injured',
    'Cyclists Killed',
    'Motorists Injured',
    'Motorists Killed'
]

# ============================================================================
# FILTERING LOGIC
# ============================================================================

def parse_search(text):
    filters = {
        'BOROUGH': None,
        'CRASH_YEAR': None,
        'CRASH_MONTH_NAME': None,
        'VEHICLE TYPE CODE 1': None,
        'CONTRIBUTING FACTOR VEHICLE 1': None,
        'INJURY_TYPE': None,
        'TIME_OF_DAY': None
    }
    if not text or not text.strip():
        return filters

    text_lower = text.lower()

    for borough in BOROUGHS:
        if borough != 'All' and borough.lower() in text_lower:
            filters['BOROUGH'] = borough
            break

    year_match = re.search(r'\b(20\d{2})\b', text)
    if year_match:
        filters['CRASH_YEAR'] = year_match.group(1)

    for month in MONTHS:
        if month != 'All' and month.lower() in text_lower:
            filters['CRASH_MONTH_NAME'] = month
            break

    for vehicle in VEHICLES:
        if vehicle != 'All' and len(vehicle) > 3 and vehicle.lower() in text_lower:
            filters['VEHICLE TYPE CODE 1'] = vehicle
            break

    for factor in FACTORS:
        if factor != 'All' and len(factor) > 3 and factor.lower() in text_lower:
            filters['CONTRIBUTING FACTOR VEHICLE 1'] = factor
            break

    for injury in INJURIES:
        if injury != 'All' and injury.lower() in text_lower:
            filters['INJURY_TYPE'] = injury
            break

    for time_label in TIME_OF_DAY:
        if time_label != 'All' and time_label.lower() in text_lower:
            filters['TIME_OF_DAY'] = time_label
            break

    return filters


def apply_filters(df_in, filters):
    dff = df_in.copy()
    for key, value in filters.items():
        if value and value != 'All':
            if key == 'CRASH_YEAR':
                dff = dff[dff['CRASH_YEAR'] == int(value)]
            elif key == 'INJURY_TYPE':
                continue
            else:
                dff = dff[dff[key] == value]
    return dff


def filter_by_injury_type(dff, injury_type):
    injury_filters = {
        'Persons Injured': dff['NUMBER OF PERSONS INJURED'] > 0,
        'Persons Killed': dff['NUMBER OF PERSONS KILLED'] > 0,
        'Pedestrians Injured': dff['NUMBER OF PEDESTRIANS INJURED'] > 0,
        'Pedestrians Killed': dff['NUMBER OF PEDESTRIANS KILLED'] > 0,
        'Cyclists Injured': dff['NUMBER OF CYCLIST INJURED'] > 0,
        'Cyclists Killed': dff['NUMBER OF CYCLIST KILLED'] > 0,
        'Motorists Injured': dff['NUMBER OF MOTORIST INJURED'] > 0,
        'Motorists Killed': dff['NUMBER OF MOTORIST KILLED'] > 0,
    }
    if injury_type in injury_filters:
        return dff[injury_filters[injury_type]]
    return dff

# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def create_monthly_trend_chart(dff, c):
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    if dff.empty:
        return go.Figure().update_layout(
            annotations=[dict(text="No data for selected filters", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=c['muted'], size=14))]
        )

    monthly = dff.groupby('CRASH_MONTH_NAME').size().reset_index(name='Crashes')
    monthly['CRASH_MONTH_NAME'] = pd.Categorical(
        monthly['CRASH_MONTH_NAME'],
        categories=month_order,
        ordered=True
    )
    monthly = monthly.sort_values('CRASH_MONTH_NAME')

    fig = px.bar(
        monthly,
        x='CRASH_MONTH_NAME',
        y='Crashes',
        title="Seasonal Crash Pattern",
        labels={'CRASH_MONTH_NAME': 'Month', 'Crashes': 'Number of Crashes'},
        color='Crashes',
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui", size=12, color=c['text']),
        title_font=dict(size=15, color=c['text']),
        title_x=0,
        margin=dict(l=20, r=10, t=40, b=40),
        coloraxis_showscale=False
    )
    fig.update_xaxes(showgrid=False, tickangle=45, linecolor=c['border'])
    fig.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.12)', linecolor=c['border'])
    return fig


def create_top_factors_chart(dff, c):
    if dff.empty:
        return go.Figure().update_layout(
            annotations=[dict(text="No data for selected filters", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=c['muted'], size=14))]
        )

    top_factors = (
        dff['CONTRIBUTING FACTOR VEHICLE 1']
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_factors.columns = ['Factor', 'Count']

    fig = px.bar(
        top_factors,
        y='Factor',
        x='Count',
        orientation='h',
        title="Top 10 Contributing Factors",
        labels={'Factor': 'Contributing Factor', 'Count': 'Number of Crashes'},
        color='Count',
        color_continuous_scale='Oranges'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui", size=12, color=c['text']),
        title_font=dict(size=15, color=c['text']),
        title_x=0,
        margin=dict(l=80, r=10, t=40, b=20),
        coloraxis_showscale=False
    )
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(148,163,184,0.12)', linecolor=c['border'])
    return fig


def create_time_of_day_chart(dff, c):
    if dff.empty:
        return go.Figure().update_layout(
            annotations=[dict(text="No data for selected filters", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=c['muted'], size=14))]
        )

    time_counts = dff['TIME_OF_DAY'].value_counts().reset_index()
    time_counts.columns = ['Time Period', 'Count']

    fig = go.Figure(
        data=[
            go.Pie(
                labels=time_counts['Time Period'],
                values=time_counts['Count'],
                hole=0.55,
                marker=dict(colors=['#0f766e', '#2563eb', '#ea580c', '#7c3aed'])
            )
        ]
    )
    fig.update_layout(
        title="Crashes by Time of Day",
        font=dict(family="Inter, system-ui", size=12, color=c['text']),
        title_font=dict(size=15, color=c['text']),
        title_x=0,
        margin=dict(l=10, r=10, t=40, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent',
        hovertemplate=(
            '<b>%{label}</b><br>'
            'Crashes: %{value:,}<br>'
            'Percentage: %{percent}<extra></extra>'
        )
    )
    return fig


def create_injury_severity_chart(dff, c):
    if dff.empty:
        return go.Figure().update_layout(
            annotations=[dict(text="No data for selected filters", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=c['muted'], size=14))]
        )

    injury_data = {
        'Category': ['Pedestrians', 'Cyclists', 'Motorists'],
        'Injured': [
            int(dff['NUMBER OF PEDESTRIANS INJURED'].sum()),
            int(dff['NUMBER OF CYCLIST INJURED'].sum()),
            int(dff['NUMBER OF MOTORIST INJURED'].sum())
        ],
        'Killed': [
            int(dff['NUMBER OF PEDESTRIANS KILLED'].sum()),
            int(dff['NUMBER OF CYCLIST KILLED'].sum()),
            int(dff['NUMBER OF MOTORIST KILLED'].sum())
        ]
    }

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Injured',
        x=injury_data['Category'],
        y=injury_data['Injured'],
        marker_color='#38bdf8',
        text=injury_data['Injured'],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name='Killed',
        x=injury_data['Category'],
        y=injury_data['Killed'],
        marker_color='#f97373',
        text=injury_data['Killed'],
        textposition='outside'
    ))
    fig.update_layout(
        title="Casualty Severity by User Type",
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui", size=12, color=c['text']),
        title_font=dict(size=15, color=c['text']),
        title_x=0,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=10, t=40, b=40)
    )
    fig.update_xaxes(showgrid=False, linecolor=c['border'])
    fig.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.12)', linecolor=c['border'])
    return fig


def create_hourly_distribution_chart(dff, c):
    if dff.empty:
        return go.Figure().update_layout(
            annotations=[dict(text="No data for selected filters", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=c['muted'], size=14))]
        )

    hourly = dff.groupby('CRASH_HOUR').size().reset_index(name='Crashes')
    hourly = hourly.sort_values('CRASH_HOUR')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly['CRASH_HOUR'],
        y=hourly['Crashes'],
        mode='lines+markers',
        line=dict(color='#22c55e', width=3),
        marker=dict(size=6, color='#bbf7d0'),
        fill='tozeroy',
        fillcolor='rgba(34,197,94,0.12)',
        hovertemplate='Hour: %{x}:00<br>Crashes: %{y:,}<extra></extra>'
    ))
    fig.update_layout(
        title="Hourly Crash Distribution",
        xaxis_title="Hour of Day",
        yaxis_title="Number of Crashes",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui", size=12, color=c['text']),
        title_font=dict(size=15, color=c['text']),
        title_x=0,
        margin=dict(l=40, r=10, t=40, b=40),
        hovermode='x unified'
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=c['border'],
        tickmode='linear',
        tick0=0,
        dtick=2
    )
    fig.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.12)', linecolor=c['border'])
    return fig


def create_bar_chart(dff, c):
    if dff.empty or 'BOROUGH' not in dff.columns:
        return go.Figure().update_layout(
            annotations=[dict(text="No data for selected filters", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=c['muted'], size=14))]
        )

    borough_counts = dff.groupby('BOROUGH').size().reset_index(name='Crashes')
    borough_counts = borough_counts.sort_values('Crashes', ascending=False)

    fig = px.bar(
        borough_counts,
        x='BOROUGH',
        y='Crashes',
        color='Crashes',
        color_continuous_scale='Blues',
        title="Crashes by Borough",
        labels={'BOROUGH': 'Borough', 'Crashes': 'Number of Crashes'}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui", size=12, color=c['text']),
        title_font=dict(size=15, color=c['text']),
        title_x=0,
        hovermode='x unified',
        margin=dict(l=30, r=10, t=40, b=40),
        coloraxis_showscale=False
    )
    fig.update_xaxes(showgrid=False, linecolor=c['border'])
    fig.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.12)', linecolor=c['border'])
    return fig


def create_line_chart(dff, c):
    if dff.empty:
        return go.Figure().update_layout(
            annotations=[dict(text="No data for selected filters", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=c['muted'], size=14))]
        )

    yearly_counts = dff.groupby('CRASH_YEAR').size().reset_index(name='Crashes')
    yearly_counts = yearly_counts.sort_values('CRASH_YEAR')

    fig = px.line(
        yearly_counts,
        x='CRASH_YEAR',
        y='Crashes',
        markers=True,
        title="Crash Trends Over Time",
        labels={'CRASH_YEAR': 'Year', 'Crashes': 'Number of Crashes'}
    )
    fig.update_traces(
        line_color='#6366f1',
        line_width=3,
        marker=dict(size=7, color='#a5b4fc', line=dict(width=1, color='#4f46e5'))
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui", size=12, color=c['text']),
        title_font=dict(size=15, color=c['text']),
        title_x=0,
        hovermode='x unified',
        margin=dict(l=30, r=10, t=40, b=40)
    )
    fig.update_xaxes(showgrid=False, linecolor=c['border'])
    fig.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.12)', linecolor=c['border'])
    return fig


def create_heatmap(dff, c):
    if dff.empty:
        return go.Figure().update_layout(
            annotations=[dict(text="No data for selected filters", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=c['muted'], size=14))]
        )

    heat_data = dff.groupby(['CRASH_DOW', 'CRASH_HOUR']).size().reset_index(name='count')
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heat_data['CRASH_DOW'] = pd.Categorical(heat_data['CRASH_DOW'], categories=days_order, ordered=True)

    fig = px.density_heatmap(
        heat_data,
        x='CRASH_HOUR',
        y='CRASH_DOW',
        z='count',
        color_continuous_scale='Viridis',
        title="Crash Density: Day of Week × Hour",
        labels={'CRASH_HOUR': 'Hour of Day', 'CRASH_DOW': 'Day of Week', 'count': 'Crashes'}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui", size=12, color=c['text']),
        title_font=dict(size=15, color=c['text']),
        title_x=0,
        margin=dict(l=40, r=10, t=40, b=40),
        coloraxis_colorbar=dict(
            title="Crashes",
            thickness=10,
            len=0.7
        )
    )
    fig.update_xaxes(linecolor=c['border'])
    fig.update_yaxes(linecolor=c['border'])
    return fig


def create_pie_chart(dff, c):
    if dff.empty:
        return go.Figure().update_layout(
            annotations=[dict(text="No data for selected filters", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=c['muted'], size=14))]
        )

    injuries = {
        'Persons Injured': int(dff['NUMBER OF PERSONS INJURED'].sum()),
        'Persons Killed': int(dff['NUMBER OF PERSONS KILLED'].sum()),
        'Pedestrians Injured': int(dff['NUMBER OF PEDESTRIANS INJURED'].sum()),
        'Pedestrians Killed': int(dff['NUMBER OF PEDESTRIANS KILLED'].sum()),
        'Cyclists Injured': int(dff['NUMBER OF CYCLIST INJURED'].sum()),
        'Cyclists Killed': int(dff['NUMBER OF CYCLIST KILLED'].sum()),
        'Motorists Injured': int(dff['NUMBER OF MOTORIST INJURED'].sum()),
        'Motorists Killed': int(dff['NUMBER OF MOTORIST KILLED'].sum()),
    }

    pie_data = pd.DataFrame(list(injuries.items()), columns=['Injury Type', 'Count'])
    pie_data = pie_data[pie_data['Count'] > 0]

    fig = px.pie(
        pie_data,
        names='Injury Type',
        values='Count',
        title="Injury Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
    )
    fig.update_layout(
        font=dict(family="Inter, system-ui", size=12, color=c['text']),
        title_font=dict(size=15, color=c['text']),
        title_x=0,
        margin=dict(l=10, r=10, t=40, b=20),
        showlegend=False
    )
    return fig


def create_map_chart(dff, c):
    if dff.empty or dff[['LATITUDE', 'LONGITUDE']].dropna().empty:
        return go.Figure().update_layout(
            annotations=[dict(text="No location data for current filters", x=0.5, y=0.5,
                              showarrow=False, font=dict(color=c['muted'], size=14))]
        )

    map_data = dff.dropna(subset=['LATITUDE', 'LONGITUDE'])
    sample_size = min(1500, len(map_data))
    if len(map_data) > sample_size:
        map_data = map_data.sample(n=sample_size, random_state=42)

    fig = px.scatter_mapbox(
        map_data,
        lat="LATITUDE",
        lon="LONGITUDE",
        hover_name="BOROUGH",
        hover_data={
            "CRASH DATE": True,
            "BOROUGH": True,
            "LATITUDE": False,
            "LONGITUDE": False
        },
        color_discrete_sequence=[c['danger']],
        zoom=9.5,
        height=500,
        title=f"Crash Locations (Sample {sample_size:,} of {len(dff):,})"
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        font=dict(family="Inter, system-ui", size=12, color=c['text']),
        title_font=dict(size=15, color=c['text']),
        title_x=0,
        margin=dict(r=0, t=40, l=0, b=0),
        hovermode='closest'
    )
    return fig

# ============================================================================
# SUMMARY / EXPORT HELPERS
# ============================================================================

def generate_summary_report(dff, filters):
    report = f"""
NYC MOTOR VEHICLE CRASHES - ANALYSIS REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'=' * 60}

FILTER CRITERIA:
{'-' * 60}
"""
    for key, value in filters.items():
        if value and value != 'All':
            report += f"{key}: {value}\n"

    report += f"""
{'=' * 60}

SUMMARY STATISTICS:
{'-' * 60}
Total Crashes: {len(dff):,}
Percentage of Dataset: {(len(dff)/len(df)*100 if len(df) else 0):.2f}%

CASUALTIES:
{'-' * 60}
Total Persons Injured: {int(dff['NUMBER OF PERSONS INJURED'].sum()):,}
Total Persons Killed: {int(dff['NUMBER OF PERSONS KILLED'].sum()):,}

Pedestrians Injured: {int(dff['NUMBER OF PEDESTRIANS INJURED'].sum()):,}
Pedestrians Killed: {int(dff['NUMBER OF PEDESTRIANS KILLED'].sum()):,}

Cyclists Injured: {int(dff['NUMBER OF CYCLIST INJURED'].sum()):,}
Cyclists Killed: {int(dff['NUMBER OF CYCLIST KILLED'].sum()):,}

Motorists Injured: {int(dff['NUMBER OF MOTORIST INJURED'].sum()):,}
Motorists Killed: {int(dff['NUMBER OF MOTORIST KILLED'].sum()):,}

TOP STATISTICS:
{'-' * 60}
"""
    if 'BOROUGH' in dff.columns and not dff.empty:
        top_boroughs = dff['BOROUGH'].value_counts().head(5)
        report += "\nTop 5 Boroughs by Crashes:\n"
        for borough, count in top_boroughs.items():
            report += f"  {borough}: {count:,}\n"

    if 'CONTRIBUTING FACTOR VEHICLE 1' in dff.columns and not dff.empty:
        top_factors = dff['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().head(5)
        report += "\nTop 5 Contributing Factors:\n"
        for factor, count in top_factors.items():
            report += f"  {factor}: {count:,}\n"

    return report

# ============================================================================
# DASH APP INITIALIZATION
# ============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.MORPH, dbc.icons.FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)
server = app.server
app.title = "NYC Crashes – Pro Analytics"

GLOBAL_STYLE = {
    'fontFamily': "Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
}

def base_card_style(dark):
    c = COLORS_DARK if dark else COLORS_LIGHT
    return {
        'borderRadius': '14px',
        'border': f'1px solid {c["border"]}',
        'background': c['surface'],
        'boxShadow': '0 14px 45px rgba(15,23,42,0.18)',
        'marginBottom': '18px',
        'overflow': 'hidden',
        'transition': 'all 0.2s ease',
    }

# ============================================================================
# LAYOUT
# ============================================================================

app.layout = dbc.Container(
    fluid=True,
    className="px-3 px-md-4 py-3",
    style={'backgroundColor': COLORS_LIGHT['bg'], **GLOBAL_STYLE},
    children=[
        dcc.Store(id="theme-mode", data="light"),
        dcc.Interval(id="auto-refresh", interval=60_000, n_intervals=0, disabled=True),

        dcc.Download(id="download-csv"),
        dcc.Download(id="download-report"),

        dbc.Row([
            dbc.Col([
                html.Div(
                    id="hero-container",
                    style={
                        'background': COLORS_LIGHT['gradient_hero'],
                        'borderRadius': '20px',
                        'padding': '22px 20px 18px 20px',
                        'color': 'white',
                        'position': 'relative',
                        'overflow': 'hidden',
                        'boxShadow': '0 24px 60px rgba(15,23,42,0.45)'
                    },
                    children=[
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Div([
                                        html.Span(
                                            "NYC Safety Intelligence",
                                            className="badge rounded-pill bg-dark text-uppercase",
                                            style={'letterSpacing': '0.08em', 'fontSize': '0.72rem'}
                                        )
                                    ], className="mb-2"),
                                    html.H1(
                                        [
                                            html.I(className="fas fa-car-burst me-2"),
                                            "Motor Vehicle Crashes"
                                        ],
                                        style={
                                            'fontSize': '1.9rem',
                                            'marginBottom': '6px',
                                            'fontWeight': 700
                                        }
                                    ),
                                    html.P(
                                        "Explore multi-dimensional crash analytics across boroughs, time, severity, "
                                        "and contributing factors with an interactive, studio-grade dashboard.",
                                        style={
                                            'maxWidth': '640px',
                                            'fontSize': '0.9rem',
                                            'opacity': 0.9,
                                            'marginBottom': '10px'
                                        }
                                    ),
                                    html.Span(
                                        html.I(className="fas fa-circle-info ms-1", id="about-info-icon"),
                                        style={'cursor': 'pointer', 'opacity': 0.9}
                                    ),
                                    dbc.Tooltip(
                                        "Interactive dashboard built with Plotly Dash, Bootstrap, and custom theming.",
                                        target="about-info-icon",
                                        placement="right",
                                    ),
                                    dbc.Row([
                                        dbc.Col(
                                            html.Div([
                                                html.Span("Total Records", className="small text-white-50"),
                                                html.Div(f"{len(df):,}", className="fw-semibold",
                                                         style={'fontSize': '1.05rem'})
                                            ]),
                                            width="auto",
                                            className="me-3"
                                        ),
                                        dbc.Col(
                                            dbc.Badge(
                                                [
                                                    html.I(className="fas fa-database me-1"),
                                                    "NYC Open Data  •  Live snapshot"
                                                ],
                                                color="light",
                                                text_color="dark",
                                                pill=True,
                                                className="mt-1"
                                            ),
                                            width="auto"
                                        )
                                    ])
                                ])
                            ], width=9),
                            dbc.Col([
                                html.Div(
                                    style={
                                        'display': 'flex',
                                        'justifyContent': 'flex-end',
                                        'alignItems': 'flex-start',
                                        'gap': '10px'
                                    },
                                    children=[
                                        dbc.ButtonGroup([
                                            dbc.Button(
                                                id="theme-toggle",
                                                color="secondary",
                                                outline=True,
                                                size="sm",
                                                children=[
                                                    html.I(className="fas fa-moon me-1"),
                                                    "Dark"
                                                ],
                                                style={'backdropFilter': 'blur(6px)'}
                                            ),
                                            dbc.Button(
                                                id="refresh-toggle",
                                                color="secondary",
                                                outline=True,
                                                size="sm",
                                                children=[
                                                    html.I(className="fas fa-sync-alt me-1"),
                                                    "Auto-refresh"
                                                ],
                                                style={'backdropFilter': 'blur(6px)'}
                                            )
                                        ])
                                    ]
                                )
                            ], width=3, className="text-end")
                        ])
                    ]
                )
            ], width=12)
        ], className="mb-3"),

        dbc.Row(
            id="kpi-row",
            className="g-3 mb-2",
        ),

        dbc.Row([
            dbc.Col(
                width=12,
                lg=3,
                children=[
                    html.Div(id="sidebar-wrapper")
                ]
            ),
            dbc.Col(
                width=12,
                lg=9,
                children=[
                    dcc.Loading(
                        id="loading-spinner",
                        type="default",
                        color=COLORS_LIGHT['primary'],
                        children=[
                            html.Div(id="charts-grid")
                        ]
                    )
                ]
            )
        ]),

        dbc.Row([
            dbc.Col([
                html.Hr(className="my-4"),
                html.P(
                    [
                        "Data Source: NYC Open Data • Dashboard v3.0 • Last Updated: ",
                        html.Strong(datetime.now().strftime("%B %d, %Y at %I:%M %p"))
                    ],
                    className="text-center text-muted small"
                )
            ], width=12)
        ])
    ]
)

# ============================================================================
# THEME-SENSITIVE LAYOUT FRAGMENTS
# ============================================================================

def build_sidebar(dark):
    c = COLORS_DARK if dark else COLORS_LIGHT
    return html.Div(
        style={
            'position': 'sticky',
            'top': '82px',
            'height': 'calc(100vh - 110px)',
            'overflowY': 'auto',
            'padding': '16px 16px 18px 16px',
            'backgroundColor': c['surface'],
            'borderRadius': '16px',
            'border': f'1px solid {c["border"]}',
            'boxShadow': '0 18px 45px rgba(15,23,42,0.25)' if dark else '0 12px 32px rgba(15,23,42,0.18)'
        },
        children=[
            html.Div([
                html.Div([
                    html.I(
                        className="fas fa-sliders-h me-2",
                        style={'color': c['primary']}
                    ),
                    html.Span("Filters & Search", className="fw-semibold")
                ], className="mb-1"),
                html.Small("Refine the view with intelligent filters or natural language search.",
                           className="text-muted")
            ], className="mb-2"),

            html.Hr(style={'margin': '10px 0 14px 0'}),

            html.Label(
                [html.I(className="fas fa-search me-1"), "Smart Search"],
                style={'fontWeight': 600, 'color': c['text_soft'], 'fontSize': '0.8rem'}
            ),
            dbc.Input(
                id='search',
                type='text',
                placeholder="e.g., Brooklyn January 2023 night pedestrians",
                className="mb-1",
                style={'borderRadius': '10px', 'fontSize': '0.8rem'}
            ),
            html.Small(
                "Understands boroughs, years, months, time of day & injury type.",
                className="text-muted d-block mb-2"
            ),

            html.Hr(style={'margin': '10px 0'}),

            html.Label(
                [html.I(className="fas fa-map-marker-alt me-1"), "Borough"],
                style={'fontWeight': 600, 'color': c['text_soft'], 'fontSize': '0.8rem'}
            ),
            dcc.Dropdown(
                id='borough',
                options=[{'label': b, 'value': b} for b in BOROUGHS],
                value='All',
                clearable=False,
                style={'marginBottom': '10px', 'fontSize': '0.8rem'}
            ),

            html.Label(
                [html.I(className="fas fa-calendar-alt me-1"), "Year"],
                style={'fontWeight': 600, 'color': c['text_soft'], 'fontSize': '0.8rem'}
            ),
            dcc.Dropdown(
                id='year',
                options=[{'label': y, 'value': y} for y in YEARS],
                value='All',
                clearable=False,
                style={'marginBottom': '10px', 'fontSize': '0.8rem'}
            ),

            html.Label(
                [html.I(className="fas fa-calendar me-1"), "Month"],
                style={'fontWeight': 600, 'color': c['text_soft'], 'fontSize': '0.8rem'}
            ),
            dcc.Dropdown(
                id='month',
                options=[{'label': m, 'value': m} for m in MONTHS],
                value='All',
                clearable=False,
                style={'marginBottom': '10px', 'fontSize': '0.8rem'}
            ),

            html.Label(
                [html.I(className="fas fa-clock me-1"), "Time of Day"],
                style={'fontWeight': 600, 'color': c['text_soft'], 'fontSize': '0.8rem'}
            ),
            dcc.Dropdown(
                id='time_of_day',
                options=[{'label': t, 'value': t} for t in TIME_OF_DAY],
                value='All',
                clearable=False,
                style={'marginBottom': '10px', 'fontSize': '0.8rem'}
            ),

            html.Label(
                [html.I(className="fas fa-car me-1"), "Vehicle Type"],
                style={'fontWeight': 600, 'color': c['text_soft'], 'fontSize': '0.8rem'}
            ),
            dcc.Dropdown(
                id='vehicle',
                options=[{'label': v, 'value': v} for v in VEHICLES],
                value='All',
                clearable=False,
                style={'marginBottom': '10px', 'fontSize': '0.8rem'}
            ),

            html.Label(
                [html.I(className="fas fa-exclamation-triangle me-1"), "Contributing Factor"],
                style={'fontWeight': 600, 'color': c['text_soft'], 'fontSize': '0.8rem'}
            ),
            dcc.Dropdown(
                id='factor',
                options=[{'label': f, 'value': f} for f in FACTORS],
                value='All',
                clearable=False,
                style={'marginBottom': '10px', 'fontSize': '0.8rem'}
            ),

            html.Label(
                [html.I(className="fas fa-user-injured me-1"), "Injury Focus"],
                style={'fontWeight': 600, 'color': c['text_soft'], 'fontSize': '0.8rem'}
            ),
            dcc.Dropdown(
                id='injury',
                options=[{'label': i, 'value': i} for i in INJURIES],
                value='All',
                clearable=False,
                style={'marginBottom': '14px', 'fontSize': '0.8rem'}
            ),

            dbc.Button(
                [html.I(className="fas fa-chart-line me-2"), "Run Analysis"],
                id="generate",
                color="primary",
                className="w-100 fw-semibold mb-2",
                size="sm",
                style={'borderRadius': '999px', 'padding': '8px 0'}
            ),

            dbc.Button(
                [html.I(className="fas fa-broom me-1"), "Reset search"],
                id="btn-reset",
                color="secondary",
                className="w-100 fw-semibold mb-2",
                size="sm",
                outline=True,
                style={'borderRadius': '999px'}
            ),

            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [html.I(className="fas fa-file-csv me-1"), "CSV"],
                        id="btn-export-csv",
                        color="success",
                        className="w-100 mb-1",
                        size="sm",
                        outline=True
                    )
                ], width=6),
                dbc.Col([
                    dbc.Button(
                        [html.I(className="fas fa-file-alt me-1"), "Report"],
                        id="btn-export-report",
                        color="info",
                        className="w-100 mb-1",
                        size="sm",
                        outline=True
                    )
                ], width=6)
            ], className="mb-1"),

            html.Div(
                id='summary',
                className="p-2",
                style={
                    'backgroundColor': c['surface_alt'],
                    'borderRadius': '12px',
                    'border': f'1px dashed {c["border"]}',
                    'fontSize': '0.78rem'
                }
            )
        ]
    )


def build_kpi_row(dark, total, injured, killed, avg_severity, filters_count):
    c = COLORS_DARK if dark else COLORS_LIGHT

    def kpi_card(icon, label, value, accent_color=None):
        return dbc.Col(
            width=6,
            md=3,
            lg=2,
            className="mb-2",
            children=[
                html.Div(
                    style={
                        'background': c['surface'],
                        'borderRadius': '14px',
                        'padding': '10px 12px',
                        'border': f'1px solid {c["border"]}',
                        'boxShadow': '0 10px 30px rgba(15,23,42,0.16)',
                        'height': '100%'
                    },
                    children=[
                        html.Div([
                            html.Div(
                                icon,
                                style={
                                    'width': '26px', 'height': '26px',
                                    'borderRadius': '999px',
                                    'display': 'flex',
                                    'alignItems': 'center',
                                    'justifyContent': 'center',
                                    'backgroundColor': accent_color or c['primary_soft'],
                                    'color': c['primary'],
                                    'marginRight': '8px',
                                    'fontSize': '0.9rem'
                                }
                            ),
                            html.Div([
                                html.Div(label, className="text-muted",
                                         style={'fontSize': '0.7rem'}),
                                html.Div(value, className="fw-semibold",
                                         style={'fontSize': '0.95rem'})
                            ])
                        ], style={'display': 'flex', 'alignItems': 'center'})
                    ]
                )
            ]
        )

    return dbc.Row(
        className="mt-2 mb-2",
        children=[
            kpi_card(
                html.I(className="fas fa-car-crash"),
                "Crashes",
                f"{total:,}",
                accent_color=c['primary_soft']
            ),
            kpi_card(
                html.I(className="fas fa-user-injured"),
                "Persons Injured",
                f"{injured:,}",
                accent_color=c['warning']
            ),
            kpi_card(
                html.I(className="fas fa-skull-crossbones"),
                "Persons Killed",
                f"{killed:,}",
                accent_color=c['danger']
            ),
            kpi_card(
                html.I(className="fas fa-radiation"),
                "Avg Severity",
                f"{avg_severity:.2f}",
                accent_color=c['accent_soft']
            ),
            kpi_card(
                html.I(className="fas fa-filter"),
                "Active Filters",
                f"{filters_count}",
                accent_color=c['success']
            ),
        ]
    )


def build_charts_grid(dark):
    c = COLORS_DARK if dark else COLORS_LIGHT
    card_style = base_card_style(dark)

    return html.Div(
        children=[
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='map', config={'displayModeBar': False})
                ])
            ], style=card_style, className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='bar', config={'displayModeBar': False})
                        ])
                    ], style=card_style)
                ], width=12, lg=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='line', config={'displayModeBar': False})
                        ])
                    ], style=card_style)
                ], width=12, lg=6)
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='heat', config={'displayModeBar': False})
                        ])
                    ], style=card_style)
                ], width=12, lg=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='pie', config={'displayModeBar': False})
                        ])
                    ], style=card_style)
                ], width=12, lg=6)
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='monthly-trend', config={'displayModeBar': False})
                        ])
                    ], style=card_style)
                ], width=12, lg=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='top-factors', config={'displayModeBar': False})
                        ])
                    ], style=card_style)
                ], width=12, lg=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='time-of-day', config={'displayModeBar': False})
                        ])
                    ], style=card_style)
                ], width=12, lg=4)
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='injury-severity', config={'displayModeBar': False})
                        ])
                    ], style=card_style)
                ], width=12, lg=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='hourly-distribution', config={'displayModeBar': False})
                        ])
                    ], style=card_style)
                ], width=12, lg=6)
            ])
        ]
    )

# ============================================================================
# CALLBACKS: THEME & LAYOUT
# ============================================================================

@app.callback(
    Output("theme-mode", "data"),
    Input("theme-toggle", "n_clicks"),
    State("theme-mode", "data"),
    prevent_initial_call=True
)
def toggle_theme(n_clicks, current):
    return "dark" if current == "light" else "light"


@app.callback(
    Output("auto-refresh", "disabled"),
    Input("refresh-toggle", "n_clicks"),
    State("auto-refresh", "disabled"),
    prevent_initial_call=True
)
def toggle_autorefresh(n_clicks, disabled):
    return not disabled


@app.callback(
    Output("sidebar-wrapper", "children"),
    Output("charts-grid", "children"),
    Output("loading-spinner", "color"),
    Output("hero-container", "style"),
    Input("theme-mode", "data")
)
def refresh_theming(theme_mode):
    dark = theme_mode == "dark"
    c = COLORS_DARK if dark else COLORS_LIGHT

    hero_style = {
        'background': c['gradient_hero'],
        'borderRadius': '20px',
        'padding': '22px 20px 18px 20px',
        'color': 'white',
        'position': 'relative',
        'overflow': 'hidden',
        'boxShadow': '0 24px 60px rgba(15,23,42,0.55)' if dark else '0 24px 60px rgba(15,23,42,0.45)'
    }

    return (
        build_sidebar(dark),
        build_charts_grid(dark),
        c['primary'],
        hero_style
    )

# ============================================================================
# MAIN DASHBOARD CALLBACK (10 FIGURES + KPI + SUMMARY)
# ============================================================================

@app.callback(
    [
        Output('map', 'figure'),
        Output('bar', 'figure'),
        Output('line', 'figure'),
        Output('heat', 'figure'),
        Output('pie', 'figure'),
        Output('monthly-trend', 'figure'),
        Output('top-factors', 'figure'),
        Output('time-of-day', 'figure'),
        Output('injury-severity', 'figure'),
        Output('hourly-distribution', 'figure'),
        Output('summary', 'children'),
        Output('kpi-row', 'children'),
        Output('borough', 'value'),
        Output('year', 'value'),
        Output('month', 'value'),
        Output('time_of_day', 'value'),
        Output('vehicle', 'value'),
        Output('factor', 'value'),
        Output('injury', 'value')
    ],
    [
        Input('generate', 'n_clicks'),
        Input('auto-refresh', 'n_intervals')
    ],
    [
        State('borough', 'value'),
        State('year', 'value'),
        State('month', 'value'),
        State('time_of_day', 'value'),
        State('vehicle', 'value'),
        State('factor', 'value'),
        State('injury', 'value'),
        State('search', 'value'),
        State("theme-mode", "data")
    ]
)
def update_dashboard(n_clicks, n_intervals, borough, year, month, time_of_day,
                     vehicle, factor, injury, search, theme_mode):
    logger.info("Dashboard update triggered")
    dark = theme_mode == "dark"
    c = COLORS_DARK if dark else COLORS_LIGHT

    parsed = parse_search(search or "")
    filters = {
        'BOROUGH': parsed['BOROUGH'] or borough,
        'CRASH_YEAR': parsed['CRASH_YEAR'] or year,
        'CRASH_MONTH_NAME': parsed['CRASH_MONTH_NAME'] or month,
        'TIME_OF_DAY': parsed['TIME_OF_DAY'] or time_of_day,
        'VEHICLE TYPE CODE 1': parsed['VEHICLE TYPE CODE 1'] or vehicle,
        'CONTRIBUTING FACTOR VEHICLE 1': parsed['CONTRIBUTING FACTOR VEHICLE 1'] or factor,
        'INJURY_TYPE': parsed['INJURY_TYPE'] or injury
    }

    dff = apply_filters(df, filters)
    if filters['INJURY_TYPE'] and filters['INJURY_TYPE'] != 'All':
        dff = filter_by_injury_type(dff, filters['INJURY_TYPE'])

    total_crashes = len(dff)
    total_injured = int(dff['NUMBER OF PERSONS INJURED'].sum()) if not dff.empty else 0
    total_killed = int(dff['NUMBER OF PERSONS KILLED'].sum()) if not dff.empty else 0
    percentage = (total_crashes / len(df) * 100) if len(df) > 0 else 0
    avg_severity = dff['SEVERITY_SCORE'].mean() if len(dff) > 0 else 0
    filters_count = sum(1 for v in filters.values() if v and v != 'All')

    if dff.empty:
        summary = html.Div([
            html.Div([
                html.Span(
                    "No records match your current filter set.",
                    className="fw-semibold",
                    style={'color': c['danger']}
                )
            ], className="mb-1"),
            html.Small(
                "Try relaxing some filters or clearing the smart search query.",
                className="text-muted"
            )
        ])
    else:
        summary = html.Div([
            html.Div([
                html.Span("Selection Overview", className="fw-semibold me-2"),
                dbc.Badge(
                    f"{total_crashes:,} rows",
                    color="primary",
                    pill=True,
                    className="me-1"
                ),
                dbc.Badge(
                    f"{percentage:.1f}% of dataset",
                    color="secondary",
                    pill=True
                )
            ], className="mb-1"),
            html.Div([
                html.Span("Injured: ", className="text-muted small me-1"),
                html.Span(f"{total_injured:,}", className="small fw-semibold me-2"),
                html.Span("Killed: ", className="text-muted small me-1"),
                html.Span(f"{total_killed:,}", className="small fw-semibold me-2"),
                html.Span("Avg severity: ", className="text-muted small me-1"),
                html.Span(f"{avg_severity:.2f}", className="small fw-semibold me-2"),
                dbc.Badge(
                    "Cleaned dataset",
                    color="success",
                    pill=True,
                    className="ms-1",
                    style={'fontSize': '0.7rem'}
                )
            ])
        ])

    kpi_children = build_kpi_row(
        dark,
        total_crashes,
        total_injured,
        total_killed,
        avg_severity,
        filters_count
    )

    map_fig = create_map_chart(dff, c)
    bar_fig = create_bar_chart(dff, c)
    line_fig = create_line_chart(dff, c)
    heat_fig = create_heatmap(dff, c)
    pie_fig = create_pie_chart(dff, c)
    monthly_fig = create_monthly_trend_chart(dff, c)
    factors_fig = create_top_factors_chart(dff, c)
    time_fig = create_time_of_day_chart(dff, c)
    injury_fig = create_injury_severity_chart(dff, c)
    hourly_fig = create_hourly_distribution_chart(dff, c)

    logger.info("Dashboard update completed")

    return (
        map_fig, bar_fig, line_fig, heat_fig, pie_fig,
        monthly_fig, factors_fig, time_fig, injury_fig, hourly_fig,
        summary, kpi_children,
        filters['BOROUGH'], filters['CRASH_YEAR'], filters['CRASH_MONTH_NAME'],
        filters['TIME_OF_DAY'], filters['VEHICLE TYPE CODE 1'],
        filters['CONTRIBUTING FACTOR VEHICLE 1'], filters['INJURY_TYPE']
    )

# ============================================================================
# RESET FILTERS CALLBACK (ONLY CLEARS SEARCH)
# ============================================================================

@app.callback(
    Output('search', 'value'),
    Input('btn-reset', 'n_clicks'),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    return ''

# ============================================================================
# EXPORT CALLBACKS
# ============================================================================

@app.callback(
    Output("download-csv", "data"),
    [Input("btn-export-csv", "n_clicks")],
    [
        State('borough', 'value'),
        State('year', 'value'),
        State('month', 'value'),
        State('time_of_day', 'value'),
        State('vehicle', 'value'),
        State('factor', 'value'),
        State('injury', 'value')
    ],
    prevent_initial_call=True
)
def export_csv(n_clicks, borough, year, month, time_of_day, vehicle, factor, injury):
    if n_clicks is None:
        return None
    from dash import dcc

    filters = {
        'BOROUGH': borough,
        'CRASH_YEAR': year,
        'CRASH_MONTH_NAME': month,
        'TIME_OF_DAY': time_of_day,
        'VEHICLE TYPE CODE 1': vehicle,
        'CONTRIBUTING FACTOR VEHICLE 1': factor,
        'INJURY_TYPE': injury
    }
    dff = apply_filters(df, filters)
    if filters['INJURY_TYPE'] and filters['INJURY_TYPE'] != 'All':
        dff = filter_by_injury_type(dff, filters['INJURY_TYPE'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nyc_crashes_filtered_{timestamp}.csv"

    logger.info(f"Exporting {len(dff)} records to CSV")
    return dcc.send_data_frame(dff.to_csv, filename, index=False)


@app.callback(
    Output("download-report", "data"),
    [Input("btn-export-report", "n_clicks")],
    [
        State('borough', 'value'),
        State('year', 'value'),
        State('month', 'value'),
        State('time_of_day', 'value'),
        State('vehicle', 'value'),
        State('factor', 'value'),
        State('injury', 'value')
    ],
    prevent_initial_call=True
)
def export_report(n_clicks, borough, year, month, time_of_day, vehicle, factor, injury):
    if n_clicks is None:
        return None

    filters = {
        'BOROUGH': borough,
        'CRASH_YEAR': year,
        'CRASH_MONTH_NAME': month,
        'TIME_OF_DAY': time_of_day,
        'VEHICLE TYPE CODE 1': vehicle,
        'CONTRIBUTING FACTOR VEHICLE 1': factor,
        'INJURY_TYPE': injury
    }

    dff = apply_filters(df, filters)
    if filters['INJURY_TYPE'] and filters['INJURY_TYPE'] != 'All':
        dff = filter_by_injury_type(dff, filters['INJURY_TYPE'])

    report_text = generate_summary_report(dff, filters)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nyc_crashes_report_{timestamp}.txt"

    logger.info("Generating summary report")
    return dict(content=report_text, filename=filename)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting NYC Crashes Dashboard – Pro UI...")
    logger.info("Dashboard running at http://127.0.0.1:8050")
    logger.info(f"Loaded {len(df):,} crash records")
    app.run(
        debug=True,
        port=8050,
        host='127.0.0.1'
    )
