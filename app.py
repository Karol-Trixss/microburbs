import os
import math
import json
import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px


API_BASE_URL = "https://www.microburbs.com.au/report_generator/api/suburb/properties"


def _coerce_none_nan(value: Any) -> Optional[float]:
    if value is None:
        return np.nan
    if isinstance(value, float) and math.isnan(value):
        return np.nan
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"none", "nan", "", "null"}:
            return np.nan
        try:
            return float(token)
        except Exception:
            return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    return np.nan


def _parse_area_to_sqm(value: Any) -> Optional[float]:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        txt = value.strip().lower()
        if txt in {"none", "nan", "", "null"}:
            return np.nan
        # Extract number and optional unit
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(m2|m\^2|sqm|m²|ha|hectare|hectares)?", txt)
        if not m:
            # Try to parse raw float
            try:
                return float(txt)
            except Exception:
                return np.nan
        val = float(m.group(1))
        unit = (m.group(2) or "").lower()
        if unit in {"ha", "hectare", "hectares"}:
            return val * 10000.0  # hectares to sqm
        # Default assume square meters
        return val
    return np.nan


def _safe_parse_date(value: Any) -> Optional[pd.Timestamp]:
    if value in (None, "", "None", "nan"):
        return pd.NaT
    try:
        return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_properties(suburb: str, token: str) -> Dict[str, Any]:
    params = {"suburb": suburb}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.get(API_BASE_URL, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def normalize_results(payload: Dict[str, Any]) -> pd.DataFrame:
    results = payload.get("results", []) or []
    if not results:
        return pd.DataFrame()

    # Flatten records
    flattened: List[Dict[str, Any]] = []
    for rec in results:
        address = rec.get("address", {}) or {}
        attrs = rec.get("attributes", {}) or {}
        coords = rec.get("coordinates", {}) or {}
        flattened.append(
            {
                "street": address.get("street"),
                "suburb": address.get("sal"),
                "state": address.get("state"),
                "sa1": address.get("sa1"),
                "area_level": rec.get("area_level"),
                "area_name": rec.get("area_name"),
                "bedrooms": attrs.get("bedrooms"),
                "bathrooms": attrs.get("bathrooms"),
                "garage_spaces": attrs.get("garage_spaces"),
                "land_size_raw": attrs.get("land_size"),
                "building_size_raw": attrs.get("building_size"),
                "description": attrs.get("description"),
                "latitude": coords.get("latitude"),
                "longitude": coords.get("longitude"),
                "gnaf_pid": rec.get("gnaf_pid"),
                "listing_date_raw": rec.get("listing_date"),
                "price": rec.get("price"),
                "property_type": rec.get("property_type"),
            }
        )

    df = pd.DataFrame(flattened)

    # Coerce numerics
    df["bedrooms"] = df["bedrooms"].map(_coerce_none_nan)
    df["bathrooms"] = df["bathrooms"].map(_coerce_none_nan)
    df["garage_spaces"] = df["garage_spaces"].map(_coerce_none_nan)
    df["price"] = df["price"].map(_coerce_none_nan)
    df["land_size_sqm"] = df["land_size_raw"].map(_parse_area_to_sqm)
    df["building_size_sqm"] = df["building_size_raw"].map(_parse_area_to_sqm)
    df["listing_date"] = df["listing_date_raw"].map(_safe_parse_date)

    # Useful derived fields
    with np.errstate(divide="ignore", invalid="ignore"):
        df["price_per_bedroom"] = df["price"] / df["bedrooms"]

    # Sort by date desc then price desc
    df = df.sort_values(by=["listing_date", "price"], ascending=[False, False], na_position="last")
    return df


def executive_summary(df: pd.DataFrame) -> None:
    """Professional, muted executive summary panel."""
    if df.empty:
        st.info("No data available for summary.")
        return

    # Neutral container with subtle border and accent bar
    summary_html = """
    <div style="background:#111827; border:1px solid #374151; border-radius:12px; padding:18px 22px; margin:14px 0;">
      <div style="height:6px; background:#3B82F6; border-radius:4px; margin-bottom:14px;"></div>
      <h3 style="color:#E5E7EB; margin:0;">Executive Summary</h3>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)

    price_series = df["price"].dropna()
    land_series = df["land_size_sqm"].dropna()

    # Top insights in muted cards
    insight_cols = st.columns(3)

    with insight_cols[0]:
        if not price_series.empty:
            max_price_idx = df["price"].idxmax()
            max_property = df.loc[max_price_idx]
            st.markdown(f"""
            <div style=\"background:#0B1220; border:1px solid #2A3342; border-radius:10px; padding:14px; color:#E5E7EB;\">
              <div style=\"font-weight:600; color:#93C5FD; margin-bottom:8px;\">Most Expensive</div>
              <div style=\"font-weight:600;\">{max_property.get('area_name', 'N/A')}</div>
              <div>Price: {_fmt_currency(max_property.get('price'))}</div>
              <div>Type: {max_property.get('property_type', '—')}</div>
            </div>
            """, unsafe_allow_html=True)

    with insight_cols[1]:
        if not land_series.empty:
            max_land_idx = df["land_size_sqm"].idxmax()
            max_land_property = df.loc[max_land_idx]
            st.markdown(f"""
            <div style=\"background:#0B1220; border:1px solid #2A3342; border-radius:10px; padding:14px; color:#E5E7EB;\">
              <div style=\"font-weight:600; color:#93C5FD; margin-bottom:8px;\">Largest Land</div>
              <div style=\"font-weight:600;\">{max_land_property.get('area_name', 'N/A')}</div>
              <div>Size: {_fmt_area(max_land_property.get('land_size_sqm'))}</div>
              <div>Price: {_fmt_currency(max_land_property.get('price'))}</div>
            </div>
            """, unsafe_allow_html=True)

    with insight_cols[2]:
        if df["listing_date"].notna().any():
            recent_idx = df["listing_date"].idxmax()
            recent_property = df.loc[recent_idx]
            st.markdown(f"""
            <div style=\"background:#0B1220; border:1px solid #2A3342; border-radius:10px; padding:14px; color:#E5E7EB;\">
              <div style=\"font-weight:600; color:#93C5FD; margin-bottom:8px;\">Most Recent</div>
              <div style=\"font-weight:600;\">{recent_property.get('area_name', 'N/A')}</div>
              <div>Listed: {_fmt_date(recent_property.get('listing_date'))}</div>
              <div>Price: {_fmt_currency(recent_property.get('price'))}</div>
            </div>
            """, unsafe_allow_html=True)

    # Market statistics with muted cards
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    stat_cols = st.columns(4)

    with stat_cols[0]:
        st.markdown("""
        <div style=\"background:#0B1220; border:1px solid #2A3342; border-radius:10px; padding:12px; color:#E5E7EB;\">
          <div style=\"font-weight:600; color:#9CA3AF;\">Price Range</div>
        </div>
        """, unsafe_allow_html=True)
        if not price_series.empty:
            st.markdown(f"Min: {_fmt_currency(price_series.min())}")
            st.markdown(f"Max: {_fmt_currency(price_series.max())}")

    with stat_cols[1]:
        st.markdown("""
        <div style=\"background:#0B1220; border:1px solid #2A3342; border-radius:10px; padding:12px; color:#E5E7EB;\">
          <div style=\"font-weight:600; color:#9CA3AF;\">Price Statistics</div>
        </div>
        """, unsafe_allow_html=True)
        if not price_series.empty:
            q1 = float(price_series.quantile(0.25))
            median = float(price_series.median())
            q3 = float(price_series.quantile(0.75))
            st.markdown(f"Q1: {_fmt_currency(q1)}")
            st.markdown(f"Median: {_fmt_currency(median)}")
            st.markdown(f"Q3: {_fmt_currency(q3)}")

    with stat_cols[2]:
        st.markdown("""
        <div style=\"background:#0B1220; border:1px solid #2A3342; border-radius:10px; padding:12px; color:#E5E7EB;\">
          <div style=\"font-weight:600; color:#9CA3AF;\">Property Mix</div>
        </div>
        """, unsafe_allow_html=True)
        type_counts = df["property_type"].value_counts()
        for prop_type, count in type_counts.items():
            st.markdown(f"{prop_type}: {count}")

    with stat_cols[3]:
        st.markdown("""
        <div style=\"background:#0B1220; border:1px solid #2A3342; border-radius:10px; padding:12px; color:#E5E7EB;\">
          <div style=\"font-weight:600; color:#9CA3AF;\">Bedroom Mix</div>
        </div>
        """, unsafe_allow_html=True)
        bed_counts = df["bedrooms"].value_counts().sort_index()
        for beds, count in bed_counts.items():
            st.markdown(f"{int(beds) if beds == int(beds) else beds} BR: {count}")


def kpi_section(df: pd.DataFrame) -> None:
    total_listings = int(df.shape[0])
    price_series = df["price"] if "price" in df else pd.Series(dtype=float)
    land_series = df["land_size_sqm"] if "land_size_sqm" in df else pd.Series(dtype=float)
    ppb_series = df["price_per_bedroom"] if "price_per_bedroom" in df else pd.Series(dtype=float)
    median_price = float(np.nanmedian(price_series)) if total_listings and price_series.notna().any() else np.nan
    avg_land = float(np.nanmean(land_series)) if total_listings and land_series.notna().any() else np.nan
    median_ppb = float(np.nanmedian(ppb_series)) if total_listings and ppb_series.notna().any() else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Listings", f"{total_listings:,}")
    c2.metric("Median Listed Price", _fmt_currency(median_price))
    c3.metric("Average Land Size", _fmt_area(avg_land))
    c4.metric("Median Price per Bedroom", _fmt_currency(median_ppb))


def _fmt_currency(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "–"
    try:
        return f"${value:,.0f}"
    except Exception:
        return "–"


def _fmt_area(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "–"
    try:
        if value >= 10000:
            return f"{value/10000:.2f} ha"
        return f"{value:,.0f} m²"
    except Exception:
        return "–"


def map_section(df: pd.DataFrame) -> None:
    df_map = df.dropna(subset=["latitude", "longitude"]).copy()
    if df_map.empty:
        st.info("No coordinates available for mapping.")
        return
    mean_lat = float(df_map["latitude"].mean())
    mean_lng = float(df_map["longitude"].mean())

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position="[longitude, latitude]",
        get_fill_color="[200, 30, 0, 160]",
        get_radius=20,
        pickable=True,
    )
    tooltip = {
        "html": "<b>{area_name}</b><br/>Price: {price}<br/>Bedrooms: {bedrooms}",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }
    view_state = pdk.ViewState(latitude=mean_lat, longitude=mean_lng, zoom=12)
    st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=view_state, layers=[layer], tooltip=tooltip))


def table_section(df: pd.DataFrame, selected_idx: Optional[int] = None) -> Optional[int]:
    # Clean column names mapping
    clean_names = {
        "area_name": "Address",
        "property_type": "Property Type",
        "bedrooms": "Bedrooms",
        "bathrooms": "Bathrooms",
        "garage_spaces": "Garage Spaces",
        "price": "Listed Price",
        "land_size_sqm": "Land Size",
        "listing_date": "Listed Date",
    }
    display_cols = [
        "area_name",
        "property_type",
        "bedrooms",
        "bathrooms",
        "garage_spaces",
        "price",
        "land_size_sqm",
        "listing_date",
    ]
    df_show = df[display_cols].copy()
    df_show["price"] = df_show["price"].map(_fmt_currency)
    df_show["land_size_sqm"] = df_show["land_size_sqm"].map(_fmt_area)
    df_show["listing_date"] = df_show["listing_date"].dt.strftime("%Y-%m-%d").fillna("")
    # Rename columns to clean names
    df_show = df_show.rename(columns=clean_names)
    sel_data = st.dataframe(df_show, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="property_table")
    if sel_data and "selection" in sel_data and "rows" in sel_data["selection"] and sel_data["selection"]["rows"]:
        return sel_data["selection"]["rows"][0]
    return selected_idx


def cards_section(df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        with st.expander(f"{row.get('area_name', 'Property')} — {_fmt_currency(row.get('price'))}"):
            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown(
                    f"""
                    - **Bedrooms:** {_fmt_num(row.get('bedrooms'))}
                    - **Bathrooms:** {_fmt_num(row.get('bathrooms'))}
                    - **Parking Spaces:** {_fmt_num(row.get('garage_spaces'))}
                    - **Land Area:** {_fmt_area(row.get('land_size_sqm'))}
                    - **Building Area:** {_fmt_area(row.get('building_size_sqm'))}
                    - **Listed Date:** {_fmt_date(row.get('listing_date'))}
                    - **Property ID:** `{row.get('gnaf_pid') or '—'}`
                    - **Type:** {row.get('property_type') or '—'}
                    """
                )
            with c2:
                st.write(row.get("description") or "(No description provided)")


def _fmt_num(value: Any) -> str:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "–"
        if float(value).is_integer():
            return f"{int(value)}"
        return f"{float(value):.2f}"
    except Exception:
        return "–"


def _fmt_date(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return "–"
        return value.tz_convert(None).strftime("%Y-%m-%d") if value.tzinfo else value.strftime("%Y-%m-%d")
    try:
        dt = pd.to_datetime(value)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "–"


def get_settings() -> Tuple[str, str]:
    token = os.getenv("MICROBURBS_TOKEN", "test")
    suburb = os.getenv("DEFAULT_SUBURB", "Belmont North")
    return suburb, token


def filter_controls(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    with st.expander("🔽 Filters (click to collapse)", expanded=True):
        col1, col2, col3 = st.columns(3)

        min_price = float(np.nanmin(df["price"])) if df["price"].notna().any() else 0.0
        max_price = float(np.nanmax(df["price"])) if df["price"].notna().any() else 0.0
        price_range = col1.slider(
            "Price range",
            min_value=0,
            max_value=int(max(1, max_price)),
            value=(int(min_price) if min_price > 0 else 0, int(max_price)),
            step=10000,
        )

        types = sorted([t for t in df["property_type"].dropna().unique().tolist() if t])
        type_sel = col2.multiselect("Property type", options=types, default=types)

        max_beds = int(np.nanmax(df["bedrooms"])) if df["bedrooms"].notna().any() else 10
        beds = col3.slider("Bedrooms", min_value=0, max_value=max(1, max_beds), value=(0, max(1, max_beds)))

        col4, col5, col6 = st.columns(3)
        max_baths = int(np.nanmax(df["bathrooms"])) if df["bathrooms"].notna().any() else 10
        baths = col4.slider("Bathrooms", min_value=0, max_value=max(1, max_baths), value=(0, max(1, max_baths)))

        land_min = float(np.nanmin(df["land_size_sqm"])) if df["land_size_sqm"].notna().any() else 0.0
        land_max = float(np.nanmax(df["land_size_sqm"])) if df["land_size_sqm"].notna().any() else 0.0
        land = col5.slider("Land size (sqm)", min_value=0, max_value=int(max(1, land_max)), value=(int(land_min) if land_min > 0 else 0, int(land_max)), step=10)

        # Date vs DateTime filter
        if df["listing_date"].notna().any():
            dtmin = pd.to_datetime(df["listing_date"].min()).tz_convert(None)
            dtmax = pd.to_datetime(df["listing_date"].max()).tz_convert(None)
        else:
            dtmin = pd.Timestamp(year=2000, month=1, day=1)
            dtmax = pd.Timestamp.today()
        mode = col6.selectbox("Date filter mode", options=["Date", "DateTime"], index=0)
        if mode == "DateTime":
            dt_range = st.slider("Listing date/time", min_value=dtmin.to_pydatetime(), max_value=dtmax.to_pydatetime(), value=(dtmin.to_pydatetime(), dtmax.to_pydatetime()))
            date_range = (pd.Timestamp(dt_range[0]), pd.Timestamp(dt_range[1]))
        else:
            dmin = dtmin.date()
            dmax = dtmax.date()
            d_range = st.slider("Listing date", min_value=dmin, max_value=dmax, value=(dmin, dmax))
            date_range = (pd.Timestamp(d_range[0]), pd.Timestamp(d_range[1] + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)))

        search = st.text_input("Search (address/description)")

    # Normalize listing_date to naive timestamps for comparison
    series_ts = df["listing_date"].dt.tz_convert(None).fillna(pd.Timestamp("1900-01-01"))
    mask = (
        df["price"].fillna(0).between(price_range[0], price_range[1])
        & df["bedrooms"].fillna(0).between(beds[0], beds[1])
        & df["bathrooms"].fillna(0).between(baths[0], baths[1])
        & df["property_type"].fillna("").isin(type_sel)
        & df["land_size_sqm"].fillna(0).between(land[0], land[1])
        & series_ts.between(date_range[0], date_range[1])
    )
    if search:
        pattern = re.escape(search.strip())
        text_mask = (
            df["description"].fillna("").str.contains(pattern, case=False, regex=True)
            | df["area_name"].fillna("").str.contains(pattern, case=False, regex=True)
        )
        mask = mask & text_mask

    return df[mask].copy()


def download_section(df: pd.DataFrame) -> None:
    if df.empty:
        return
    csv = df.to_csv(index=False)
    st.download_button("Download filtered CSV", data=csv, file_name="properties_filtered.csv", mime="text/csv")


def property_selector_dropdown(df: pd.DataFrame, selected_idx: Optional[int]) -> Optional[int]:
    options = ["All Properties"] + [f"{row.get('area_name', 'Property')} — {_fmt_currency(row.get('price'))}" for _, row in df.iterrows()]
    idx = 0 if selected_idx is None else selected_idx + 1
    sel = st.selectbox("Select a property to view details", options, index=idx, key="property_dropdown")
    return None if sel == "All Properties" else (options.index(sel) - 1 if sel in options else selected_idx)


def detail_section(df: pd.DataFrame, selected_idx: Optional[int]) -> None:
    if selected_idx is None or selected_idx >= len(df):
        st.info("Select a property from the dropdown or table above to view detailed information.")
        return

    row = df.iloc[selected_idx]
    st.markdown("## Property Details")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"**Full Address:** {row.get('area_name', '—')}")
        st.markdown(f"**Property Type:** {row.get('property_type', '—')}")
        st.markdown(f"**Listed Price:** {_fmt_currency(row.get('price'))}")
        st.markdown(f"**Bedrooms:** {_fmt_num(row.get('bedrooms'))}")
        st.markdown(f"**Bathrooms:** {_fmt_num(row.get('bathrooms'))}")
        st.markdown(f"**Parking Spaces:** {_fmt_num(row.get('garage_spaces'))}")
    
    with col2:
        st.markdown(f"**Land Area:** {_fmt_area(row.get('land_size_sqm'))}")
        st.markdown(f"**Building Area:** {_fmt_area(row.get('building_size_sqm'))}")
        st.markdown(f"**Listed Date:** {_fmt_date(row.get('listing_date'))}")
        st.markdown(f"**Property ID:** `{row.get('gnaf_pid') or '—'}`")
        if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
            st.markdown(f"📍 **Coordinates:** [{row.get('latitude'):.6f}, {row.get('longitude'):.6f}]")
    
    st.markdown("---")
    # Mini KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Price per Bedroom", _fmt_currency((row.get("price") or np.nan) / (row.get("bedrooms") or np.nan)))
    land = row.get("land_size_sqm")
    k2.metric("Price per Square Meter", _fmt_currency((row.get("price") or np.nan) / land) if land and not math.isnan(land) and land > 0 else "–")
    k3.metric("Geo Coordinates", f"{row.get('latitude'):.4f}, {row.get('longitude'):.4f}" if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')) else "–")

    st.markdown("### Description")
    st.write(row.get("description") or "(No description provided)")


def charts_section(df: pd.DataFrame) -> None:
    st.subheader("Market Visualizations")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Price distribution**")
        s = df["price"].dropna()
        if not s.empty:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=s, nbinsx=20, name="Properties"))
            fig.update_layout(
                xaxis_title="Price ($)",
                yaxis_title="Count",
                margin={"l": 20, "r": 10, "t": 10, "b": 40},
                showlegend=False,
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price data available.")

    with c2:
        st.markdown("**Listings by date**")
        if df["listing_date"].notna().any():
            ts = df.set_index("listing_date").assign(n=1)["n"].resample("W").sum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines+markers", name="Weekly Listings", line=dict(width=2)))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Listings",
                margin={"l": 20, "r": 10, "t": 10, "b": 40},
                showlegend=False,
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No listing dates available.")

    with c3:
        st.markdown("**Price vs land size**")
        d = df.dropna(subset=["price", "land_size_sqm"])[:500]
        if not d.empty:
            fig = px.scatter(
                d,
                x="land_size_sqm",
                y="price",
                color="property_type",
                hover_data=["area_name"],
                labels={"land_size_sqm": "Land Size (sqm)", "price": "Price ($)"},
            )
            fig.update_traces(marker=dict(size=8, opacity=0.6))
            fig.update_layout(
                margin={"l": 40, "r": 10, "t": 10, "b": 40},
                showlegend=False,
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for scatter plot.")

    # Second row of charts
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Median price by property type**")
        if df["price"].notna().any() and df["property_type"].notna().any():
            grouped = df.groupby("property_type")["price"].median().sort_values(ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=grouped.index, y=grouped.values, name="Median Price", marker_color="steelblue"))
            fig.update_layout(
                xaxis_title="Property Type",
                yaxis_title="Median Price ($)",
                margin={"l": 40, "r": 10, "t": 10, "b": 60},
                showlegend=False,
                height=300,
                xaxis=dict(tickangle=45),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for this chart.")
    with r2:
        st.markdown("**Price by bedrooms (box plot)**")
        d = df.dropna(subset=["price", "bedrooms"])[:1000]
        if not d.empty:
            fig = go.Figure()
            for bed in sorted(d["bedrooms"].unique()):
                subset = d[d["bedrooms"] == bed]
                fig.add_trace(go.Box(y=subset["price"], name=f"{int(bed)} BR" if bed == int(bed) else f"{bed:.1f} BR", boxpoints="outliers"))
            fig.update_layout(
                xaxis_title="Number of Bedrooms",
                yaxis_title="Price ($)",
                margin={"l": 40, "r": 10, "t": 10, "b": 40},
                showlegend=False,
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for box plot.")


def main() -> None:
    st.set_page_config(page_title="Suburb Properties Dashboard", layout="wide")
    st.title("Suburb Properties Dashboard")
    st.caption("Powered by Microburbs API — professional insights for property decisions")

    suburb, token = get_settings()

    try:
        with st.spinner("Fetching data..."):
            payload = fetch_properties(suburb=suburb, token=token)
            df = normalize_results(payload)
    except requests.HTTPError as http_err:
        st.error(f"HTTP error: {http_err}")
        st.stop()
    except Exception as ex:
        st.error(f"Unexpected error: {ex}")
        st.stop()

    if df.empty:
        st.warning("No results returned for this suburb.")
        st.stop()

    # Initialize session state
    if "selected_property_idx" not in st.session_state:
        st.session_state.selected_property_idx = None

    filtered = filter_controls(df)

    # Property selector dropdown at top
    selected_idx = property_selector_dropdown(filtered, st.session_state.selected_property_idx)
    if selected_idx is not None and selected_idx != st.session_state.selected_property_idx:
        st.session_state.selected_property_idx = selected_idx
    elif selected_idx is None:
        st.session_state.selected_property_idx = None

    # Show details for selected property or all
    if st.session_state.selected_property_idx is not None and st.session_state.selected_property_idx < len(filtered):
        df_display = filtered.iloc[[st.session_state.selected_property_idx]]
    else:
        df_display = filtered

    kpi_section(df_display)

    if st.session_state.selected_property_idx is None:
        executive_summary(df_display)
        charts_section(df_display)
    else:
        # Hide charts when viewing single property
        st.markdown("*Charts hidden when viewing a single property.*")

    st.subheader("Map")
    map_section(df_display)

    st.subheader("Results")
    table_idx = table_section(df_display, st.session_state.selected_property_idx)
    if table_idx is not None:
        st.session_state.selected_property_idx = table_idx
    download_section(df_display)

    # Recent listings section (last 5 by date)
    st.subheader("Recent Listings")
    recent = df.sort_values(by=["listing_date"], ascending=[False]).head(5)
    if not recent.empty:
        cards_section(recent)

    detail_section(filtered, st.session_state.selected_property_idx)


if __name__ == "__main__":
    main()


