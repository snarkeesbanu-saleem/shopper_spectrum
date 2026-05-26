# 🛒 Shopper Spectrum – Customer Segmentation Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shopperspectrum-k8dft9thjyeg7rvbamfhus.streamlit.app/))
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**Shopper Spectrum** is an interactive customer analytics dashboard built with **Python**, **Streamlit**, and **Plotly**. It segments customers using **RFM (Recency, Frequency, Monetary) analysis**, helping e‑commerce businesses identify high‑value customers, predict churn risk, and create targeted marketing strategies.

The dashboard is deployed live and accessible to anyone with the link.

🌐 **Live App:** [https://shopperspectrum-k8dft9thjyeg7rvbamfhus.streamlit.app/](https://shopperspectrum-k8dft9thjyeg7rvbamfhus.streamlit.app/)

---

## 📌 Overview

This project demonstrates a complete **customer segmentation pipeline**:

1. **Extract** transaction data from the [Online Retail Data Set (UCI)](https://archive-beta.ics.uci.edu/ml/datasets/Online+Retail).
2. **Transform** raw invoices into RFM metrics (Recency, Frequency, Monetary value) using pandas.
3. **Segment** customers into 11 actionable groups (e.g., Champions, Loyal, At Risk) based on quartile scoring.
4. **Visualize** insights through 20+ interactive charts, including a 3D RFM space, cohort heatmaps, and CLV predictions.
5. **Deploy** the entire pipeline as a live dashboard on Streamlit Cloud.

The dashboard is fully interactive – users can filter by segment, recency, frequency, and monetary value, with all filters **shared via URL parameters** for easy collaboration.

---

## 🎯 Live Dashboard Features

| Category | Features |
|----------|----------|
| **KPIs** | Total customers, average recency/frequency/monetary, predicted CLV |
| **Segmentation** | Pie chart, treemap, and sunburst of RFM segments |
| **Customer Health** | Churn risk gauge, predicted next purchase days, CLV distribution |
| **Advanced Analytics** | Linear‑regression CLV forecast, next purchase estimation, churn risk scoring |
| **Cohort Analysis** | Retention heatmap + revenue per customer over time |
| **Comparison Tools** | Side‑by‑side segment comparison (box plot + radar chart) |
| **Product Insights** | Top 10 frequently bought together items, most popular product descriptions |
| **Interactivity** | Segment‑specific filtering, customer ID search, what‑if simulator for segment prediction |
| **Data Export** | Download filtered customer data as CSV |

The dashboard includes **over 15 distinct chart types**, far beyond standard bar/pie charts – treemaps, 3D scatter, radar, funnel‑style gauges, and animated heatmaps are all used to tell the customer story.

---

## 🛠️ Tech Stack

- **Frontend & Dashboard:** [Streamlit](https://streamlit.io/) (no HTML/JS required)
- **Data Processing:** pandas, numpy
- **Visualization:** Plotly Express + Graph Objects (interactive, publication‑ready)
- **Machine Learning:** scikit‑learn (Linear Regression for CLV, K‑Means for segmentation)
- **Data Source:** [UCI Online Retail Dataset](https://archive-beta.ics.uci.edu/ml/datasets/Online+Retail) – real transactional data (2010‑2011)

---

## 📊 Dataset

The dashboard uses the **Online Retail Data Set** from the UCI Machine Learning Repository. It contains **541,909** transactions from a UK‑based online retailer between 01/12/2010 and 09/12/2011. The dataset includes:

| Column | Description |
|--------|-------------|
| `InvoiceNo` | Unique invoice number |
| `StockCode` | Product code |
| `Description` | Product name |
| `Quantity` | Units purchased |
| `InvoiceDate` | Transaction date/time |
| `UnitPrice` | Price per unit |
| `CustomerID` | Unique customer identifier |
| `Country` | Customer country |

The raw data is processed in real‑time when the app starts. RFM scores are computed and cached for performance.

---
## 🔮 Future Enhancements

- **Real‑time data refresh** – pull latest transactions from an API or database.
- **Email alerts** – notify marketing teams when a high‑value customer becomes “At Risk”.
- **Dynamic clustering** – replace quartile‑based segmentation with K‑Means or DBSCAN for data‑driven segments.
- **Product recommendation engine** – suggest items based on past purchases of similar segments.
- **Animated transitions** – show how customers move between segments over time.

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the Online Retail Dataset.
- **Streamlit** for the amazing framework that turns Python scripts into interactive dashboards.
- **Plotly** for the beautiful, interactive charts.

---

## 💡 Usage

Once the app is running, you can:

- **Filter** customers by segment, recency range, frequency, or monetary value.
- **Search** for a specific customer by ID to see their RFM profile.
- **Compare** two customer segments side by side.
- **Simulate** a hypothetical customer (set custom recency/frequency/monetary) and see which segment they would fall into.
- **Download** the filtered dataset as CSV for further analysis.
