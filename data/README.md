# Data

`geo_all_channels.csv` is the simulated dataset published with
[Google Meridian](https://github.com/google/meridian) (Apache License 2.0), copied from
`meridian/data/simulated_data/csv/geo_all_channels.csv`. A copy is kept here so the results are
reproducible even if the upstream file changes.

One row per region (`geo`, 40 regions) and week (`time`, 156 weeks from 2021-01-25 to 2024-01-15):

| Column | Description |
|---|---|
| `Channel0_impression` ... `Channel4_impression` | Paid media impressions |
| `Channel0_spend` ... `Channel4_spend` | Paid media spend |
| `Organic_channel0_impression` | Organic media impressions |
| `Promo` | Promotion intensity (non-media treatment) |
| `competitor_sales_control`, `sentiment_score_control` | Control variables |
| `conversions` | KPI |
| `revenue_per_conversion` | Converts conversions into revenue |
| `population` | Region population |
