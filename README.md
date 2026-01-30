Oil Market Analytics Dashboard
Overview

This project is a Python-based analytics and visualisation tool for analysing oil options and futures market data.
It focuses on data ingestion, quantitative analytics, and an interactive Streamlit UI for exploring results.

The repository contains only code and configuration.
Raw market data, PDFs, and databases are excluded from version control.

Data Ingestion

The system ingests historical market data from multiple sources:

CSV files containing processed market data

Structured PDF reports from the Intercontinental Exchange (ICE), converted to tabular form for analysis

All processed data is normalised and stored in a local SQLite database to support efficient querying and downstream analytics.

Analytics & Feature Engineering

The analytics layer is implemented in Python using NumPy and Pandas and includes:

Black–Scholes option pricing

Implied volatility calculations

Time-series feature engineering:

returns

rolling statistics

volatility measures

These features are used to analyse market behaviour and support strategy evaluation.

Visualisation & UI

A lightweight Streamlit application provides an interactive interface to:

Explore processed market data

Visualise option pricing outputs and derived features

Inspect strategy-level performance and analytics results

Additional plots and diagnostics are generated using Matplotlib.

Project Structure
oil-trading/
├── src/                # Core analytics and processing code
├── ui/                 # Streamlit application
├── scripts/            # Data ingestion and parsing scripts
├── README.md
├── requirements.txt / environment.yml
└── .gitignore

Technologies Used

Python

NumPy

Pandas

SQLite

Streamlit

Matplotlib

Notes

Raw market data, exchange reports, and databases are intentionally excluded from the repository.

The project is designed for local analysis and experimentation rather than cloud deployment.
