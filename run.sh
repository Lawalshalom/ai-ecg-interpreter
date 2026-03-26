#!/bin/bash
# Launch the ECG Interpreter Streamlit app
cd "$(dirname "$0")"
streamlit run app/app.py --server.port 8501 --server.headless false
