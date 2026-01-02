docker-compose up -d
source venv/bin/activate
uvicorn api:app --reload --host 0.0.0.0 --port 8000
streamlit run app.py
