
python app.py
## Quick Start (Immediate)

git clone https://github.com/tim3959951/weather_agent.git

cd weather_agent

pip install -r requirements.txt

# Set API keys (provided separately)
export OPENAI_API_KEY="your_openai_key_here"

export WEATHER_API_KEY="b9b44f4c0e8949bb95a90524250204"

python app.py  # Same experience as HuggingFace demo

## Full ML Experience (30-60 mins)  
pip install -r requirements.txt

python ml_predictor.py  # Train ML models for 13-day forecasting

python app.py          # Complete ML-powered experience
