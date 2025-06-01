# Smart Weather Forecasting AI Agent

## Live Demo
**Try it now**: [https://huggingface.co/spaces/your-username/weather-agent](https://huggingface.co/spaces/ChienChung/weather_agent)

No installation required. Click the link above to test immediately.

---

## Local Installation Instructions

### Before You Start
You need:
- Python 3.8+ ([download here](https://www.python.org/downloads/))
- Git ([download here](https://git-scm.com/downloads))
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Step 1: Download the Code
```bash
git clone https://github.com/tim3959951/weather_agent.git
cd weather_agent
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set API Keys

**For Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your_openai_key_here
set WEATHER_API_KEY=b9b44f4c0e8949bb95a90524250204
```

**For Mac/Linux (Terminal):**
```bash
export OPENAI_API_KEY="your_openai_key_here"
export WEATHER_API_KEY="b9b44f4c0e8949bb95a90524250204"
```

### Step 4: Choose Your Experience

## Quick Start (5 minutes)
```bash
python app.py
```
- Uses pre-trained models
- Supports 7 days historical + 3 days forecast


Access: http://localhost:7860

## Full ML Experience (60+ minutes)
```bash
# First, train the ML models (this takes 45-60 minutes)
python ml_predictor.py

# Then run the application
python app.py
```
- Trains your own XGBoost models
- Supports 7 days historical + 13 days ML forecast  

Access: http://localhost:7860

### CLI Testing (Command Line Interface)
```bash
# Test directly in terminal/command prompt
python weather_cli.py "weather in Tokyo tomorrow"
python weather_cli.py "3-day forecast for London"
python weather_cli.py "will it rain in Paris tomorrow at 5pm?"
```


---

## Troubleshooting

**"API key not found" error:**
- Make sure you completed Step 3 correctly
- Close and reopen your terminal/command prompt
- Try setting the API keys again

**"Module not found" error:**
- Make sure you ran `pip install -r requirements.txt`
- Check you're in the weather_agent folder (`cd weather_agent`)

**"python: command not found" error:**
- Python is not installed or not in your PATH
- Download Python from the link above and try again

**Application won't start:**
- Check if something else is using port 7860
- Try closing other applications and run `python app.py` again

**Still having problems?**
- Check Python version: `python --version` (should show 3.8 or higher)
- Check if you're in the right folder: `ls` (Mac/Linux) or `dir` (Windows) should show app.py

---

## What This Agent Can Do

- **Global coverage**: Works for any city worldwide
- **Smart time understanding**: "tomorrow at 3pm", "next week", "past 5 days" 
- **Multiple output formats**: Text summaries, data tables, and charts
- **Machine learning predictions**: Advanced forecasting models
- **Real-time data**: Live weather API integration

### Example queries to try:
- "What's the weather like for the next 3 days in Tokyo?"
- "Show me the temperature forecast for this week in London as a chart"
- "Summarize last week's weather in New York in table format"
- "Will it rain in Paris tomorrow at 5pm?"

---

## Sample Outputs

**Text response:**
```
Weather forecast for Tokyo tomorrow:
It is very likely that tomorrow will be partly cloudy with temperatures 
reaching 22°C. There is a moderate chance of light rain in the afternoon.
```

**Data table:**
```
Time           | Temp(°C) | Humidity | Condition   
2024-06-03 09  | 18.5     | 65%      | Sunny       
2024-06-03 12  | 22.1     | 58%      | Partly cloudy
2024-06-03 15  | 24.3     | 62%      | Cloudy      
```

**Charts:** The system automatically generates visual charts when appropriate.
