import gradio as gr
import os
import pandas as pd
import matplotlib.pyplot as plt
from Smart_Weather_Forecasting_Agent import weather_agent_tool

def process_weather_query(query):
    """Process weather queries and return results"""
    try:
        # Clear old chart files
        old_chart_files = ["/tmp/weather_chart.png", "weather_chart.png", "./weather_chart.png"]
        for path in old_chart_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Removed old chart: {path}")
                except:
                    pass
        
        result = weather_agent_tool(query)
        
        # Check for generated charts
        chart_path = None
        chart_files = ["/tmp/weather_chart.png", "weather_chart.png", "./weather_chart.png"]
        
        for path in chart_files:
            if os.path.exists(path):
                chart_path = path
                print(f"New chart found at: {path}")
                break
        
        # Extract table data if present
        table_content = ""
        if "Here is your table:" in result:
            table_start = result.find("Here is your table:") + len("Here is your table:")
            table_content = result[table_start:].strip()
            result = result[:result.find("Here is your table:")].strip()
        
        return result, chart_path, table_content
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nTry simpler queries like 'weather in London tomorrow'"
        return error_msg, None, ""

def create_interface():
    """Create the main interface"""
    
    css = """
    .gradio-container {
        max-width: 1000px !important;
        margin: 0 auto !important;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 8px;
        margin-bottom: 25px;
    }
    .main-header h1 {
        margin: 0 0 10px 0;
        font-size: 2.2em;
        font-weight: 600;
    }
    .main-header p {
        margin: 5px 0;
        font-size: 1.1em;
        opacity: 0.9;
    }
    """
    
    with gr.Blocks(css=css, title="Smart Weather Agent") as demo:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🌤️ Smart Weather Agent</h1>
        </div>
        """)
        
        gr.Markdown("""
        Ask any questions about weather. The system automatically decides when to show charts or detailed data.
        
        ** Full version supports 13 days forecast with ML predictions, please make sure you've run python ml_predictor.py to enable full function.
        """)
        
        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="Your weather question",
                    placeholder="How's the weather in Tokyo tomorrow?",
                    lines=2
                )
            with gr.Column(scale=1):
                submit_btn = gr.Button("Weather Analysis", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", variant="secondary")
        
        with gr.Row():
            gr.Examples(
                examples=[
                    "Provide a past 6 day weather summary for Tokyo.",
                    "Humidity in New York today",
                    "Weather forecast for Paris next 2 days", 
                    "Is it good beach weather in Miami tomorrow?",
                    "Should I bring a jacket to Berlin the day after tomorrow?",
                    "What's the weather like for the next 3 days in Mumbai",
                    "Summarize last 7 days weather for Nottingham in table format.",
                    "What's the temperature in Taipei from 3pm today to 2am tomorrow?",
                    "What was the average temperature in Kaohsiung 5 days ago?",
                    
                        
                ],
                inputs=query_input,
                label="Try these examples"
            )
        
        result_output = gr.Textbox(
            label="Weather Analysis",
            lines=12,
            show_copy_button=True
        )
        
        with gr.Row():
            chart_output = gr.Image(
                label="Weather Charts",
                height=350
            )
            table_output = gr.Textbox(
                label="Detailed Data",
                lines=12,
                show_copy_button=True
            )
        
        clear_btn.click(
            fn=lambda: ("", None, ""),
            outputs=[result_output, chart_output, table_output]
        )
        
        submit_btn.click(
            fn=process_weather_query,
            inputs=[query_input],
            outputs=[result_output, chart_output, table_output]
        )
        
        query_input.submit(
            fn=process_weather_query,
            inputs=[query_input],
            outputs=[result_output, chart_output, table_output]
        )
        
        gr.Markdown("""
        ---
        
        **About this demo:**
        - Covers any location worldwide (any cities)
        - Historical data (7 days back) to 13-day forecasts
        - Automatically generates charts for time-series data
        - For details, check the [GitHub repository](https://github.com/tim3959951/weather_agent)
        """)
    
    return demo

if __name__ == "__main__":
    print("Starting Smart Weather Agent...")
    
    if os.getenv("OPENAI_API_KEY") and os.getenv("WEATHER_API_KEY"):
        print("API keys found")
    else:
        print("Warning: API keys not found")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
