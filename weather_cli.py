#!/usr/bin/env python3
"""
CLI interface for Smart Weather Forecasting Agent
Usage: python weather_cli.py "your weather query"
"""

import sys
import os
from Smart_Weather_Forecasting_Agent import weather_agent_tool

def main():
    if len(sys.argv) < 2:
        print("Smart Weather Forecasting Agent - CLI Interface")
        print("=" * 50)
        print("Usage: python weather_cli.py 'your weather query'")
        print("\nExample queries:")
        print('  python weather_cli.py "weather in Tokyo tomorrow"')
        print('  python weather_cli.py "3-day forecast for London"')
        print('  python weather_cli.py "temperature in New York today"')
        print('  python weather_cli.py "will it rain in Paris tomorrow at 5pm?"')
        print("\nNote: Charts will be saved as files, tables will be displayed in text format.")
        return
    
    # Get the query from command line arguments
    query = " ".join(sys.argv[1:])
    
    print("Smart Weather Forecasting Agent")
    print("=" * 50)
    print(f"Query: {query}")
    print("-" * 50)
    
    try:
        # Check if API keys are set
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable not set")
            print("Please set it using:")
            print("  export OPENAI_API_KEY='your_key'  (Mac/Linux)")
            print("  set OPENAI_API_KEY=your_key       (Windows)")
            return
            
        if not os.getenv("WEATHER_API_KEY"):
            print("Error: WEATHER_API_KEY environment variable not set")
            print("Please set it using:")
            print("  export WEATHER_API_KEY='your_key'  (Mac/Linux)")  
            print("  set WEATHER_API_KEY=your_key       (Windows)")
            return
        
        # Call the weather agent
        result = weather_agent_tool(query)
        print(result)
        
        # Check if charts were generated
        chart_files = ["/tmp/weather_chart.png", "weather_chart.png", "./weather_chart.png"]
        for chart_path in chart_files:
            if os.path.exists(chart_path):
                print(f"\n📊 Chart saved to: {chart_path}")
                break
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure API keys are set correctly")
        print("2. Check your internet connection")
        print("3. Try a simpler query like 'weather in London today'")

if __name__ == "__main__":
    main()
