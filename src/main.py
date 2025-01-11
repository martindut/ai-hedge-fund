from pathlib import Path
from dotenv import load_dotenv
import os
import yaml

# Load the global .env file first, then the local one if it exists
home = str(Path.home())
load_dotenv(f"{home}/.env")
load_dotenv(Path(__file__).parent.parent / ".env")  # Load local .env if it exists

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Back, Style, init
import questionary

from agents.fundamentals import fundamentals_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.technicals import technical_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.sentiment import sentiment_agent
from graph.state import AgentState
from agents.valuation import valuation_agent
from utils.display import print_trading_output

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tabulate import tabulate

init(autoreset=True)


def parse_hedge_fund_response(response):
    import json
    try:
        return json.loads(response)
    except:
        print(f"Error parsing response: {response}")
        return None

##### Run the Hedge Fund #####
def run_hedge_fund(
    ticker: str,
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list = None,
):
    # Create a new workflow if analysts are customized
    if selected_analysts is not None:
        workflow = create_workflow(selected_analysts)
        agent = workflow.compile()
    else:
        agent = app

    final_state = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.",
                )
            ],
            "data": {
                "ticker": ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "analyst_signals": {},
            },
            "metadata": {
                "show_reasoning": show_reasoning,
            },
        },
    )
    return {
        "decision": parse_hedge_fund_response(final_state["messages"][-1].content),
        "analyst_signals": final_state["data"]["analyst_signals"],
    }


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)
    
    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = ["technical_analyst", "fundamentals_analyst", "sentiment_analyst", "valuation_analyst"]
    
    # Dictionary of all available analysts
    analyst_nodes = {
        "technical_analyst": ("technical_analyst_agent", technical_analyst_agent),
        "fundamentals_analyst": ("fundamentals_agent", fundamentals_agent),
        "sentiment_analyst": ("sentiment_agent", sentiment_agent),
        "valuation_analyst": ("valuation_agent", valuation_agent),
    }
    
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)
    
    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)
    
    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")
    
    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)
    
    workflow.set_entry_point("start_node")
    return workflow

# # Initialize app as None - it will be set in __main__
# app = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tickers", type=str, nargs='+', help="One or more stock ticker symbols")
    group.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 3 months before end date",
    )
    parser.add_argument(
        "--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today"
    )
    parser.add_argument(
        "--show-reasoning", action="store_true", help="Show reasoning from each agent"
    )

    args = parser.parse_args()

    # Load configuration from YAML if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        positions = config.get('positions', {})
        tickers = list(positions.keys())
        base_portfolio = {
            "cash": config.get('portfolio', {}).get('cash', 100000.0),
        }
    else:
        tickers = args.tickers
        base_portfolio = {
            "cash": 100000.0,
        }
        positions = {ticker: {"stock": 0} for ticker in tickers}

    selected_analysts = None
    choices = questionary.checkbox(
        "Select your AI analysts.",
        choices=[
            questionary.Choice("Technical Analyst", value="technical_analyst"),
            questionary.Choice("Fundamentals Analyst", value="fundamentals_analyst"),
            questionary.Choice("Sentiment Analyst", value="sentiment_analyst"),
            questionary.Choice("Valuation Analyst", value="valuation_analyst"),
        ],
        instruction="\n\nInstructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done to run the hedge fund.\n",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style([
            ('checkbox-selected', 'fg:green'),       
            ('selected', 'fg:green noinherit'),
            ('highlighted', 'noinherit'),  
            ('pointer', 'noinherit'),             
        ])
    ).ask()
    
    if not choices:
        print("You must select at least one analyst. Using all analysts by default.")
        selected_analysts = None
    else:
        selected_analysts = choices
        print(f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}")

    # Create the workflow with selected analysts
    workflow = create_workflow(selected_analysts)
    app = workflow.compile()

    # Validate dates if provided
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM-DD format")

    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("End date must be in YYYY-MM-DD format")

    # Set the start and end dates
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    # Process each ticker
    for ticker in tickers:
        print(f"\n{Fore.CYAN}Analyzing {ticker}{Style.RESET_ALL}")
        print("=" * 50)
        
        # Create portfolio for this ticker
        portfolio = base_portfolio.copy()
        portfolio["stock"] = positions[ticker].get("stock", 0)
        
        # Run the hedge fund
        try:

            result = run_hedge_fund(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                portfolio=portfolio,
                show_reasoning=args.show_reasoning,
                selected_analysts=selected_analysts,
            )
            print_trading_output(result)
        except Exception as e:
            print(f"{Fore.RED}Error processing {ticker}: {e}{Style.RESET_ALL}")