"""
Demo script for the Blockchain Transaction Tools package.

This script demonstrates how to use the BlockchainTxAnalyzer class
to collect, analyze, and export blockchain transactions.
"""

from blockchaintxtools import BlockchainTxAnalyzer, TransactionFilter, IntervalType
from datetime import datetime
import json

def main():
    # Initialize the analyzer
    analyzer = BlockchainTxAnalyzer(output_file='demo_transactions.csv', checkpoint_file='demo_checkpoint.json')

    # Define the transaction filter
    transaction_filter = TransactionFilter(
        min_usd_value=10000000,  # Minimum transaction value of 10 million USD
        max_usd_value=None,  # No maximum value
        min_btc_amount=0,  # No minimum BTC amount
        max_btc_amount=None,  # No maximum BTC amount
        start_date=datetime(2023, 1, 1),  # Start from January 1, 2023
        end_date=datetime(2023, 1, 7)  # End at January 7, 2023 (for demo purposes)
    )

    # Scenario 1: Collect, analyze, and export
    print("Scenario 1: Collect, analyze, and export")
    print("Collecting transactions...")
    analyzer.collect_transactions(transaction_filter, interval=IntervalType.DAILY)

    print("\nAnalyzing collected transactions...")
    analysis, transactions = analyzer.analyze_transactions()
    print(json.dumps(analysis, indent=2))

    print("\nExporting collected data...")
    analyzer.export_analysis_to_json(analysis, "demo_analysis_collected.json")
    analyzer.export_transactions_to_csv(transactions, "demo_transactions_collected.csv")
    analyzer.export_transactions_to_json(transactions, "demo_transactions_collected.json")

    # Scenario 2: Analyze and export from CSV
    print("\nScenario 2: Analyze and export from CSV")
    print("Analyzing transactions from CSV...")
    csv_file = "demo_transactions.csv"  # This file should exist from the previous collection
    analysis, transactions = analyzer.analyze_transactions(csv_file=csv_file)
    print(json.dumps(analysis, indent=2))

    print("\nExporting data from CSV analysis...")
    analyzer.export_analysis_to_json(analysis, "demo_analysis_from_csv.json")
    analyzer.export_transactions_to_csv(transactions, "demo_transactions_from_csv.csv")
    analyzer.export_transactions_to_json(transactions, "demo_transactions_from_csv.json")

    print("\nDemo completed. Check the output files for results.")

if __name__ == "__main__":
    main()