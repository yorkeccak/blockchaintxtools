# Blockchain Transaction Collector

A comprehensive toolkit for collecting and analyzing blockchain transactions using Blockchair's database dumps.

## Features

- Collect blockchain transactions from Blockchair's data dumps
- Filter transactions based on value, amount, and date range
- Flexible collection intervals: daily, weekly, or random
- Progress saving and resuming capabilities
- Transaction analysis tools
- Export data to CSV and JSON formats

## Installation

You can install the Blockchain Transaction Collector using pip:

```
pip install blockchaintxcollector
```

## Usage

Here's a quick example of how to use the BTC Transaction Collector:

```python
from blockchaintxcollector import BTCTransactionCollector, TransactionFilter, IntervalType
from datetime import datetime

# Initialize the collector
collector = BTCTransactionCollector()

# Define the transaction filter
transaction_filter = TransactionFilter(
    min_usd_value=10000000, # Minimum transaction value in USD
    start_date=datetime(2018, 1, 1),
    end_date=datetime(2024, 6, 26)
)

# Collect transactions
collector.collect_transactions(transaction_filter, interval=IntervalType.WEEKLY)
# Analyze transactions

analysis, transactions = collector.analyze_transactions()
print(analysis)

# Export data
collector.export_analysis_to_json(analysis, "large_transactions_analysis.json")
collector.export_transactions_to_csv(transactions, "large_transactions.csv")
collector.export_transactions_to_json(transactions, "large_transactions.json")
```

## Documentation

For full documentation, please refer to the [docs](https://github.com/yorkeccak/blockchaintxtools/docs) directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.