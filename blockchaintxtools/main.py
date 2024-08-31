"""
BTC Transaction Collector

A comprehensive library for collecting and analyzing Bitcoin transactions using Blockchair's database dumps.
"""

import requests
import pandas as pd
import gzip
from io import BytesIO
from datetime import datetime, timedelta
import signal
import sys
import json
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, Tuple, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ValidationInfo
import random
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    filename='bitcoin_transactions.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class IntervalType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    RANDOM = "random"

class TransactionFilter(BaseModel):
    min_usd_value: float = Field(0, description="Minimum USD value of the transaction")
    max_usd_value: Optional[float] = Field(None, description="Maximum USD value of the transaction")
    min_btc_amount: float = Field(0, description="Minimum BTC amount of the transaction")
    max_btc_amount: Optional[float] = Field(None, description="Maximum BTC amount of the transaction")
    start_date: datetime = Field(..., description="Start date for the date range")
    end_date: datetime = Field(..., description="End date for the date range")

    @field_validator('end_date')
    def end_date_must_be_after_start_date(cls, v: datetime, info: ValidationInfo) -> datetime:
        if 'start_date' in info.data and v <= info.data['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

class BlockchainTxAnalyzer:
    """
    A class for collecting, analyzing, and exporting blockchain transactions using Blockchair's database dumps.

    This class provides functionality to download, process, analyze, and export blockchain transaction data
    from Blockchair's database dumps. It supports filtering transactions, saving progress, and exporting
    results in various formats.

    Attributes:
        output_file (str): The file path for saving collected transactions.
        checkpoint_file (str): The file path for saving checkpoint information.
        all_transactions (pd.DataFrame): A DataFrame containing all collected transactions.
        checkpoint (dict): A dictionary containing checkpoint information.
        base_url (str): The base URL for Blockchair's database dumps.
    """

    def __init__(self, output_file: str = 'blockchain_transactions.csv', 
                 checkpoint_file: str = 'checkpoint.json',
                 api_url: str = None):
        """
        Initialize the BlockchainTxAnalyzer.

        Args:
            output_file (str): The file path for saving collected transactions.
            checkpoint_file (str): The file path for saving checkpoint information.
            api_url (str): The base URL for Blockchair's database dumps.
        """
        self.output_file = output_file
        self.checkpoint_file = checkpoint_file
        self.all_transactions = pd.DataFrame()
        self.checkpoint = {'last_processed_date': None}
        self.base_url = api_url or os.getenv('BLOCKCHAIR_API_URL', "https://gz.blockchair.com/bitcoin/transactions/")

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """
        Handle interrupt signals by saving progress and exiting.

        Args:
            sig: The signal number.
            frame: The current stack frame.
        """
        logging.info('Interrupt received. Saving progress...')
        print('Interrupt received. Saving progress...')
        self._save_progress()
        sys.exit(0)

    def _save_progress(self):
        """Save the current progress to files."""
        logging.info("Saving current progress...")
        print("Saving current progress...")
        self.all_transactions.to_csv(self.output_file, index=False)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f)
        logging.info("Progress saved.")
        print("Progress saved.")

    def _load_progress(self):
        """Load the previous progress from files."""
        try:
            self.all_transactions = pd.read_csv(self.output_file)
            with open(self.checkpoint_file, 'r') as f:
                self.checkpoint = json.load(f)
            logging.info(f"Loaded {len(self.all_transactions)} transactions. Resuming from {self.checkpoint['last_processed_date']}")
            print(f"Loaded {len(self.all_transactions)} transactions. Resuming from {self.checkpoint['last_processed_date']}")
        except FileNotFoundError:
            logging.info("No previous progress found. Starting from the beginning.")
            print("No previous progress found. Starting from the beginning.")

    def _download_and_process_file(self, url: str, date: str, transaction_filter: TransactionFilter) -> pd.DataFrame:
        """
        Download and process a single file of transaction data from Blockchair's database dump.

        Args:
            url (str): The URL of the database dump file to download.
            date (str): The date of the transactions in the file.
            transaction_filter (TransactionFilter): The filter to apply to the transactions.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered transactions.
        """
        logging.info(f"Downloading data for {date}...")
        print(f"Downloading data for {date}...")
        try:
            response = requests.get(url)
            logging.info(f"Downloaded data for {date}")
            print(f"Downloaded data for {date}")
            if response.status_code == 200:
                with gzip.open(BytesIO(response.content), 'rt') as f:
                    logging.info(f"Processing data for {date}")
                    print(f"Processing data for {date}")
                    df = pd.read_csv(f, sep='\t')
                    return self._filter_transactions(df, transaction_filter)
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error downloading or processing data for {date}: {e}")
            print(f"Error downloading or processing data for {date}: {e}")
            return pd.DataFrame()

    def _filter_transactions(self, df: pd.DataFrame, transaction_filter: TransactionFilter) -> pd.DataFrame:
        """
        Apply filters to a DataFrame of transactions.

        Args:
            df (pd.DataFrame): The DataFrame of transactions to filter.
            transaction_filter (TransactionFilter): The filter to apply.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered transactions.
        """
        mask = (
            (df['output_total_usd'] >= transaction_filter.min_usd_value) &
            (df['output_total'] >= transaction_filter.min_btc_amount)
        )
        if transaction_filter.max_usd_value:
            mask &= (df['output_total_usd'] <= transaction_filter.max_usd_value)
        if transaction_filter.max_btc_amount:
            mask &= (df['output_total'] <= transaction_filter.max_btc_amount)
        return df[mask]

    def _get_next_date(self, current_date: datetime, interval: IntervalType) -> datetime:
        """
        Get the next date based on the specified interval.

        Args:
            current_date (datetime): The current date.
            interval (IntervalType): The type of interval to use.

        Returns:
            datetime: The next date.
        """
        if interval == IntervalType.DAILY:
            return current_date + timedelta(days=1)
        elif interval == IntervalType.WEEKLY:
            return current_date + timedelta(days=7)
        elif interval == IntervalType.RANDOM:
            return current_date + timedelta(days=random.randint(1, 7))

    def collect_transactions(self, transaction_filter: TransactionFilter, interval: IntervalType = IntervalType.WEEKLY, days_per_week: int = 1):
        """
        Collect Bitcoin transactions from Blockchair's database dumps based on the specified filter and interval.

        This method downloads and processes database dump files from Blockchair, which contain
        comprehensive transaction data for entire days. It allows for efficient retrieval and analysis
        of large-scale historical transaction data.

        Args:
            transaction_filter (TransactionFilter): The filter to apply to the transactions.
            interval (IntervalType): The interval type for collection (daily, weekly, or random).
            days_per_week (int): The number of days per week to collect when using random interval.
        """
        self._load_progress()
        
        if self.checkpoint['last_processed_date']:
            current_date = datetime.strptime(self.checkpoint['last_processed_date'], "%Y-%m-%d")
            current_date = self._get_next_date(current_date, interval)
        else:
            current_date = transaction_filter.start_date

        total_days = (transaction_filter.end_date - current_date).days
        
        with tqdm(total=total_days, desc="Processing Days") as pbar:
            while current_date <= transaction_filter.end_date:
                if interval == IntervalType.RANDOM and random.randint(1, 7) > days_per_week:
                    current_date = self._get_next_date(current_date, interval)
                    pbar.update(1)
                    continue

                try:
                    date_str = current_date.strftime("%Y%m%d")
                    file_url = f"{self.base_url}blockchair_bitcoin_transactions_{date_str}.tsv.gz?202001ZjMvj8R3BF"
                    
                    logging.info(f"Processing data for {date_str}")
                    print(f"Processing data for {date_str}")
                    df = self._download_and_process_file(file_url, date_str, transaction_filter)
                    self.all_transactions = pd.concat([self.all_transactions, df], ignore_index=True)
                    logging.info(f"Data processed for {date_str}")
                    print(f"Data processed for {date_str}")
                    
                    self.checkpoint['last_processed_date'] = current_date.strftime("%Y-%m-%d")
                    self._save_progress()
                    
                    pbar.update(1)
                except Exception as e:
                    logging.error(f"Error processing {date_str}: {e}")
                    print(f"Error processing {date_str}: {e}")
                    self._save_progress()
                
                current_date = self._get_next_date(current_date, interval)

        self._save_progress()
        logging.info(f"Processing complete. Total transactions found: {len(self.all_transactions)}")
        print(f"Processing complete. Total transactions found: {len(self.all_transactions)}")

    def get_transactions(self) -> pd.DataFrame:
        """
        Get the DataFrame of all collected transactions.

        Returns:
            pd.DataFrame: A DataFrame containing all collected transactions.
        """
        return self.all_transactions

    def analyze_transactions(self, csv_file: Optional[str] = None) -> Tuple[Dict, pd.DataFrame]:
        """
        Analyze the collected transactions or transactions from a CSV file.

        This method performs analysis on transaction data obtained from Blockchair's database dumps.
        It can analyze either the transactions collected by this class or transactions loaded from a CSV file.

        Args:
            csv_file (Optional[str]): The path to a CSV file containing transactions to analyze.

        Returns:
            Tuple[Dict, pd.DataFrame]: A tuple containing the analysis results and the transactions DataFrame.
        """
        if csv_file:
            try:
                transactions = pd.read_csv(csv_file)
            except FileNotFoundError:
                logging.error(f"CSV file not found: {csv_file}")
                return {}, pd.DataFrame()
            except pd.errors.EmptyDataError:
                logging.error(f"CSV file is empty: {csv_file}")
                return {}, pd.DataFrame()
            except Exception as e:
                logging.error(f"Error reading CSV file: {csv_file}. Error: {str(e)}")
                return {}, pd.DataFrame()
        else:
            transactions = self.all_transactions

        if transactions.empty:
            logging.warning("No transactions to analyze.")
            return {}, pd.DataFrame()

        required_columns = ['output_total', 'output_total_usd', 'fee', 'fee_usd']
        missing_columns = [col for col in required_columns if col not in transactions.columns]
        if missing_columns:
            logging.error(f"Missing required columns in the transaction data: {', '.join(missing_columns)}")
            return {}, pd.DataFrame()

        def convert_to_native(value):
            if isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return float(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            else:
                return value

        analysis = {
            "total_transactions": int(len(transactions)),
            "total_btc_volume": float(transactions['output_total'].sum()),
            "total_usd_volume": float(transactions['output_total_usd'].sum()),
            "average_transaction_size_btc": float(transactions['output_total'].mean()),
            "average_transaction_size_usd": float(transactions['output_total_usd'].mean()),
            "largest_transaction_btc": float(transactions['output_total'].max()),
            "largest_transaction_usd": float(transactions['output_total_usd'].max()),
            "total_fees_btc": float(transactions['fee'].sum()),
            "total_fees_usd": float(transactions['fee_usd'].sum()),
            "average_fee_btc": float(transactions['fee'].mean()),
            "average_fee_usd": float(transactions['fee_usd'].mean()),
        }

        analysis = {k: convert_to_native(v) for k, v in analysis.items()}
        return analysis, transactions

    def export_analysis_to_json(self, analysis: Dict, filename: str):
        """
        Export the analysis results to a JSON file.

        Args:
            analysis (Dict): The analysis results to export.
            filename (str): The name of the file to export to.
        """
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis exported to {filename}")

    def export_transactions_to_csv(self, transactions: pd.DataFrame, filename: str):
        """
        Export the transactions to a CSV file.

        Args:
            transactions (pd.DataFrame): The transactions to export.
            filename (str): The name of the file to export to.
        """
        if transactions.empty:
            logging.warning(f"No transactions to export to {filename}")
            print(f"No transactions to export to {filename}")
            return
        
        transactions.to_csv(filename, index=False)
        print(f"Transactions exported to {filename}")

    def export_transactions_to_json(self, transactions: pd.DataFrame, filename: str):
        """
        Export the transactions to a JSON file.

        Args:
            transactions (pd.DataFrame): The transactions to export.
            filename (str): The name of the file to export to.
        """
        if transactions.empty:
            logging.warning(f"No transactions to export to {filename}")
            print(f"No transactions to export to {filename}")
            return
        
        transactions.to_json(filename, orient='records')
        print(f"Transactions exported to {filename}")