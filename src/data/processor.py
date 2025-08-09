"""
Data Processing Module

Advanced data processing pipeline for financial domain LLM training.
Handles SPGISpeech dataset preprocessing, validation, and instruction formatting.
"""

import pandas as pd
import numpy as np
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer
from rich.console import Console
from rich.progress import Progress, TaskID
import random

from ..core.config import DataConfig
from ..utils.validators import DataValidator
from ..utils.logger import get_logger

console = Console()
logger = get_logger(__name__)

@dataclass
class DataStatistics:
    """Data statistics container"""
    total_samples: int
    avg_transcript_length: float
    max_transcript_length: int
    min_transcript_length: int
    vocab_size: int
    duplicate_count: int
    empty_count: int
    
    def __str__(self) -> str:
        return f"""Data Statistics:
  Total samples: {self.total_samples:,}
  Average transcript length: {self.avg_transcript_length:.1f} words
  Max transcript length: {self.max_transcript_length:,} words
  Min transcript length: {self.min_transcript_length:,} words
  Vocabulary size: {self.vocab_size:,}
  Duplicates: {self.duplicate_count:,}
  Empty transcripts: {self.empty_count:,}"""

class InstructionTemplate(ABC):
    """Abstract base class for instruction templates"""
    
    @abstractmethod
    def format(self, transcript: str, **kwargs) -> Dict[str, str]:
        """Format transcript into instruction format"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Template name"""
        pass

class FinancialUnderstandingTemplate(InstructionTemplate):
    """Template for financial understanding tasks"""
    
    @property
    def name(self) -> str:
        return "financial_understanding"
    
    def format(self, transcript: str, **kwargs) -> Dict[str, str]:
        return {
            "instruction": "Analyze the following business call transcript and provide key insights about the financial discussion:",
            "input": transcript,
            "output": f"This transcript discusses {self._extract_financial_themes(transcript)}. Key financial points include revenue trends, market positioning, and strategic initiatives.",
            "template_type": self.name
        }
    
    def _extract_financial_themes(self, transcript: str) -> str:
        """Extract main financial themes from transcript"""
        financial_keywords = {
            "revenue": ["revenue", "sales", "income", "earnings"],
            "growth": ["growth", "expansion", "increase", "momentum"],
            "market": ["market", "clients", "customers", "franchise"],
            "investment": ["investment", "capital", "funding", "buyback"],
            "performance": ["performance", "results", "metrics", "outstanding"]
        }
        
        transcript_lower = transcript.lower()
        themes = []
        
        for theme, keywords in financial_keywords.items():
            if any(keyword in transcript_lower for keyword in keywords):
                themes.append(theme)
        
        return ", ".join(themes) if themes else "business operations"

class TranscriptionCompletionTemplate(InstructionTemplate):
    """Template for transcription completion tasks"""
    
    @property
    def name(self) -> str:
        return "transcription_completion"
    
    def format(self, transcript: str, **kwargs) -> Dict[str, str]:
        # Split transcript and use first 70% as input
        words = transcript.split()
        split_point = int(len(words) * 0.7)
        partial_transcript = " ".join(words[:split_point])
        completion = " ".join(words[split_point:])
        
        return {
            "instruction": "Complete this business call transcript:",
            "input": partial_transcript + "...",
            "output": completion,
            "template_type": self.name
        }

class ContentSummarizationTemplate(InstructionTemplate):
    """Template for content summarization tasks"""
    
    @property
    def name(self) -> str:
        return "content_summarization"
    
    def format(self, transcript: str, **kwargs) -> Dict[str, str]:
        summary = self._generate_summary(transcript)
        return {
            "instruction": "Summarize the key points from this financial call excerpt:",
            "input": transcript,
            "output": summary,
            "template_type": self.name
        }
    
    def _generate_summary(self, transcript: str) -> str:
        """Generate a basic extractive summary"""
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Take first and most informative sentences
        key_sentences = sentences[:min(3, len(sentences))]
        return ". ".join(key_sentences) + "."

class QAGenerationTemplate(InstructionTemplate):
    """Template for question-answer generation tasks"""
    
    @property
    def name(self) -> str:
        return "qa_generation"
    
    def format(self, transcript: str, **kwargs) -> Dict[str, str]:
        questions = self._generate_questions(transcript)
        return {
            "instruction": "Based on this financial call transcript, what questions might investors ask?",
            "input": transcript,
            "output": "\n".join(questions),
            "template_type": self.name
        }
    
    def _generate_questions(self, transcript: str) -> List[str]:
        """Generate relevant questions based on transcript content"""
        questions = [
            "What are the key financial metrics mentioned?",
            "What growth strategies are being discussed?",
            "How is the company positioning itself in the market?",
            "What are the main revenue drivers mentioned?"
        ]
        return questions[:2]  # Return subset for brevity

class FinancialDataProcessor:
    """
    Advanced data processor for financial domain LLM training.
    
    Features:
    - Comprehensive data validation and cleaning
    - Multiple instruction template support
    - Statistical analysis and quality metrics
    - Memory-efficient processing for large datasets
    - Robust error handling and logging
    """
    
    def __init__(self, config: DataConfig, tokenizer: Optional[PreTrainedTokenizer] = None):
        """
        Initialize data processor.
        
        Args:
            config: Data configuration
            tokenizer: Optional tokenizer for length validation
        """
        self.config = config
        self.tokenizer = tokenizer
        self.validator = DataValidator()
        
        # Initialize instruction templates
        self.templates = {
            "financial_understanding": FinancialUnderstandingTemplate(),
            "transcription_completion": TranscriptionCompletionTemplate(),
            "content_summarization": ContentSummarizationTemplate(),
            "qa_generation": QAGenerationTemplate()
        }
        
        # Set random seed for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        logger.info(f"Initialized FinancialDataProcessor with {len(self.templates)} templates")
    
    def set_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        """Attach or update the tokenizer after initialization."""
        self.tokenizer = tokenizer
    
    def process_dataset(self, train_path: str, val_path: str) -> Tuple[DatasetDict, DataStatistics]:
        """
        Process complete dataset from CSV files.
        
        Args:
            train_path: Path to training CSV
            val_path: Path to validation CSV
            
        Returns:
            Tuple of (processed_datasets, statistics)
        """
        logger.info("Starting dataset processing...")
        
        # Load and validate raw data
        train_df = self._load_csv_data(train_path, "training")
        val_df = self._load_csv_data(val_path, "validation")
        
        # Preprocess data
        train_df = self._preprocess_dataframe(train_df)
        val_df = self._preprocess_dataframe(val_df)
        
        # Generate statistics
        stats = self._compute_statistics(train_df, val_df)
        
        # Apply sampling if configured
        if self.config.max_samples:
            train_df = train_df.sample(min(self.config.max_samples, len(train_df)), 
                                     random_state=self.config.random_seed)
            logger.info(f"Sampled {len(train_df)} training samples")
        
        # Convert to instruction format
        train_instructions = self._convert_to_instructions(train_df)
        val_instructions = self._convert_to_instructions(val_df)
        
        # Create HuggingFace datasets
        datasets = DatasetDict({
            'train': Dataset.from_list(train_instructions),
            'validation': Dataset.from_list(val_instructions)
        })
        
        logger.info(f"Dataset processing complete. Train: {len(datasets['train'])}, Val: {len(datasets['validation'])}")
        
        return datasets, stats
    
    def _load_csv_data(self, file_path: str, dataset_type: str) -> pd.DataFrame:
        """Load and validate CSV data."""
        try:
            logger.info(f"Loading {dataset_type} data from: {file_path}")
            
            # Handle different file formats
            if file_path.endswith('.bz2'):
                df = pd.read_csv(file_path, sep='|', compression='bz2')
            else:
                # Try different separators
                try:
                    df = pd.read_csv(file_path, sep='|')
                except:
                    df = pd.read_csv(file_path, sep=',')
            
            logger.info(f"Loaded {len(df)} rows from {dataset_type} dataset")
            
            # Validate required columns
            required_columns = ['transcript']
            if 'wav_filename' in df.columns:
                # Original SPGISpeech format
                df = df.rename(columns={'wav_filename': 'filename'})
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {dataset_type} data: {e}")
            raise
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess DataFrame with cleaning and validation."""
        original_size = len(df)
        
        # Remove empty transcripts
        df = df.dropna(subset=['transcript'])
        df = df[df['transcript'].str.strip() != '']
        
        # Apply text preprocessing
        if self.config.normalize_whitespace:
            df['transcript'] = df['transcript'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        if self.config.lowercase:
            df['transcript'] = df['transcript'].str.lower()
        
        if self.config.remove_special_chars:
            df['transcript'] = df['transcript'].str.replace(r'[^\w\s]', '', regex=True)
        
        # Filter by length
        word_counts = df['transcript'].str.split().str.len()
        df = df[
            (word_counts >= self.config.min_transcript_length) & 
            (word_counts <= self.config.max_transcript_length)
        ]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['transcript'])
        
        removed_count = original_size - len(df)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} samples during preprocessing")
        
        return df.reset_index(drop=True)
    
    def _compute_statistics(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> DataStatistics:
        """Compute comprehensive data statistics."""
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        
        # Word-level statistics
        word_counts = combined_df['transcript'].str.split().str.len()
        
        # Vocabulary analysis
        all_words = ' '.join(combined_df['transcript']).split()
        vocab_size = len(set(all_words))
        
        # Quality metrics
        duplicates = combined_df['transcript'].duplicated().sum()
        empty_count = combined_df['transcript'].isna().sum()
        
        stats = DataStatistics(
            total_samples=len(combined_df),
            avg_transcript_length=word_counts.mean(),
            max_transcript_length=word_counts.max(),
            min_transcript_length=word_counts.min(),
            vocab_size=vocab_size,
            duplicate_count=duplicates,
            empty_count=empty_count
        )
        
        console.print(f"\n[bold green]Dataset Statistics[/bold green]")
        console.print(str(stats))
        
        return stats
    
    def _convert_to_instructions(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Convert DataFrame to instruction format using multiple templates."""
        instructions = []
        
        # Default template weights (can be overridden by config)
        template_weights = {
            "financial_understanding": 0.4,
            "transcription_completion": 0.3,
            "content_summarization": 0.2,
            "qa_generation": 0.1
        }
        
        with Progress() as progress:
            task = progress.add_task("Converting to instructions...", total=len(df))
            
            for _, row in df.iterrows():
                transcript = row['transcript']
                
                # Select template based on weights
                template_name = np.random.choice(
                    list(template_weights.keys()),
                    p=list(template_weights.values())
                )
                
                template = self.templates[template_name]
                instruction_data = template.format(transcript)
                
                # Add metadata
                instruction_data.update({
                    'original_filename': row.get('filename', ''),
                    'transcript_length': len(transcript.split()),
                    'preprocessing_applied': True
                })
                
                instructions.append(instruction_data)
                progress.advance(task)
        
        return instructions
    
    def tokenize_instructions(
        self,
        datasets: DatasetDict,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: Optional[int] = None,
    ) -> DatasetDict:
        """Tokenize instruction datasets for training.

        Args:
            datasets: Instruction-formatted datasets to tokenize
            tokenizer: Optional tokenizer override
            max_length: Optional maximum token length to enforce
        """
        tk = tokenizer or self.tokenizer
        if tk is None:
            raise ValueError("Tokenizer required for tokenization")
        
        def tokenize_function(examples):
            # Format prompt
            prompts = []
            for i in range(len(examples['instruction'])):
                if examples['input'][i]:
                    prompt = f"{examples['instruction'][i]}\n\n{examples['input'][i]}\n\nResponse: {examples['output'][i]}"
                else:
                    prompt = f"{examples['instruction'][i]}\n\nResponse: {examples['output'][i]}"
                prompts.append(prompt)
            
            # Tokenize
            # Determine an effective max length. Some models expose extremely large
            # defaults (e.g., 1e30). Prefer an explicit override when provided.
            reported_max_len = int(getattr(tk, "model_max_length", 2048))
            effective_max_len = int(max_length) if max_length is not None else min(reported_max_len, 8192)

            tokenized = tk(
                prompts,
                truncation=True,
                padding=False,
                max_length=effective_max_len,
                return_overflowing_tokens=False,
            )
            
            # Set labels for causal LM
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        # Tokenize datasets
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=datasets['train'].column_names,
            desc="Tokenizing datasets"
        )
        
        logger.info("Tokenization complete")
        return tokenized_datasets
    
    def save_processed_data(self, datasets: DatasetDict, output_dir: str) -> None:
        """Save processed datasets to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        datasets.save_to_disk(output_path / "processed_datasets")
        
        # Save sample data for inspection
        sample_data = {
            'train_sample': datasets['train'][:5],
            'validation_sample': datasets['validation'][:5]
        }
        
        with open(output_path / "dataset_samples.json", 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Processed data saved to {output_path}")
    
    def create_data_collator(self, tokenizer: Optional[PreTrainedTokenizer] = None):
        """Create appropriate data collator for the model."""
        from transformers import DataCollatorForLanguageModeling
        
        tk = tokenizer or self.tokenizer
        if tk is None:
            raise ValueError("Tokenizer required for data collator")
        
        return DataCollatorForLanguageModeling(
            tokenizer=tk,
            mlm=False,  # We're doing causal LM, not masked LM
        )