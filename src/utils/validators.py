"""
Data Validation Module

Comprehensive validation utilities for data quality assurance,
schema validation, and input sanitization.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    
    def add_error(self, error: str):
        """Add an error to the result"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning to the result"""
        self.warnings.append(warning)
    
    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        result = f"Validation Result: {status}\n"
        
        if self.errors:
            result += f"Errors ({len(self.errors)}):\n"
            for error in self.errors:
                result += f"  - {error}\n"
        
        if self.warnings:
            result += f"Warnings ({len(self.warnings)}):\n"
            for warning in self.warnings:
                result += f"  - {warning}\n"
        
        return result

class DataQualityValidator:
    """Validator for data quality and consistency checks"""
    
    def __init__(self, 
                 min_transcript_length: int = 5,
                 max_transcript_length: int = 2048,
                 allow_empty: bool = False):
        """
        Initialize data quality validator.
        
        Args:
            min_transcript_length: Minimum words in transcript
            max_transcript_length: Maximum words in transcript
            allow_empty: Whether to allow empty transcripts
        """
        self.min_transcript_length = min_transcript_length
        self.max_transcript_length = max_transcript_length
        self.allow_empty = allow_empty
    
    def validate_transcript(self, transcript: str) -> ValidationResult:
        """
        Validate a single transcript.
        
        Args:
            transcript: Transcript text to validate
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(True, [], [], {})
        
        # Basic existence check
        if transcript is None:
            result.add_error("Transcript is None")
            return result
        
        # Convert to string if needed
        transcript = str(transcript).strip()
        
        # Empty check
        if not transcript:
            if not self.allow_empty:
                result.add_error("Transcript is empty")
            else:
                result.add_warning("Transcript is empty")
            return result
        
        # Length validation
        words = transcript.split()
        word_count = len(words)
        
        if word_count < self.min_transcript_length:
            result.add_error(
                f"Transcript too short: {word_count} words "
                f"(minimum: {self.min_transcript_length})"
            )
        
        if word_count > self.max_transcript_length:
            result.add_error(
                f"Transcript too long: {word_count} words "
                f"(maximum: {self.max_transcript_length})"
            )
        
        # Character validation
        char_count = len(transcript)
        result.metadata['word_count'] = word_count
        result.metadata['char_count'] = char_count
        result.metadata['avg_word_length'] = char_count / word_count if word_count > 0 else 0
        
        # Content quality checks
        self._validate_content_quality(transcript, result)
        
        return result
    
    def _validate_content_quality(self, transcript: str, result: ValidationResult):
        """Validate transcript content quality"""
        
        # Check for excessive repetition
        words = transcript.lower().split()
        if len(words) > 10:
            # Check for repeated phrases
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            if len(set(bigrams)) < len(bigrams) * 0.7:  # Less than 70% unique bigrams
                result.add_warning("High repetition detected in transcript")
        
        # Check for suspicious patterns
        if re.search(r'(.)\1{10,}', transcript):  # 10+ repeated characters
            result.add_warning("Excessive character repetition detected")
        
        # Check for balanced punctuation
        open_parens = transcript.count('(')
        close_parens = transcript.count(')')
        if abs(open_parens - close_parens) > 0:
            result.add_warning("Unbalanced parentheses detected")
        
        # Check for reasonable capitalization
        if transcript.isupper():
            result.add_warning("Transcript is all uppercase")
        elif transcript.islower() and len(transcript) > 50:
            result.add_warning("Transcript has no capitalization")
        
        # Check encoding issues
        if '�' in transcript:
            result.add_warning("Possible encoding issues detected")

class SchemaValidator:
    """Validator for data schema and structure"""
    
    def __init__(self, required_columns: List[str], optional_columns: List[str] = None):
        """
        Initialize schema validator.
        
        Args:
            required_columns: List of required column names
            optional_columns: List of optional column names
        """
        self.required_columns = required_columns
        self.optional_columns = optional_columns or []
    
    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate DataFrame schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(True, [], [], {})
        
        # Check if DataFrame exists and is not empty
        if df is None:
            result.add_error("DataFrame is None")
            return result
        
        if df.empty:
            result.add_error("DataFrame is empty")
            return result
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            result.add_error(f"Missing required columns: {list(missing_columns)}")
        
        # Check for unexpected columns
        expected_columns = set(self.required_columns + self.optional_columns)
        unexpected_columns = set(df.columns) - expected_columns
        if unexpected_columns:
            result.add_warning(f"Unexpected columns found: {list(unexpected_columns)}")
        
        # Store metadata
        result.metadata.update({
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict()
        })
        
        return result
    
    def validate_csv_file(self, file_path: Union[str, Path]) -> ValidationResult:
        """
        Validate CSV file structure.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(True, [], [], {})
        file_path = Path(file_path)
        
        # Check file existence
        if not file_path.exists():
            result.add_error(f"File does not exist: {file_path}")
            return result
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        result.metadata['file_size_mb'] = file_size_mb
        
        if file_size_mb == 0:
            result.add_error("File is empty")
            return result
        
        if file_size_mb > 1000:  # 1GB
            result.add_warning(f"Large file size: {file_size_mb:.1f}MB")
        
        # Try to read file header
        try:
            # Handle different separators and compression
            if file_path.suffix == '.bz2':
                sample_df = pd.read_csv(file_path, nrows=5, sep='|', compression='bz2')
            else:
                # Try common separators
                for sep in ['|', ',', '\t']:
                    try:
                        sample_df = pd.read_csv(file_path, nrows=5, sep=sep)
                        break
                    except:
                        continue
                else:
                    result.add_error("Could not determine CSV separator")
                    return result
            
            # Validate schema
            schema_result = self.validate_dataframe(sample_df)
            result.errors.extend(schema_result.errors)
            result.warnings.extend(schema_result.warnings)
            result.is_valid = result.is_valid and schema_result.is_valid
            
        except Exception as e:
            result.add_error(f"Failed to read CSV file: {str(e)}")
        
        return result

class FinancialContentValidator:
    """Validator for financial domain-specific content"""
    
    def __init__(self):
        """Initialize financial content validator"""
        # Financial keywords and patterns
        self.financial_keywords = {
            'revenue', 'sales', 'income', 'earnings', 'profit', 'loss',
            'investment', 'capital', 'funding', 'buyback', 'dividend',
            'market', 'growth', 'expansion', 'performance', 'metrics',
            'clients', 'customers', 'franchise', 'outstanding', 'shares',
            'equity', 'debt', 'assets', 'liabilities', 'cash', 'flow',
            'margin', 'ebitda', 'roi', 'returns', 'valuation'
        }
        
        # Common financial number patterns
        self.number_patterns = [
            r'\$[\d,]+\.?\d*[KMB]?',  # Dollar amounts
            r'\d+\.?\d*%',            # Percentages
            r'\d+\.?\d*[KMB]',        # Large numbers with K/M/B
            r'\d{4}',                 # Years
        ]
    
    def validate_financial_content(self, transcript: str) -> ValidationResult:
        """
        Validate financial content relevance.
        
        Args:
            transcript: Transcript to validate
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(True, [], [], {})
        
        if not transcript:
            result.add_error("Empty transcript")
            return result
        
        transcript_lower = transcript.lower()
        
        # Check for financial keywords
        found_keywords = [kw for kw in self.financial_keywords if kw in transcript_lower]
        keyword_ratio = len(found_keywords) / len(self.financial_keywords)
        
        result.metadata['financial_keywords_found'] = found_keywords
        result.metadata['financial_keyword_ratio'] = keyword_ratio
        
        if keyword_ratio < 0.02:  # Less than 2% of financial keywords found
            result.add_warning("Low financial content detected")
        
        # Check for financial numbers
        found_numbers = []
        for pattern in self.number_patterns:
            matches = re.findall(pattern, transcript)
            found_numbers.extend(matches)
        
        result.metadata['financial_numbers_found'] = found_numbers
        result.metadata['financial_numbers_count'] = len(found_numbers)
        
        if len(found_numbers) == 0:
            result.add_warning("No financial numbers detected")
        
        # Check for business call characteristics
        call_indicators = [
            'call', 'conference', 'earnings', 'quarterly', 'annual',
            'management', 'analyst', 'investor', 'question', 'answer'
        ]
        
        found_indicators = [ind for ind in call_indicators if ind in transcript_lower]
        result.metadata['call_indicators_found'] = found_indicators
        
        if not found_indicators:
            result.add_warning("No business call indicators detected")
        
        return result

class DataValidator:
    """Main data validator orchestrating all validation types"""
    
    def __init__(self, 
                 min_transcript_length: int = 5,
                 max_transcript_length: int = 2048,
                 required_columns: List[str] = None):
        """
        Initialize comprehensive data validator.
        
        Args:
            min_transcript_length: Minimum words in transcript
            max_transcript_length: Maximum words in transcript
            required_columns: Required columns for DataFrame validation
        """
        self.quality_validator = DataQualityValidator(
            min_transcript_length=min_transcript_length,
            max_transcript_length=max_transcript_length
        )
        
        self.schema_validator = SchemaValidator(
            required_columns=required_columns or ['transcript']
        )
        
        self.financial_validator = FinancialContentValidator()
    
    def validate_dataset(self, 
                        df: pd.DataFrame,
                        sample_fraction: float = 0.1) -> ValidationResult:
        """
        Comprehensive dataset validation.
        
        Args:
            df: DataFrame to validate
            sample_fraction: Fraction of data to sample for content validation
            
        Returns:
            ValidationResult
        """
        logger.info(f"Starting comprehensive validation of {len(df)} samples")
        
        overall_result = ValidationResult(True, [], [], {})
        
        # 1. Schema validation
        schema_result = self.schema_validator.validate_dataframe(df)
        overall_result.errors.extend(schema_result.errors)
        overall_result.warnings.extend(schema_result.warnings)
        overall_result.is_valid = overall_result.is_valid and schema_result.is_valid
        overall_result.metadata['schema_validation'] = schema_result.metadata
        
        if not schema_result.is_valid:
            logger.error("Schema validation failed")
            return overall_result
        
        # 2. Sample content validation
        sample_size = max(100, int(len(df) * sample_fraction))
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        content_results = []
        transcript_column = 'transcript'
        
        for idx, row in sample_df.iterrows():
            transcript = row[transcript_column]
            
            # Quality validation
            quality_result = self.quality_validator.validate_transcript(transcript)
            
            # Financial content validation
            financial_result = self.financial_validator.validate_financial_content(transcript)
            
            # Combine results
            combined_result = ValidationResult(
                is_valid=quality_result.is_valid and financial_result.is_valid,
                errors=quality_result.errors + financial_result.errors,
                warnings=quality_result.warnings + financial_result.warnings,
                metadata={
                    'index': idx,
                    'quality': quality_result.metadata,
                    'financial': financial_result.metadata
                }
            )
            
            content_results.append(combined_result)
        
        # 3. Aggregate content validation results
        total_errors = sum(len(r.errors) for r in content_results)
        total_warnings = sum(len(r.warnings) for r in content_results)
        valid_samples = sum(1 for r in content_results if r.is_valid)
        
        error_rate = total_errors / len(content_results) if content_results else 0
        valid_rate = valid_samples / len(content_results) if content_results else 0
        
        overall_result.metadata['content_validation'] = {
            'samples_validated': len(content_results),
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'error_rate': error_rate,
            'valid_rate': valid_rate,
            'sample_results': [r.metadata for r in content_results[:10]]  # First 10 for review
        }
        
        # Add warnings based on aggregated results
        if error_rate > 0.1:  # More than 10% error rate
            overall_result.add_warning(f"High error rate in content validation: {error_rate:.1%}")
        
        if valid_rate < 0.8:  # Less than 80% valid samples
            overall_result.add_warning(f"Low validity rate: {valid_rate:.1%}")
        
        # 4. Statistical validation
        self._validate_statistics(df, overall_result)
        
        logger.info(f"Validation completed. Valid: {overall_result.is_valid}, "
                   f"Errors: {len(overall_result.errors)}, "
                   f"Warnings: {len(overall_result.warnings)}")
        
        return overall_result
    
    def _validate_statistics(self, df: pd.DataFrame, result: ValidationResult):
        """Validate dataset statistics"""
        
        if 'transcript' not in df.columns:
            return
        
        # Word count statistics
        word_counts = df['transcript'].str.split().str.len()
        
        stats = {
            'mean_words': word_counts.mean(),
            'median_words': word_counts.median(),
            'std_words': word_counts.std(),
            'min_words': word_counts.min(),
            'max_words': word_counts.max(),
            'total_samples': len(df),
            'null_transcripts': df['transcript'].isnull().sum(),
            'empty_transcripts': (df['transcript'].str.strip() == '').sum()
        }
        
        result.metadata['statistics'] = stats
        
        # Statistical warnings
        if stats['null_transcripts'] > 0:
            result.add_warning(f"{stats['null_transcripts']} null transcripts found")
        
        if stats['empty_transcripts'] > 0:
            result.add_warning(f"{stats['empty_transcripts']} empty transcripts found")
        
        if stats['std_words'] > stats['mean_words']:
            result.add_warning("High variance in transcript lengths")
    
    def validate_file(self, file_path: Union[str, Path]) -> ValidationResult:
        """
        Validate a data file.
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            ValidationResult
        """
        return self.schema_validator.validate_csv_file(file_path)
    
    def create_validation_report(self, result: ValidationResult, output_path: Optional[Path] = None) -> str:
        """
        Create a detailed validation report.
        
        Args:
            result: ValidationResult to report on
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report = f"""
Data Validation Report
=====================

Status: {'PASSED' if result.is_valid else 'FAILED'}
Errors: {len(result.errors)}
Warnings: {len(result.warnings)}

"""
        
        if result.errors:
            report += "ERRORS:\n"
            for i, error in enumerate(result.errors, 1):
                report += f"{i:2d}. {error}\n"
            report += "\n"
        
        if result.warnings:
            report += "WARNINGS:\n"
            for i, warning in enumerate(result.warnings, 1):
                report += f"{i:2d}. {warning}\n"
            report += "\n"
        
        # Add metadata summary
        if 'statistics' in result.metadata:
            stats = result.metadata['statistics']
            report += f"""
Dataset Statistics:
- Total samples: {stats['total_samples']:,}
- Mean words per transcript: {stats['mean_words']:.1f}
- Median words per transcript: {stats['median_words']:.1f}
- Word count range: {stats['min_words']}-{stats['max_words']}
- Null transcripts: {stats['null_transcripts']}
- Empty transcripts: {stats['empty_transcripts']}

"""
        
        if 'content_validation' in result.metadata:
            content = result.metadata['content_validation']
            report += f"""
Content Validation:
- Samples validated: {content['samples_validated']:,}
- Error rate: {content['error_rate']:.1%}
- Valid rate: {content['valid_rate']:.1%}

"""
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Validation report saved to {output_path}")
        
        return report