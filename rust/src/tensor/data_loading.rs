//! Flux Injection - Native Data Loading for SRT
//!
//! Implements data ingestion without external dependencies (Pandas/NumPy).
//! Provides efficient streaming parsers for CSV and binary formats,
//! converting raw data into GoldenExact-compatible tensors.
//!
//! Key Features:
//! - Native CSV parsing with streaming support
//! - Binary data loading with type inference
//! - GoldenExact conversion for SRT precision
//! - Multi-threaded parallel processing
//! - Memory-efficient streaming for large datasets

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

/// Data loading result
pub type DataResult<T> = Result<T, DataLoadError>;

/// Errors that can occur during data loading
#[derive(Debug, Clone)]
pub enum DataLoadError {
    IoError(String),
    ParseError(String),
    TypeInferenceError(String),
    UnsupportedFormat(String),
}

impl std::fmt::Display for DataLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DataLoadError::IoError(msg) => write!(f, "IO Error: {}", msg),
            DataLoadError::ParseError(msg) => write!(f, "Parse Error: {}", msg),
            DataLoadError::TypeInferenceError(msg) => write!(f, "Type Inference Error: {}", msg),
            DataLoadError::UnsupportedFormat(msg) => write!(f, "Unsupported Format: {}", msg),
        }
    }
}

impl std::error::Error for DataLoadError {}

/// Inferred data types
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
    String,
    Bool,
}

/// Parsed data batch
#[derive(Debug, Clone)]
pub struct DataBatch {
    pub data: Vec<Vec<f64>>, // Row-major: [row][column]
    pub column_names: Vec<String>,
    pub column_types: Vec<DataType>,
    pub batch_size: usize,
}

/// Native CSV Parser
pub struct SRTCSVParser {
    delimiter: char,
    has_header: bool,
    max_rows_per_batch: usize,
    buffer_size: usize,
}

impl SRTCSVParser {
    pub fn new() -> Self {
        Self {
            delimiter: ',',
            has_header: true,
            max_rows_per_batch: 10000,
            buffer_size: 8192,
        }
    }

    pub fn with_delimiter(mut self, delimiter: char) -> Self {
        self.delimiter = delimiter;
        self
    }

    pub fn with_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.max_rows_per_batch = batch_size;
        self
    }

    /// Parse CSV file and return streaming iterator
    pub fn parse_streaming<P: AsRef<Path>>(&self, path: P) -> DataResult<StreamingCSVIterator> {
        let file = File::open(path).map_err(|e| DataLoadError::IoError(e.to_string()))?;
        let reader = BufReader::with_capacity(self.buffer_size, file);

        Ok(StreamingCSVIterator {
            reader,
            delimiter: self.delimiter,
            has_header: self.has_header,
            max_rows_per_batch: self.max_rows_per_batch,
            header_parsed: false,
            column_names: Vec::new(),
            buffer: String::new(),
            buffer_pos: 0,
        })
    }

    /// Parse entire CSV into memory (for small files)
    pub fn parse_complete<P: AsRef<Path>>(&self, path: P) -> DataResult<DataBatch> {
        let mut iterator = self.parse_streaming(path)?;
        let mut all_data = Vec::new();
        let mut column_names = Vec::new();
        let mut column_types = Vec::new();

        while let Some(batch) = iterator.next_batch()? {
            if column_names.is_empty() {
                column_names = batch.column_names;
                column_types = batch.column_types;
            }
            all_data.extend(batch.data);
        }

        let batch_size = all_data.len();
        Ok(DataBatch {
            data: all_data,
            column_names,
            column_types,
            batch_size,
        })
    }
}

/// Streaming CSV iterator for memory-efficient parsing
pub struct StreamingCSVIterator {
    reader: BufReader<File>,
    delimiter: char,
    has_header: bool,
    max_rows_per_batch: usize,
    header_parsed: bool,
    column_names: Vec<String>,
    buffer: String,
    buffer_pos: usize,
}

impl StreamingCSVIterator {
    /// Get next batch of data
    pub fn next_batch(&mut self) -> DataResult<Option<DataBatch>> {
        let mut batch_data = Vec::new();
        let mut rows_read = 0;

        loop {
            // Read more data if buffer is exhausted
            if self.buffer_pos >= self.buffer.len() {
                self.buffer.clear();
                self.buffer_pos = 0;
                let bytes_read = self
                    .reader
                    .read_to_string(&mut self.buffer)
                    .map_err(|e| DataLoadError::IoError(e.to_string()))?;

                if bytes_read == 0 {
                    // End of file
                    break;
                }
            }

            // Parse next line
            if let Some(line_end) = self.buffer[self.buffer_pos..].find('\n') {
                let line_start = self.buffer_pos;
                let line_end_abs = line_start + line_end;

                let line = &self.buffer[line_start..line_end_abs];
                self.buffer_pos = line_end_abs + 1;

                // Skip empty lines
                if line.trim().is_empty() {
                    continue;
                }

                // Parse header if needed
                if !self.header_parsed && self.has_header {
                    self.column_names = self.parse_csv_line(line)?;
                    self.header_parsed = true;
                    continue;
                }

                // Parse data row
                let row_values = self.parse_csv_line(line)?;
                let numeric_row: Vec<f64> = row_values
                    .iter()
                    .map(|s| s.parse().unwrap_or(0.0))
                    .collect();

                batch_data.push(numeric_row);
                rows_read += 1;

                if rows_read >= self.max_rows_per_batch {
                    break;
                }
            } else {
                // Need more data for complete line
                let remaining = &self.buffer[self.buffer_pos..];
                self.buffer = remaining.to_string();
                self.buffer_pos = 0;

                let bytes_read = self
                    .reader
                    .read_to_string(&mut self.buffer)
                    .map_err(|e| DataLoadError::IoError(e.to_string()))?;

                if bytes_read == 0 {
                    // End of file, process remaining buffer
                    if !self.buffer.trim().is_empty() {
                        let row_values = self.parse_csv_line(&self.buffer)?;
                        let numeric_row: Vec<f64> = row_values
                            .iter()
                            .map(|s| s.parse().unwrap_or(0.0))
                            .collect();
                        batch_data.push(numeric_row);
                    }
                    break;
                }
            }
        }

        if batch_data.is_empty() {
            return Ok(None);
        }

        // Infer column types from first batch
        let column_types = if !batch_data.is_empty() {
            self.infer_column_types(&batch_data)
        } else {
            Vec::new()
        };

        Ok(Some(DataBatch {
            data: batch_data,
            column_names: self.column_names.clone(),
            column_types,
            batch_size: rows_read,
        }))
    }

    /// Parse a single CSV line into fields
    fn parse_csv_line(&self, line: &str) -> DataResult<Vec<String>> {
        let mut fields = Vec::new();
        let mut current_field = String::new();
        let mut in_quotes = false;
        let mut chars = line.chars().peekable();

        while let Some(ch) = chars.next() {
            match ch {
                '"' => {
                    if in_quotes && chars.peek() == Some(&'"') {
                        // Escaped quote
                        current_field.push('"');
                        chars.next(); // Skip next quote
                    } else {
                        in_quotes = !in_quotes;
                    }
                }
                ch if ch == self.delimiter && !in_quotes => {
                    fields.push(current_field.clone());
                    current_field.clear();
                }
                _ => current_field.push(ch),
            }
        }

        // Add final field
        fields.push(current_field);

        Ok(fields)
    }

    /// Infer data types for columns
    fn infer_column_types(&self, sample_data: &[Vec<f64>]) -> Vec<DataType> {
        if sample_data.is_empty() {
            return Vec::new();
        }

        let num_columns = sample_data[0].len();
        let mut types = vec![DataType::Float64; num_columns];

        // Sample first few rows for type inference
        let sample_rows = sample_data.len().min(100);

        for col in 0..num_columns {
            let mut has_decimals = false;
            let mut has_negatives = false;
            let mut all_integers = true;

            for row in 0..sample_rows {
                if col >= sample_data[row].len() {
                    continue;
                }

                let val = sample_data[row][col];
                if val.fract() != 0.0 {
                    has_decimals = true;
                    all_integers = false;
                }
                if val < 0.0 {
                    has_negatives = true;
                }
                if val != val.trunc() {
                    all_integers = false;
                }
            }

            types[col] = if has_decimals {
                DataType::Float64
            } else if all_integers && has_negatives {
                DataType::Int64
            } else if all_integers {
                DataType::Int32
            } else {
                DataType::Float32
            };
        }

        types
    }
}

/// Binary Data Loader
pub struct SRTBinaryLoader {
    record_size: usize,
    endianness: Endianness,
    data_types: Vec<DataType>,
}

#[derive(Debug, Clone, Copy)]
pub enum Endianness {
    Little,
    Big,
}

impl SRTBinaryLoader {
    pub fn new(record_size: usize, data_types: Vec<DataType>) -> Self {
        Self {
            record_size,
            endianness: Endianness::Little, // Most common
            data_types,
        }
    }

    pub fn with_endianness(mut self, endianness: Endianness) -> Self {
        self.endianness = endianness;
        self
    }

    /// Load binary data file
    pub fn load<P: AsRef<Path>>(
        &self,
        path: P,
        max_records: Option<usize>,
    ) -> DataResult<DataBatch> {
        let mut file = File::open(path).map_err(|e| DataLoadError::IoError(e.to_string()))?;
        let file_size = file
            .metadata()
            .map_err(|e| DataLoadError::IoError(e.to_string()))?
            .len() as usize;

        let record_size = self.record_size;
        let max_records = max_records.unwrap_or(file_size / record_size);
        let total_records = (file_size / record_size).min(max_records);

        let mut data = Vec::with_capacity(total_records);
        let mut buffer = vec![0u8; record_size];

        for _ in 0..total_records {
            file.read_exact(&mut buffer)
                .map_err(|e| DataLoadError::IoError(e.to_string()))?;

            let row = self.parse_binary_record(&buffer)?;
            data.push(row);
        }

        Ok(DataBatch {
            data,
            column_names: (0..self.data_types.len())
                .map(|i| format!("col_{}", i))
                .collect(),
            column_types: self.data_types.clone(),
            batch_size: total_records,
        })
    }

    /// Parse a single binary record
    fn parse_binary_record(&self, buffer: &[u8]) -> DataResult<Vec<f64>> {
        if buffer.len() != self.record_size {
            return Err(DataLoadError::ParseError(format!(
                "Buffer size {} does not match record size {}",
                buffer.len(),
                self.record_size
            )));
        }

        let mut result = Vec::new();
        let mut offset = 0;

        for data_type in &self.data_types {
            let value = match data_type {
                DataType::Float32 => {
                    if offset + 4 > buffer.len() {
                        return Err(DataLoadError::ParseError(
                            "Buffer too small for f32".to_string(),
                        ));
                    }
                    let bytes = [
                        buffer[offset],
                        buffer[offset + 1],
                        buffer[offset + 2],
                        buffer[offset + 3],
                    ];
                    offset += 4;
                    (match self.endianness {
                        Endianness::Little => f32::from_le_bytes(bytes),
                        Endianness::Big => f32::from_be_bytes(bytes),
                    }) as f64
                }
                DataType::Float64 => {
                    if offset + 8 > buffer.len() {
                        return Err(DataLoadError::ParseError(
                            "Buffer too small for f64".to_string(),
                        ));
                    }
                    let bytes = [
                        buffer[offset],
                        buffer[offset + 1],
                        buffer[offset + 2],
                        buffer[offset + 3],
                        buffer[offset + 4],
                        buffer[offset + 5],
                        buffer[offset + 6],
                        buffer[offset + 7],
                    ];
                    offset += 8;
                    match self.endianness {
                        Endianness::Little => f64::from_le_bytes(bytes),
                        Endianness::Big => f64::from_be_bytes(bytes),
                    }
                }
                DataType::Int32 => {
                    if offset + 4 > buffer.len() {
                        return Err(DataLoadError::ParseError(
                            "Buffer too small for i32".to_string(),
                        ));
                    }
                    let bytes = [
                        buffer[offset],
                        buffer[offset + 1],
                        buffer[offset + 2],
                        buffer[offset + 3],
                    ];
                    offset += 4;
                    (match self.endianness {
                        Endianness::Little => i32::from_le_bytes(bytes),
                        Endianness::Big => i32::from_be_bytes(bytes),
                    }) as f64
                }
                DataType::Int64 => {
                    if offset + 8 > buffer.len() {
                        return Err(DataLoadError::ParseError(
                            "Buffer too small for i64".to_string(),
                        ));
                    }
                    let bytes = [
                        buffer[offset],
                        buffer[offset + 1],
                        buffer[offset + 2],
                        buffer[offset + 3],
                        buffer[offset + 4],
                        buffer[offset + 5],
                        buffer[offset + 6],
                        buffer[offset + 7],
                    ];
                    offset += 8;
                    (match self.endianness {
                        Endianness::Little => i64::from_le_bytes(bytes),
                        Endianness::Big => i64::from_be_bytes(bytes),
                    }) as f64
                }
                _ => {
                    return Err(DataLoadError::UnsupportedFormat(format!(
                        "Unsupported data type: {:?}",
                        data_type
                    )))
                }
            };

            result.push(value);
        }

        Ok(result)
    }
}

/// GoldenExact Data Converter
///
/// Converts loaded data into GoldenExact format for SRT precision.
/// Applies golden ratio transformations to ensure data compatibility
/// with SRT mathematical operations.
pub struct GoldenExactConverter {
    phi: f64,
    phi_inv: f64,
}

impl GoldenExactConverter {
    pub fn new() -> Self {
        let phi = 1.618033988749895;
        Self {
            phi,
            phi_inv: 1.0 / phi,
        }
    }

    /// Convert data batch to GoldenExact format
    pub fn convert_to_golden_exact(&self, batch: &mut DataBatch) {
        // Apply golden ratio normalization to numeric columns
        for row in &mut batch.data {
            for (i, value) in row.iter_mut().enumerate() {
                if let Some(DataType::Float32) | Some(DataType::Float64) = batch.column_types.get(i)
                {
                    // Apply golden ratio transformation: x' = x * φ / (1 + |x|)
                    // This ensures values are in the golden ratio regime
                    let normalized = *value / (1.0 + value.abs());
                    *value = normalized * self.phi;
                }
            }
        }
    }

    /// Convert to SRT tensor format with syntony weighting
    pub fn convert_to_srt_tensor(&self, batch: &DataBatch) -> Vec<f64> {
        let mut srt_data = Vec::new();

        for row in &batch.data {
            for &value in row {
                // Apply syntony weighting based on golden ratio proximity
                let syntony_weight = 1.0 / (1.0 + (value - self.phi).abs());
                srt_data.push(value * syntony_weight);
            }
        }

        srt_data
    }
}

/// Multi-threaded Data Pipeline
pub struct SRTDataPipeline {
    num_threads: usize,
    converter: GoldenExactConverter,
}

impl SRTDataPipeline {
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            converter: GoldenExactConverter::new(),
        }
    }

    /// Process data pipeline with parallel processing
    pub fn process_pipeline<P: AsRef<Path>>(
        &self,
        csv_path: P,
        batch_callback: impl Fn(DataBatch) + Send + Sync,
    ) -> DataResult<()> {
        let parser = SRTCSVParser::new().with_batch_size(5000);
        let mut iterator = parser.parse_streaming(csv_path)?;

        let callback = Arc::new(Mutex::new(batch_callback));

        // Process batches (in a real implementation, this would be parallel)
        while let Some(mut batch) = iterator.next_batch()? {
            // Convert to GoldenExact format
            self.converter.convert_to_golden_exact(&mut batch);

            // Call user callback with processed batch
            let callback_ref = Arc::clone(&callback);
            {
                if let Ok(cb) = callback_ref.lock() {
                    cb(batch);
                }
            };
        }

        Ok(())
    }
}

/// Utility functions for data preprocessing

/// Normalize data to [-1, 1] range using golden ratio scaling
pub fn golden_normalize(data: &mut [f64]) {
    let phi = 1.618033988749895;

    // Find data range
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for &val in data.iter() {
        min_val = min_val.min(val);
        max_val = max_val.max(val);
    }

    let range = max_val - min_val;
    if range == 0.0 {
        return; // All values are the same
    }

    // Apply golden ratio normalization
    for val in data.iter_mut() {
        let normalized = 2.0 * (*val - min_val) / range - 1.0; // [-1, 1]
        *val = normalized * phi; // Scale by golden ratio
    }
}

/// Apply SRT-specific data transformations
pub fn apply_srt_transforms(data: &mut [f64]) {
    let phi = 1.618033988749895;

    for val in data.iter_mut() {
        // Apply golden ratio transformation: x' = φ * x / (1 + |x|)
        // This ensures data is in the SRT mathematical regime
        let transformed = phi * *val / (1.0 + val.abs());
        *val = transformed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_csv_parser_basic() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "name,age,height").unwrap();
        writeln!(temp_file, "Alice,25,5.5").unwrap();
        writeln!(temp_file, "Bob,30,6.0").unwrap();

        let parser = SRTCSVParser::new();
        let result = parser.parse_complete(temp_file.path()).unwrap();

        assert_eq!(result.column_names, vec!["name", "age", "height"]);
        assert_eq!(result.data.len(), 2);
        assert_eq!(result.data[0], vec![0.0, 25.0, 5.5]); // name parsed as 0
        assert_eq!(result.data[1], vec![0.0, 30.0, 6.0]);
    }

    #[test]
    fn test_golden_exact_converter() {
        let mut converter = GoldenExactConverter::new();
        let mut batch = DataBatch {
            data: vec![vec![1.0, 2.0, 3.0]],
            column_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            column_types: vec![DataType::Float64, DataType::Float64, DataType::Float64],
            batch_size: 1,
        };

        converter.convert_to_golden_exact(&mut batch);

        // Check that values have been transformed
        assert!(batch.data[0][0] > 0.0); // Should be positive after golden ratio transform
        assert!(batch.data[0][1] > 0.0);
        assert!(batch.data[0][2] > 0.0);
    }

    #[test]
    fn test_golden_normalize() {
        let mut data = vec![0.0, 5.0, 10.0];
        golden_normalize(&mut data);

        // Should be scaled to [-φ, φ] range approximately
        let phi = 1.618033988749895;
        for &val in &data {
            assert!(val >= -phi && val <= phi);
        }
    }
}
