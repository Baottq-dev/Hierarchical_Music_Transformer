# 🏗️ System Architecture Report - AMT Project

## Overview
This report details the overall system architecture of the AMT (Audio Music Transformer) project, providing a comprehensive view of how all components work together to create a text-to-music generation system.

## 🎯 System Objectives
- Create a modular, scalable text-to-music generation system
- Implement end-to-end pipeline from text input to MIDI output
- Provide flexible configuration and extensible architecture
- Ensure high performance and maintainability

## 🏗️ High-Level Architecture

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Text Input    │───▶│  AMT Pipeline   │───▶│  MIDI Output    │
│                 │    │                 │    │                 │
│ "A happy jazz   │    │ 6-Stage Process │    │ Generated.mid   │
│  piece"         │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Pipeline Stages
1. **Data Collection** → MIDI metadata + Wikipedia text
2. **Data Processing** → BERT embeddings + clustering
3. **Data Preparation** → Training data creation
4. **Model Training** → GPT-2 fine-tuning
5. **Music Generation** → Text-to-MIDI conversion
6. **Evaluation** → Quality assessment

## 🔧 Component Architecture

### 1. Data Collection Layer
```
┌─────────────────────────────────────────────────────────────┐
│                    Data Collection                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ MIDI Metadata   │    │ Wikipedia       │                │
│  │ Extractor       │    │ Collector       │                │
│  │                 │    │                 │                │
│  │ • File scanning │    │ • API queries   │                │
│  │ • Metadata      │    │ • Rate limiting │                │
│  │   extraction    │    │ • Text pairing  │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### 2. Data Processing Layer
```
┌─────────────────────────────────────────────────────────────┐
│                   Data Processing                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ MIDI Processor  │    │ Text Processor  │                │
│  │                 │    │                 │                │
│  │ • Event         │    │ • BERT          │                │
│  │   conversion    │    │   embeddings    │                │
│  │ • Sequence      │    │ • Text cleaning │                │
│  │   generation    │    │ • Feature       │                │
│  └─────────────────┘    │   extraction    │                │
│                         └─────────────────┘                │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Clustering      │    │ Data Prep       │                │
│  │ Engine          │    │ Engine          │                │
│  │                 │    │                 │                │
│  │ • K-means       │    │ • Training data │                │
│  │ • Semantic      │    │   creation      │                │
│  │   tokens        │    │ • Sequence      │                │
│  └─────────────────┘    │   combination   │                │
│                         └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### 3. Model Layer
```
┌─────────────────────────────────────────────────────────────┐
│                      Model Layer                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Training        │    │ Generation      │                │
│  │ Engine          │    │ Engine          │                │
│  │                 │    │                 │                │
│  │ • GPT-2         │    │ • Text-to-token │                │
│  │   fine-tuning   │    │   mapping       │                │
│  │ • Loss          │    │ • Sequence      │                │
│  │   optimization  │    │   generation    │                │
│  │ • Checkpoint    │    │ • MIDI          │                │
│  │   management    │    │   conversion    │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### 4. Evaluation Layer
```
┌─────────────────────────────────────────────────────────────┐
│                   Evaluation Layer                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Metrics         │    │ Quality         │                │
│  │ Calculator      │    │ Assessor        │                │
│  │                 │    │                 │                │
│  │ • Note density  │    │ • Score         │                │
│  │ • Velocity      │    │   aggregation   │                │
│  │ • Note range    │    │ • Threshold     │                │
│  │ • Time sig      │    │   checking      │                │
│  │ • Tempo         │    │ • Report        │                │
│  └─────────────────┘    │   generation    │                │
│                         └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow Architecture

### Complete Data Flow
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   MIDI      │───▶│  Metadata   │───▶│  Wikipedia  │───▶│  Paired     │
│  Files      │    │  Extraction │    │  Collection │    │  Data       │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Training   │◀───│  Training   │◀───│  Clustered  │◀───│  BERT       │
│  Data       │    │  Data Prep  │    │  Data       │    │  Embeddings │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Trained    │───▶│  Text       │───▶│  Generated  │───▶│  Evaluation │
│  Model      │    │  Input      │    │  MIDI       │    │  Results    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Data Transformation Pipeline
1. **Raw MIDI** → **Metadata** → **Paired Data**
2. **Paired Data** → **BERT Embeddings** → **Clustered Data**
3. **Clustered Data** → **Training Data** → **Trained Model**
4. **Text Input** → **Semantic Token** → **Generated MIDI**

## 🏗️ Module Architecture

### Directory Structure
```
AMT/
├── source/                          # Core modules
│   ├── data_collection/             # Data collection modules
│   │   ├── midi_metadata.py         # MIDI file processing
│   │   ├── wikipedia_collector.py   # Text collection
│   │   └── __init__.py             # Package initialization
│   ├── data_processing/             # Data processing modules
│   │   ├── midi_processor.py        # MIDI event conversion
│   │   ├── text_processor.py        # Text embedding
│   │   ├── process_data.py          # Main processing pipeline
│   │   ├── prepare_training_data.py # Training data prep
│   │   └── __init__.py             # Package initialization
│   ├── model/                       # Model modules
│   │   ├── training.py              # Model training
│   │   ├── generation.py            # Music generation
│   │   ├── clustering.py            # Semantic clustering
│   │   └── __init__.py             # Package initialization
│   ├── evaluation/                  # Evaluation modules
│   │   ├── metrics.py               # Evaluation metrics
│   │   └── __init__.py             # Package initialization
│   ├── utils/                       # Utility modules
│   │   ├── data_preparation.py      # Data preparation utilities
│   │   ├── environment.py           # Environment verification
│   │   └── __init__.py             # Package initialization
│   └── config.py                    # Configuration management
├── main.py                          # Main pipeline controller
├── collect_data.py                  # Data collection script
└── requirements.txt                 # Dependencies
```

### Module Dependencies
```
main.py
├── collect_data.py
│   ├── source/data_collection/midi_metadata.py
│   └── source/data_collection/wikipedia_collector.py
├── source/data_processing/process_data.py
│   ├── source/data_processing/text_processor.py
│   └── source/model/clustering.py
├── source/data_processing/prepare_training_data.py
│   ├── source/data_processing/midi_processor.py
│   └── source/utils/data_preparation.py
├── source/model/training.py
├── source/model/generation.py
└── source/evaluation/metrics.py
```

## 🔧 Configuration Architecture

### Configuration Management
```python
# Centralized configuration in config.py
CONFIG = {
    "data": {
        "midi": MIDI_CONFIG,
        "text": TEXT_CONFIG
    },
    "model": MODEL_CONFIG,
    "training": TRAINING_CONFIG,
    "generation": GENERATION_CONFIG,
    "evaluation": EVALUATION_CONFIG
}
```

### Environment Configuration
- **Development**: Local file paths, debug logging
- **Testing**: Test data paths, minimal processing
- **Production**: Optimized paths, production logging

## 🔄 Pipeline Architecture

### Main Pipeline Controller
```python
# main.py - Unified pipeline control
PIPELINE = [
    ('collect', COLLECT_CMD),
    ('process', PROCESS_CMD),
    ('prepare', PREPARE_CMD),
    ('train', TRAIN_CMD),
    ('generate', GENERATE_CMD),
    ('evaluate', EVALUATE_CMD),
]
```

### Pipeline Execution Flow
1. **Command Parsing**: Parse user arguments
2. **Step Selection**: Determine which steps to run
3. **Dependency Check**: Verify prerequisites
4. **Execution**: Run selected steps sequentially
5. **Error Handling**: Handle failures gracefully
6. **Reporting**: Provide execution summary

## 🎯 Interface Architecture

### Command Line Interface
```bash
# Main pipeline control
python main.py [step] [options]

# Individual step execution
python main.py collect
python main.py process prepare train
python main.py generate --text "A happy jazz piece"
```

### Programmatic Interface
```python
# Direct module usage
from source.data_collection import midi_metadata
from source.model.generation import AMTGenerator

# Use modules directly
metadata = midi_metadata.list_midi_files_and_metadata(...)
generator = AMTGenerator(model_path)
music = generator.generate_music("A happy jazz piece")
```

## 🔒 Error Handling Architecture

### Error Handling Strategy
1. **Input Validation**: Validate all inputs before processing
2. **Graceful Degradation**: Continue processing when possible
3. **Error Logging**: Comprehensive error logging
4. **Recovery Mechanisms**: Automatic retry and fallback
5. **User Feedback**: Clear error messages to users

### Error Types and Handling
- **File Not Found**: Skip and log, continue processing
- **API Errors**: Retry with exponential backoff
- **Memory Errors**: Reduce batch size, retry
- **Model Errors**: Load fallback model or skip generation

## 📊 Performance Architecture

### Performance Optimization
1. **Batch Processing**: Process multiple items together
2. **Parallel Processing**: Use multiple CPU cores
3. **Memory Management**: Efficient memory usage
4. **Caching**: Cache expensive computations
5. **Lazy Loading**: Load data only when needed

### Scalability Considerations
- **Horizontal Scaling**: Process multiple datasets
- **Vertical Scaling**: Use more powerful hardware
- **Distributed Processing**: Split work across machines
- **Cloud Deployment**: Deploy on cloud infrastructure

## 🔧 Deployment Architecture

### Local Deployment
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User          │───▶│  Local          │───▶│  File System    │
│   Interface     │    │  Python         │    │  Output         │
│                 │    │  Environment    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Cloud Deployment
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web           │───▶│  Cloud          │───▶│  Cloud          │
│   Interface     │    │  API            │    │  Gateway        │
│                 │    │  Gateway        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                       ┌─────────────────┐
                       │  Containerized  │
                       │  AMT Pipeline   │
                       └─────────────────┘
```

## 🔄 Integration Architecture

### External Integrations
- **Wikipedia API**: Text description collection
- **BERT Model**: Text embedding generation
- **GPT-2 Model**: Music sequence generation
- **MIDI Libraries**: MIDI file processing

### Internal Integrations
- **Data Flow**: Seamless data transfer between modules
- **Configuration**: Centralized configuration management
- **Logging**: Unified logging across all modules
- **Error Handling**: Consistent error handling strategy

## 📈 Monitoring Architecture

### System Monitoring
- **Performance Metrics**: Processing time, memory usage
- **Quality Metrics**: Generation quality scores
- **Error Tracking**: Error rates and types
- **Resource Usage**: CPU, memory, disk usage

### Health Checks
- **Module Health**: Verify each module is working
- **Data Health**: Check data quality and completeness
- **Model Health**: Verify model performance
- **Pipeline Health**: Check end-to-end functionality

## 🔧 Security Architecture

### Security Considerations
- **Input Validation**: Validate all user inputs
- **File Security**: Secure file handling
- **API Security**: Secure API communications
- **Data Privacy**: Protect user data

### Best Practices
- **Principle of Least Privilege**: Minimal required permissions
- **Input Sanitization**: Clean all inputs
- **Error Handling**: Don't expose sensitive information
- **Logging**: Secure logging practices

## 📈 Future Architecture

### Planned Enhancements
1. **Microservices**: Split into independent services
2. **API Gateway**: Centralized API management
3. **Message Queues**: Asynchronous processing
4. **Distributed Storage**: Scalable data storage

### Scalability Roadmap
- **Phase 1**: Optimize current architecture
- **Phase 2**: Add microservices
- **Phase 3**: Implement distributed processing
- **Phase 4**: Add real-time capabilities

## 📝 Conclusion

The AMT system architecture provides a robust, modular, and scalable foundation for text-to-music generation. The design supports both development and production use cases while maintaining flexibility for future enhancements.

### Key Architectural Strengths
- ✅ Modular design with clear separation of concerns
- ✅ Scalable pipeline architecture
- ✅ Comprehensive error handling
- ✅ Flexible configuration management
- ✅ Extensible module system

### Next Steps
1. Implement microservices architecture
2. Add distributed processing capabilities
3. Enhance monitoring and observability
4. Improve security and privacy features 