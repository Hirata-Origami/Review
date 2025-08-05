graph TB
    subgraph "🎯 Orchestrator Layer"
        O[FinancialLLMOrchestrator<br/>Main Pipeline Coordinator]
    end
    
    subgraph "⚙️ Core Services Layer"
        CM[ConfigManager<br/>Environment-aware Config]
        LM[LoggerManager<br/>Structured Logging]
        MM[MemoryManager<br/>Resource Optimization]
        V[DataValidator<br/>Quality Assurance]
        SA[StatisticalAnalyzer<br/>Significance Testing]
    end
    
    subgraph "🔧 Business Logic Layer"
        DP[FinancialDataProcessor<br/>Data Pipeline]
        MA[LlamaFinancialAdapter<br/>Model Management]
        T[FinancialLLMTrainer<br/>Training Framework]
        E[FinancialLLMEvaluator<br/>Evaluation System]
    end
    
    subgraph "📊 Data Processing"
        IT1[FinancialUnderstanding<br/>Template]
        IT2[TranscriptionCompletion<br/>Template]
        IT3[ContentSummarization<br/>Template]
        IT4[QAGeneration<br/>Template]
    end
    
    subgraph "🤖 Model Components"
        BM[Base Model<br/>Llama-3-8B]
        LA[LoRA Adapter<br/>Memory Efficient]
        QC[Quantization<br/>4-bit/8-bit]
        UO[Unsloth<br/>Optimization]
    end
    
    subgraph "📈 Evaluation Metrics"
        RM[ROUGE Metrics]
        BM2[BLEU Score]
        BS[BERTScore]
        SS[Semantic Similarity]
        FM[Financial Metrics]
    end
    
    subgraph "🛠 Infrastructure"
        GPU[GPU Management]
        CPU[CPU Optimization]
        STOR[Storage Systems]
        MON[Monitoring Tools]
    end
    
    O --> CM
    O --> LM
    O --> MM
    O --> V
    O --> DP
    O --> MA
    O --> T
    O --> E
    
    DP --> IT1
    DP --> IT2
    DP --> IT3
    DP --> IT4
    DP --> V
    
    MA --> BM
    MA --> LA
    MA --> QC
    MA --> UO
    MA --> MM
    
    T --> MA
    T --> MM
    T --> LM
    
    E --> RM
    E --> BM2
    E --> BS
    E --> SS
    E --> FM
    E --> SA
    
    MM --> GPU
    MM --> CPU
    LM --> STOR
    LM --> MON
    
    style O fill:#ff9999,stroke:#333,stroke-width:3px
    style CM fill:#99ccff,stroke:#333,stroke-width:2px
    style LM fill:#99ccff,stroke:#333,stroke-width:2px
    style MM fill:#99ccff,stroke:#333,stroke-width:2px
    style V fill:#99ccff,stroke:#333,stroke-width:2px
    style SA fill:#99ccff,stroke:#333,stroke-width:2px
    style DP fill:#99ff99,stroke:#333,stroke-width:2px
    style MA fill:#99ff99,stroke:#333,stroke-width:2px
    style T fill:#99ff99,stroke:#333,stroke-width:2px
    style E fill:#99ff99,stroke:#333,stroke-width:2px