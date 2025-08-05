sequenceDiagram
    participant User
    participant Orchestrator as FinancialLLMOrchestrator
    participant Config as ConfigManager
    participant Data as DataProcessor
    participant Model as ModelAdapter
    participant Trainer as LLMTrainer
    participant Evaluator as LLMEvaluator
    participant Memory as MemoryManager
    
    User->>Orchestrator: run_complete_pipeline()
    
    rect rgb(240, 248, 255)
        Note over Orchestrator, Memory: Stage 1: Initialization & Configuration
        Orchestrator->>Config: load_config(environment)
        Config-->>Orchestrator: validated_config
        Orchestrator->>Memory: initialize_memory_manager()
        Memory-->>Orchestrator: memory_stats
    end
    
    rect rgb(240, 255, 240)
        Note over Orchestrator, Memory: Stage 2: Data Processing
        Orchestrator->>Data: load_and_validate_data()
        Data->>Data: validate_schema()
        Data->>Data: preprocess_transcripts()
        Data->>Data: convert_to_instructions()
        Data-->>Orchestrator: processed_datasets
    end
    
    rect rgb(255, 248, 240)
        Note over Orchestrator, Memory: Stage 3: Model Preparation
        Orchestrator->>Model: load_base_model()
        Model->>Model: setup_quantization()
        Model->>Model: setup_lora_adaptation()
        Model->>Memory: optimize_memory()
        Memory-->>Model: optimization_stats
        Model-->>Orchestrator: adapted_model
    end
    
    rect rgb(248, 240, 255)
        Note over Orchestrator, Memory: Stage 4: Training
        Orchestrator->>Trainer: setup_training()
        Trainer->>Model: get_model_for_training()
        Model-->>Trainer: training_model
        Trainer->>Trainer: configure_callbacks()
        
        loop Training Loop
            Trainer->>Memory: monitor_memory()
            Trainer->>Trainer: training_step()
            Memory->>Memory: cleanup_if_needed()
        end
        
        Trainer-->>Orchestrator: training_results
    end
    
    rect rgb(255, 240, 248)
        Note over Orchestrator, Memory: Stage 5: Evaluation
        Orchestrator->>Evaluator: evaluate_baseline()
        Evaluator->>Model: baseline_inference()
        Model-->>Evaluator: baseline_predictions
        
        Orchestrator->>Evaluator: evaluate_finetuned()
        Evaluator->>Model: finetuned_inference()
        Model-->>Evaluator: finetuned_predictions
        
        Evaluator->>Evaluator: compare_models()
        Evaluator->>Evaluator: statistical_analysis()
        Evaluator-->>Orchestrator: evaluation_results
    end
    
    rect rgb(240, 255, 255)
        Note over Orchestrator, Memory: Stage 6: Export & Cleanup
        Orchestrator->>Model: save_adapter()
        Orchestrator->>Model: merge_and_save_model()
        Model-->>Orchestrator: export_paths
        
        Orchestrator->>Memory: final_cleanup()
        Memory-->>Orchestrator: cleanup_stats
    end
    
    Orchestrator-->>User: pipeline_results