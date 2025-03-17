#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: CommandLineTool
baseCommand: ["python", ""] #path to the training model
arguments:
  - position: 10
    prefix: --name
    valueFrom: '$(inputs.model_type)$(inputs.model_layers)_lr$(inputs.learning_rate)_step$(inputs.lr_step_size)_wd$(inputs.weight_decay)_epochs$(inputs.epochs)' # training model to change
  - position: 11
    prefix: --outdir  
    valueFrom: 'training_$(inputs.model_type)$(inputs.model_layers)_lr$(inputs.learning_rate)_step$(inputs.lr_step_size)_wd$(inputs.weight_decay)_epochs$(inputs.epochs)' 
inputs:
    preprocessed_dataset:
        type: Directory
        inputBinding:
            position: 1
            prefix: --preprocessed_dataset
    epochs:
        type: int
        inputBinding:
            position: 3
            prefix: --epochs
    learning_rate:
        type: float
        inputBinding:
            position: 4
            prefix: --learning-rate
    optimizer:
        type: File
        inputBinding:
            position: 5
            prefix: --optimizer
    batch_size:
        type: int
        inputBinding:
            position: 6
            prefix: --batch-size
    metrics:
        type: File
        inputBinding:
            position: 7
            prefix: --metrics
    augmentation:
        type: File
        inputBinding:
            position: 8
            prefix: --augmentation
    l2_egulator:
        type: float
        inputBinding:
            position: 9
            prefix: --l2-egulator
outputs:
  autotuned_models:
    type: Directory
    outputBinding:
      glob: "training_*/autotuned_models/*"
