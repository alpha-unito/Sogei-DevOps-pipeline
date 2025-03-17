#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: CommandLineTool
baseCommand: ["python", ""] #path to the training model
arguments:
  - position: 3
    prefix: --name
    valueFrom: '$(inputs.model_type)$(inputs.model_layers)_lr$(inputs.learning_rate)_step$(inputs.lr_step_size)_wd$(inputs.weight_decay)_epochs$(inputs.epochs)' # training model to change
  - position: 4
    prefix: --outdir  
    valueFrom: 'training_$(inputs.model_type)$(inputs.model_layers)_lr$(inputs.learning_rate)_step$(inputs.lr_step_size)_wd$(inputs.weight_decay)_epochs$(inputs.epochs)' 
inputs:
    preprocessed_private_dataset:
        type: Directory
        inputBinding:
            position: 1
            prefix: --preprocessed_private_dataset
    autotuned_models:
        type: Directory
        #source: scatter_models/autotuned_models
        inputBinding:
            position: 2
            prefix: --autotuned-models
outputs:
    autotuned_models_private:
        type: Directory
        outputBinding:
            glob: "training_*/autotuned_models_private/*"
