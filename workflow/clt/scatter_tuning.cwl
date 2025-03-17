#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: Workflow
requirements:
    ScatterFeatureRequirement: {}
    StepInputExpressionRequirement: {}
    SubworkflowFeatureRequirement: {}
inputs:
    preprocessed_private_dataset: Directory
    autotuned_models: autotuned_models
       # source: autotuning_unito/autotuned_models
outputs:
    autotuned_models_private:
        type: Directory
steps:
    train_models:
        run: train_private.cwl
        in:
            preprocessed_private_dataset: preprocessed_private_dataset
            autotuned_models: autotuned_models
        out: [autotuned_models_private]
