#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: Workflow
requirements:
    ScatterFeatureRequirement: {}
    StepInputExpressionRequirement: {}
    SubworkflowFeatureRequirement: {}
inputs:
    preprocessed-datasets: Directory
    models: Directory  #not sure

outputs:
    model_training_results:
    type: Directory
steps:
    scatter_models:
    training:
