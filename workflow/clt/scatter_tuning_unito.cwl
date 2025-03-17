#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: Workflow
requirements:
    ScatterFeatureRequirement: {}
    StepInputExpressionRequirement: {}
    SubworkflowFeatureRequirement: {}
inputs:
    preprocessed_dataset: Directory
    learning_rate: File #Valore: Costante, Esponenziale decrescente, Lambda function
    optimizer: File  #valore: SGD, Adam
    batch_size: int[]
    metrics: File  #Validation loss, Precision, F1-score
    augmentation: File  #Horizontal flip, Flip, Random brightness, Rotation90
    l2_egulator: float
    epochs: int
outputs:
    autotuned_models:
        type: Directory
        outputSource: scatter_models/autotuned_models
steps:
    #get_array:
    scatter_models:
        run: train.cwl
        in:
            preprocessed_dataset: preprocessed_dataset
            learning_rate: learning_rate
            optimizer: optimizer
            batch_size: batch_size
            metrics: metrics  #Validation loss, Precision, F1-score
            augmentation: augmentation
            l2_egulator:  l2_egulator
            epochs: epochs
        out: [autotuned_models]
    #training: