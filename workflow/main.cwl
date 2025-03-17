#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: Workflow
requirements:
    SubworkflowFeatureRequirement: {}
inputs:
    dataset: Directory
    dataset_private: Directory
    learning_rate: File #Valore: Costante, Esponenziale decrescente, Lambda function
    optimizer: File  #valore: SGD, Adam
    batch_size: int[]
    metrics: File  #Validation loss, Precision, F1-score
    augmentation: File   #Horizontal flip, Flip, Random brightness, Rotation90
    l2_egulator: float[]
    epochs: int[]
outputs:
    postprocessed_dataset: 
        type: Directory
        outputSource: post_processing_step/postrocessed_private_dataset
steps:
    preprocessing_azure:
        run: clt/preprocess.cwl
        in:
            dataset: dataset  #not sure it is correct
        out: [preprocessed_dataset]
    autotuning_unito:
        run: clt/scatter_tuning_unito.cwl
        in:
            preprocessed_dataset: 
                source: preprocessing_azure/preprocessed_dataset
            learning_rate: learning_rate
            optimizer: optimizer
            batch_size: batch_size
            metrics: metrics  #Validation loss, Precision, F1-score
            augmentation: augmentation
            l2_egulator:  l2_egulator
            epochs: epochs
           
        out:
            autotuned_models:
                type: Directory[]
                outputSource: autotuning_unito/autotuned_models
        scatter: batch_size
    private_workflow:
        run: clt/private.cwl
        in:
            dataset_private: dataset_private
            autotuned_models:
                source: autotuning_unito/autotuned_models
        out: [autotuned_models_private]
    post_processing_step:
        run: clt/post_process.cwl
        in:
            autotuned_models_private: 
                source: private_workflow/autotuned_models_private  #not sure it is correct
        out: [postrocessed_private_dataset]