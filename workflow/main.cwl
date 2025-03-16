#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: Workflow
requirements:
    SubworkflowFeatureRequirement: {}
inputs:
    dataset: Directory
outputs:
    result: 
        type: Directory
        outputSource: private_workflow/result
steps:
    preprocessing_azure:
        run: clt/preprocess.cwl
        in:
            dataset: dataset  #not sure it is correct
        out: [preprocessed_dataset]
    autotuning_unito:
        run: clt/scatter_tuning_unito.cwl
        in:
            dataset: dataset
                outputSource: preprocessing_azure/preprocessed_dataset
            learning_rate: File #Valore: Costante, Esponenziale decrescente, Lambda function
            optimizer: File: #valore: SGD, Adam
            batch_size: int []
            metrics: File  #Validation loss, Precision, F1-score
            augmentation: File   #Horizontal flip, Flip, Random brightness, Rotation90
            l2_egulator: float[]
            epochs: int[]
        out:
            autotuned_models:
                type: Directory[]
                outputSource: autotuning_unito/autotuned_models

    private_workflow:
        run:
            class: Workflow
            requirements: 
                StepInputExpressionRequirement: {}
            inputs:
                 dataset: Directory
            outputs:
                result:
                    type: Directory []
                    outputSource: post_processing/ #missing the last data name
            steps:
                data_preprocessing:
                    run: clt/preprocess_private.cwl
                    in:
                        dataset_private: dataset_private    #not sure it is correct
                    out: [preprocessed_private_dataset]
                scatter_fine_tuning:
                    run: scatter_tuning.cwl
                    in:
                    out:

                model_selection:
                post_processing:
              