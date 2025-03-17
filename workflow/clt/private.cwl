#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: Workflow
requirements: 
    StepInputExpressionRequirement: {}
inputs:
    dataset_private: Directory
    autotuned_models:
        source: autotuning_unito/autotuned_models
outputs:
    autotuned_models_private:
        type: Directory
        outputSource: fine_tuning/autotuned_models_private #missing the last data name
steps:
    data_preprocessing:
        run: preprocess_private.cwl
        in:
            dataset_private: dataset_private    #not sure it is correct
        out: [preprocessed_private_dataset]
    fine_tuning: 
        run: scatter_tuning.cwl
        in: 
            preprocessed_private_dataset: 
                source: data_preprocessing/preprocessed_private_dataset
            autotuned_models: autotuned_models                
        out: [autotuned_models_private]
    #model_selection:
   
              