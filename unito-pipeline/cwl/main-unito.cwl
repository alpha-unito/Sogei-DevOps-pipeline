cwlVersion: v1.2
class: Workflow

requirements:
    ScatterFeatureRequirement: {}

inputs:
  train_script: File
  choose_best_script: File  
  end_epochs: 
    type: int[]
  batch_sizes:
    type: int[]

outputs:
  training_logs:
    type: File[]
    outputSource: train_step/log_output
  final_models:
    type: Directory[]
    outputSource: train_step/output_training
  metricses:
    type: File[]
    outputSource: train_step/log_metrics
  best_model_final:
    type: Directory
    outputSource: best_models/best_model

steps:
  train_step:
    run: train3-old-all.cwl
    in:
      train_script: train_script
      end_epoch: end_epochs
      batch_size: batch_sizes
      #output_dir: output_dir
    out: [log_output,log_metrics, output_training]         
    scatter: [end_epoch, batch_size]
    scatterMethod:  dotproduct     #flat_crossproduct mi fa 4 step-training, dotproduct - mi fa due step 

  best_models:  #second step
    run: choosing-best.cwl
    in: 
        metrics_file: train_step/log_metrics
        training_directory: train_step/output_training
        choose_best_script: choose_best_script
    out: [log_output, best_model]