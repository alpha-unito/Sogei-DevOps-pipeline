cwlVersion: v1.2
class: CommandLineTool

baseCommand: [python3, calculating_best1.py]
arguments:
  - position: 1
    valueFrom: $(runtime.outdir)
    prefix: --output_dir


requirements:
  InitialWorkDirRequirement:
    listing:
      - entryname: calculating_best1.py
        entry: $(inputs.choose_best_script)
      - entry: $(inputs.metrics_file)
      - entry: $(inputs.training_directory)

inputs:
  choose_best_script:
    type: File
    #ho tolto inputBinding: {}  This is what was causing cwltool to pass the filename as a positional argument, which argparse didn’t expect.
  
  metrics_file:
    type: File[]
  training_directory:
    type: Directory[]


outputs:
  best_model:
    type: Directory
    outputBinding:
        glob: "Final_models"
        
  log_output:
    type: stdout
stdout: train_log1.txt