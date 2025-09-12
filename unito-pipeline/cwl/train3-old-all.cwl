cwlVersion: v1.2
class: CommandLineTool

baseCommand: [python3, example-epoch-batch.py]
arguments:
  - position: 3
    valueFrom: $(runtime.outdir)
    prefix: --output_dir
  - prefix: --end_epoch
    valueFrom: $(inputs.end_epoch)
  - prefix: --batch_size
    valueFrom: $(inputs.batch_size)
#   - prefix: --output_dir
#     valueFrom: $(inputs.output_dir)

requirements:
  InitialWorkDirRequirement:
    listing:
      - entryname: example-epoch-batch.py
        entry: $(inputs.train_script)

inputs:
  train_script:
    type: File
    #ho tolto inputBinding: {}  This is what was causing cwltool to pass the filename as a positional argument, which argparse didn’t expect.

  end_epoch:
    type: int
    inputBinding: 
        prefix: --end_epoch
 
  batch_size:
    type: int
    inputBinding: 
        prefix: --batch_size

#   output_dir:
#     type: string
#     outputBinding:
#         glob: $(runtime.outdir)

outputs:
  output_training:
    type: Directory
    outputBinding:
      glob: epoch$(inputs.end_epoch)*
  log_metrics: 
    type: File
    outputBinding:
      glob: "metrics_epoch$(inputs.end_epoch)_batch$(inputs.batch_size).csv"

  log_output:
    type: stdout
stdout: train_log1.txt
