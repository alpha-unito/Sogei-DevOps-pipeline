#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: CommandLineTool
baseCommand: ["python", "path"] #path to the script file
inputs:
  dataset:
    type: Directory
    inputBinding:  #binding if necessary
      position: 1
outputs:
  preprocessed_dataset:
    type: Directory
    outputBinding:
      glob: "$(runtime.outdir)"