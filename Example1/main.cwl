#!/usr/bin/env cwl-runner
cwlVersion: v1.2

class: Workflow

#requirements:
  #  InlineJavascriptRequirement {}

inputs:
    message: string

outputs: []

steps:
    step1:
        run: step1.cwl
        in: 
            message: message
        out: []