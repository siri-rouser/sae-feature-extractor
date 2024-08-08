# SAE feature extractor
This repo is based on sae-stage-tempelate:https://github.com/starwit/sae-stage-template.git that can be directly insert into the SAE engine pipeline.
The network currently refers to the feature extracting network in : https://github.com/coder-wangzhen/AIC22-MCVT
there are three models to be chosen:
  - resnet101_ibn_a_2
  - resnet101_ibn_a_3
  - resnext101_ibn_a_2

The visionapi module for this stage is also changes with adding message feature into message detections, the modified version of feature will upload to the umberrla YQ pipeline later then.
