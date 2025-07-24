# FT HMC

Try to find a better design of the neural network for the field transformation.

### Methods

- fabric can help to train with multiple GPUs, enable larger batch size, but not for evaluation
- inductor backend of compile can help to accelerate the evaluation, but not for training
- resnet and attention are both useful


### Tests 
- train on beta 4 can be applied to beta 6 and beta 8
- train on L=64 can be applied to L=128, and the performance is quite good
- when the training loss does not decay from train beta 3 to train beta 4, the behavior of models are similar
- at beta 8, the regular HMC is almost freezed, but the stable model trained on beta 4 can still give some improvement


### Running
- test train of stablev2, v3 and v4, first beta 2, then up to beta 4
- train the stable on beta 4.5 and beta 5
- evaluate the stable train b4 on L=128, note the L128 model is copied from L64 model 
