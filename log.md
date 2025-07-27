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

- choose the best model called "rsat", tune hyperparams to get the best performance

- train on L=32 can be applied to L=64, and the performance is almost the same as the model that trained on L=64
[train_on_L64](evaluation/plots/comparison_fthmc_L64_beta6.0_train_beta4.0_ftstep0.05_stable.pdf)
[train_on_L32](evaluation_test/plots/comparison_fthmc_L64_beta6.0_train_beta4.0_ftstep0.05_rsat_L32_lr0.001_wd0.0001_init0.001.pdf)
- rsat has some improvement compared to simple
- training on L32 has similar behavior as training on L64, but the performance is slightly worse
- seems the first few betas are useful in the training, while the last few betas are not

Note: train on which L is denoted in the save tag.


### Running
- train the rsat model on L32 for round1, then train on L64 for round2


### Questions
- Can this trained FT be effective even when the regular HMC is freezed?
- Is the rect term useful?
- Will batch size affect the behavior of FT?
- How to apply to SU(3)?
- What if you continue to train on a specific beta, instead of move to the next beta?