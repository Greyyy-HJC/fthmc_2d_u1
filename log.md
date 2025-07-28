# FT HMC

Try to find a better design of the neural network for the field transformation.

### Methods

- fabric can help to train with multiple GPUs, enable larger batch size, but not for evaluation
- inductor backend of compile can help to accelerate the evaluation, but not for training
- possibly useful techniques:
    - residual block
    - channel attention
    - weight normalization
    - ReZero
    - coordconv

### Results
- train on small beta can be applied to larger beta
- train on small size can be applied to larger size, and the performance is similar to the model that trained on larger size


### Test Training on different models:
- v1(baseline)
- v2(residual block + channel attention)



### Tests 
- train on beta 4 can be applied to beta 6 and beta 8
- train on L=64 can be applied to L=128, and the performance is quite good
- when the training loss does not decay from train beta 3 to train beta 4, the behavior of models are similar
- at beta 8, the regular HMC is almost freezed, but the stable model trained on beta 4 can still give some improvement
- choose the model called "rsat", tune hyperparams to get the best performance

- seems stable is the best model
[best_simple_L64_b7](evaluation/plots/comparison_fthmc_L64_beta7.0_train_beta4.0_ftstep0.05_simple_tuned_best.pdf)
[best_rsat_L64_b7](evaluation/plots/comparison_fthmc_L64_beta7.0_train_beta4.0_ftstep0.05_rsat_tuned_best.pdf)
[best_stable_L64_b7](evaluation/plots/comparison_fthmc_L64_beta7.0_train_beta2.0_ftstep0.05_stable_L32_tuned_no_init.pdf)

- train on L=32 can be applied to L=64, and the performance is almost the same as the model that trained on L=64
[train_on_L64](evaluation/plots/comparison_fthmc_L64_beta6.0_train_beta4.0_ftstep0.05_stable.pdf)
[train_on_L32](evaluation_test/plots/comparison_fthmc_L64_beta6.0_train_beta4.0_ftstep0.05_rsat_L32_lr0.001_wd0.0001_init0.001.pdf)

- rsat has some improvement compared to simple, while stable seems to be better than rsat
[train_on_b2_simple](ft_train_tune/plots/cnn_loss_L32_train_beta2.0_simple_L32_tuned_no_init.pdf)
[train_on_b2_rsat](ft_train_tune/plots/cnn_loss_L32_train_beta2.0_rsat_L32_tuned_no_init.pdf)
[train_on_b2_stable](ft_train_tune/plots/cnn_loss_L32_train_beta2.0_stable_L32_tuned_no_init.pdf)

- training on L32 has similar behavior as training on L64, but the performance is slightly worse
[train_on_L64](ft_train/plots/cnn_loss_L64_train_beta2.0_rsat_L64_tuned.pdf)
[train_on_L32](ft_train/plots/cnn_loss_L32_train_beta2.0_rsat_L32_tuned_base.pdf)

- with identity init, the performance is better
[train_on_b2_simple_with_init](ft_train_tune/plots/cnn_loss_L32_train_beta2.0_simple_L32_tuned_with_init.pdf)
[train_on_b2_rsat_with_init](ft_train_tune/plots/cnn_loss_L32_train_beta2.0_rsat_L32_tuned_with_init.pdf)
[train_on_b2_stable_with_init](ft_train_tune/plots/cnn_loss_L32_train_beta2.0_stable_L32_tuned_with_init.pdf)
[train_on_b2_simple_no_init](ft_train_tune/plots/cnn_loss_L32_train_beta2.0_simple_L32_tuned_no_init.pdf)
[train_on_b2_rsat_no_init](ft_train_tune/plots/cnn_loss_L32_train_beta2.0_rsat_L32_tuned_no_init.pdf)
[train_on_b2_stable_no_init](ft_train_tune/plots/cnn_loss_L32_train_beta2.0_stable_L32_tuned_no_init.pdf)

- keep training on larger beta and tunning hyperparams are useful
[train_b2_simple_L64](evaluation/plots/comparison_fthmc_L64_beta7.0_train_beta2.0_ftstep0.05_simple_L32_tuned_no_init.pdf)
[best_simple_L64_b7](evaluation/plots/comparison_fthmc_L64_beta7.0_train_beta4.0_ftstep0.05_simple_tuned_best.pdf)

- Add kernel size from 3 to 4 makes it more difficult to bring down the loss

- batch size 32 seems give better performance than batch size 64

- with identity init, the performance is better

- loss on L32 is somehow not totally consistent with the performance on L64 b7

- add conv layer may not be a good idea, it can cause overfitting

Note: train on which L is denoted in the save tag.


### Running
- can localv2_batch32 reach a loss smaller than 16.94? if so, how about the performance on L64 b7?

### Questions
- Can this trained FT be effective even when the regular HMC is freezed?
- Is the rect term useful?
- Will batch size affect the behavior of FT?
- How to apply to SU(3)?
- What if you continue to train on a specific beta, instead of move to the next beta?