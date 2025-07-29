# FT HMC

Try to find a better design of the neural network for the field transformation.

### Methods

- PyTorch Lightning Fabric enables multi-GPU training with larger batch sizes by automatically handling data parallelism and gradient synchronization. However, it's not needed for evaluation since inference is simpler.

- PyTorch 2.0's inductor backend optimizes model execution through just-in-time compilation, providing significant speedup for evaluation. Training doesn't benefit due to the dynamic nature of gradient computation. So I use the inductor backend for evaluation, and the eager backend for training.

- Promising neural network techniques explored:
    - Residual blocks: Skip connections that help train deeper networks by allowing gradients to flow more easily
    - Channel attention: Mechanism to adaptively recalibrate channel-wise feature responses by explicitly modeling interdependencies between channels
    - Weight normalization: Reparameterization of weights that accelerates convergence by decoupling the magnitude of weights from their direction
    - ReZero: Initialize residual branches with zero weights to stabilize training of very deep networks
    - CoordConv: Adds coordinate information to convolutions to help networks better understand spatial relationships

### Results
- Models trained on smaller beta values generalize effectively to larger beta values, demonstrating good transfer learning capabilities
- Models trained on smaller lattice sizes scale successfully to larger lattices, achieving comparable performance to models directly trained on larger sizes


### Find the best model:
- Explored 15 model variants (v1-v15) in [cnn_models.py](utils/cnn_models.py) by systematically combining different neural network techniques
- Selected the 5 most promising architectures and consolidated them in [best_model.py](utils/best_model.py)
- Initial evaluation: Trained all models on L=32 lattice at β=2.0 for 64 epochs to compare training loss
- Further evaluation: Trained the three best models (simple, rsat, stable, lite) on L=32 lattice at β=6.0 for 64 epochs
- Final testing: Evaluated models trained at β=2.0 and β=6.0 on larger L=64 lattice at β=7.0 to assess generalization capabilities

- Now I prefer the model "lite"


### Tuning hyperparams
- Testing training with and without identity initialization showed similar loss values, however identity initialization led to more stable training behavior;
- Tried to tune the hyperparams: learning rate, weight decay, and initial standard deviation of the normal distribution, training loss is not sensitive to these hyperparams;


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

- don't do drop out, cause it will break the inversion of the FT
[dropout](ft_train_test/logs/train_L32_b2.0-b2.0_test_localv2_batch32_dropout.log)

Note: train on which L is denoted in the save tag.


### Running
- can localv2 give a better performance on fthmc compared to localv1?
- can localv2_batch32 reach a loss smaller than 16.94? if so, how about the performance on L64 b7?
- can localv9 give a better performance on fthmc compared to localv1?


### Questions
- Can this trained FT be effective even when the regular HMC is freezed?
- Is the rect term useful?
- Will batch size affect the behavior of FT?
- How to apply to SU(3)?
- What if you continue to train on a specific beta, instead of move to the next beta?