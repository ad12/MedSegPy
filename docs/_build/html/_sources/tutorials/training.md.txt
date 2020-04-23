# Training

From the previous tutorials, you may now have a custom model and data loader.

MedSegPy supports the Adam optimizer both with and without gradient accumulation. You are free to create your own optimizer, and write the training logic.

We also provide a "trainer" abstraction with
[`DefaultTrainer`](../modules/engine.html#medsegpy.engine.trainer.DefaultTrainer)
that helps simplify the standard types of training.


### Logging of Metrics
By default the training/validation losses and learning rate are logged to
Tensorboard. Other logging coming soon!
