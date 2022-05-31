# Quantized Adversarial_Patch_Attack
Pytorch implementation of Adversarial Patch (arXiv: https://arxiv.org/abs/1712.09665)

Forked from https://github.com/zhaojb17/Adversarial_Patch_Attack

We adapted adversarial patches[1] to attack the quantized ai85cdnet on the MAX78000. We generated adversarial patches against the cats_vs_dogs model provided with the SDK as well as our humans vs robots model we retrained for GTC. We acheived a 70% attack success rate for the cats_vs_dogs model and 45% success rate for our robots/humans model. These models are 98% and 90% accurate, respectively when no adversarial patch is introduced.

## To run

Copy the python scripts to the ai8x_training folder and the train_attack shell scripts to the scripts directory and make a folder in the ai8x_training folder called training_pictures . To run the cats_vs_dogs attack on the quantized catsvsdogs model,
source scripts/train_attackcatsdogs-q.sh 

The generated patches will be saved to training_pictures.

#Adversarial Patches


The inputs to the model are 128x128 images and we chose a 40x40 pixel sized patch, 10% of the image area. We started with a randomly generated patch image. For each image the patch was placed in a random location and orientation to make the patch more translatable to the physical world[1]. Invariance to scaling would also make the patch more translatable to the physical world[1]. The original code in [6] did not implement optimization for scaling, and neither did we. After applying the patch we update the patch in the direction of the gradient at the output to minimize the error for the chosen target output.

When attempting to generate the patch on the quantized model, we encountered vanishing gradients at the output layer. We find that for most samples in the dataset that the output was almost always saturated and clipped and that the gradient is equal to zero making it difficult to optimize the patch.

Initially the patches trained against the quantized model performed poorly with a 10-15% attack success rate.

We attempted to generate adversarial patches using the full-precision model and then transfer the adversarial patch to the quantized model. For a 40x40 pixel patch we achieved a 70% attack success rate against the full-precision cats_vs_dogs model, but when transfered to the quantized model the success rate fell to 10% which was a similar rate for a patch optimized on the quantized model.


We revisited training on the quantized models directly. Because of quantization, when the gradient is small it gets rounded off and does not contribute any perturbation to the input image. Applying a simple sign function to the gradient similar to the FGSM and forcing the perturbations to {-1,1} improved the attack success rate to 45% for humans/robots and 70% for cats_vs_dogs.

The success rate quickly plateaued and stopped improving. To avoid getting stuck at a local maxima[4] we adapted the momentum iterative fast gradient sign method (MI-FGSM), algorithm 1 from [5] to the quantized model with alpha = 1 and decay factor mu = 0.9. We did not observe any improvement using the MI_FGSM.


## Results

Below are the generated patches for cats_vs_dogs 
<img src="https://github.com/emaclean-adi/Adversarial_Patch_Attack/blob/master/cats_vs_dogs_patch.png" width = 30% height = 30% div align=center />

When the cats_vs_dogs patch is placed in an image with a cat, the image is misclassified as a dog 70% of the time.

For humans vs robots:

<img src="https://github.com/emaclean-adi/Adversarial_Patch_Attack/blob/master/robot_patch.png" width = 30% height = 30% div align=center />



When the robot patch is placed in an image with a human, it is misclassified as a robot 45% of the time.

## References:
[1] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer [Adversarial Patch. arXiv:1712.09665](https://arxiv.org/abs/1712.09665)

[2] https://www.cesar-conference.org/wp-content/uploads/2019/10/s5_p3_21_1430.pdf

[3] https://arxiv.org/pdf/2105.00227.pdf

[4] https://arxiv.org/abs/1710.06081v3

[5] https://arxiv.org/pdf/1706.02379.pdf

[6] https://github.com/zhaojb17/Adversarial_Patch_Attack
