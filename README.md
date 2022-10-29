# Pytorch Implementation of RiemannianFlow (learning stable Symmetric Positive Definite (SPD) and Unit Quaternion (UQ) data)
## Based on Stable Dynamic Flows (ImitationFlows)
This original library provides the models and the learning algorithms for learning deep stochastic stable dynamics by Invertible Flows.
The original model is composed by a latent stochastic stable dynamic system and an invertible flow. See [1](https://arxiv.org/abs/2010.13129) and [origin](https://github.com/TheCamusean/iflow)

We propose a geometry-aware approach to learn data on the manifold such as SPD or UQ data.

The models and the learning algorithms are implemented in PyTorch.

## Installation
Inside the repository,

```
pip install -r requirements.txt
```

```
pip install -e .
```

## Examples
### New created Dataset
2 artificially made dataset based on LASA dataset:'LASA_HandWriting_SPD' and 'LASA_HandWriting_SPD' in [`data`](./data)
### Example of manually made dataset
Corresponding examples are also placed in the [`examples`](./examples) directory.
Take [`train_SPD.py`](./examples/train_SPD.py) as an example:
```
python examples/train_SPD.py --depth 11 --lr 0.01
```
you can set part of the hyperparameters by the above command.(for more information check the script)

If you uncomment the lines of 118 amd 119, you will finally get the following figure:
Generated and demonstration trajectories on the tagent space:
![image](https://github.com/WeitaoZC/iflow/blob/main/results/trajectories/Sine_SPD.svg)

Stream lines for the whole related space:
![image](https://github.com/WeitaoZC/iflow/blob/main/results/stream3d/100dpi/Sine_SPD.pdf)

There are also model saving commands in the script, check it to change its file name or its directory.

We also provide the scripts to handel the output from well-trained model, including selecting points from the generated data corresponding to the demonstration; transferring data on the tangent space to their original maniflods; comparing the prediction and demonstration on the manifold and so on.

Check [`SPD_comp.py`](./examples/SPD_comp.py) and [`UQ_comp.py`](./examples/UQ_comp.py) for more details.

### Train your own dataset
If you want to train your own data, you need to create a new data file like [`robot_uq_dataset.py`](./ifloe/dataset/robot_uq_dataset.py), and modify the new file to fit your desire.

Then create a new training file or modify [`train_uq.py`](./examples/train_SPD.py) file to load your own data, and adjust the parameters of the model to train it.

Take our real robot experiment as an example, check [`from_scratch.ipynb`](./examples/from_scratch.ipynb)

## Basics

Stable Dynamic Flows, named ImitationFlows in [1], represents a family of neural network architectures, 
which combines a latent stable dynamic system and an invertible neural network(Normalizing Flows).

You can find the set of stable dynamic system models in  [`dynamics`](./iflow/model/dynamics).

For the invertible networks, we have used RealNVP layer [3], Neural Spline Flows [4] and 
Neural ODE [5]. You can find our models in [`flows`](./iflow/model/flows).


### References
[1] Julen Urain, Michele Ginesi, Davide Tateo, Jan Peters. "ImitationFlows: Learning Deep Stable Stochastic 
Dynamic Systems by Normalizing Flows" *IEEE/RSJ International Conference on
Intelligent Robots and Systems.* 2020.[https://arxiv.org/abs/2010.13129](https://arxiv.org/abs/2010.13129)

[2] Khansari-Zadeh, S. Mohammad, and Aude Billard. "Learning stable nonlinear dynamical 
systems with gaussian mixture models." *IEEE Transactions on Robotics* 2011.

[3] Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio. 
"Density estimation using real nvp." *International Conference on Learning Representations* 2016.

[4] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. 
"Neural spline flows." *Advances in Neural Information Processing Systems* 2019.

[5] Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. 
"Neural ordinary differential equations". *In Advances in neural information processing systems* 2018.

---

If you found this library useful in your research, please consider citing
```
@ARTICLE{wang2022Learning,  
    author={Wang, Weitao and Saveriano, Matteo and Abu-Dakka, Fares J.},
    journal={IEEE Access},   title={Learning Deep Robotic Skills on Riemannian manifolds},
    year={2022},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/ACCESS.2022.3217800}
}
```

_Our Flows library has been highly influenced by the amazing repositories_
 
 https://github.com/bayesiains/nsf
 https://github.com/rtqichen/torchdiffeq
