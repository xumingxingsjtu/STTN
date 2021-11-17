## STTN: Spatial temporal transformer networks for traffic flow forecasting
The official code for "Spatial temporal transformer networks for traffic flow forecasting"
## Introduction
Traffic forecasting has emerged as a core component of intelligent transportation systems.
However, timely accurate traffic forecasting, especially long-term forecasting, still remains an open challenge due to the highly nonlinear and dynamic spatial-temporal dependencies of traffic flows. 
In this paper, we propose a novel paradigm of Spatial-Temporal Transformer Networks (STTNs) that leverages dynamical directed spatial dependencies and long-range temporal dependencies to improve the accuracy of long-term traffic forecasting. 
Specifically, we present a new variant of graph neural networks, named spatial transformer, by dynamically modeling directed spatial dependencies with self-attention mechanism to capture realtime traffic conditions as well as the directionality of traffic flows.
Furthermore, different spatial dependency patterns can be jointly modeled with multi-heads attention mechanism to consider diverse relationships related to different factors (e.g. similarity, connectivity and covariance). 
On the other hand, the temporal transformer is utilized to model long-range bidirectional temporal dependencies across multiple time steps. Finally, they are composed as a block to jointly model the spatial-temporal dependencies for accurate traffic prediction.
Compared to existing works, the proposed model enables fast and scalable training over a long range spatial-temporal dependencies. Experimental results demonstrate that the proposed model achieves competitive results compared with the state-of-the-arts, especially in forecasting long-term traffic flows on real-world PeMS-Bay and PeMSD7(M) datasets.

## Prerequisites
Our code is based on Python3.6, a few depended libraries as as follows:
1. Tensorflow>=1.4.0
2. NumPy (>= 1.15)
3. SciPy (>= 1.1.0)
4. Pandas (>= 0.24)

## Dataset
We adopted the same dataset as "Spatio-Temporal Graph Convolutional Networks: 
A Deep Learning Framework for Traffic Forecasting" and "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting". Please refer to [STGCN](https://github.com/VeritasYin/STGCN_IJCAI-18) for 
the description and preprocessing of the dataset [PeMSD7](https://pems.dot.ca.gov) and [DCRNN](https://github.com/liyaguang/DCRNN) for that of the dataset [PeMS-bay](https://github.com/liyaguang/DCRNN).

## Citation
If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:

    @article{xu2020spatial,title={Spatial-temporal transformer networks for traffic flow forecasting},
    author={Xu, Mingxing and Dai, Wenrui and Liu, Chunmiao and Gao, Xing and Lin, Weiyao and Qi, Guo-Jun and Xiong, Hongkai},
    journal={arXiv preprint arXiv:2001.02908},
    year={2020}
    }   
