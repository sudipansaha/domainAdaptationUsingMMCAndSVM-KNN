# Domain adaptation using Maximum Margin Clustering (MMC) and SVM-KNN
## Implementation of a part of the paper (see citation details below)

For detailed algorithm, please refer to the paper. The implementation here shows here 3 steps in Figure 1 of paper (MMC using iterative SVR; Identification of
overalapping samples from MMC result; SVM-KNN based reestimation of cluster labels for such samples).

The main part of the code is in mmcRetrainKnn.m <br/>
Check demo.m for a demonstration of the usage.<br/>
data.mat contains sample data after Geodesic Flow Kernel based subspace projection

### Citation
If you find this code useful, please consider citing:
```[bibtex]
@inproceedings{saha2016unsupervised,
  title={Unsupervised domain adaptation without source domain training samples: a maximum margin clustering based approach},
  author={Saha, Sudipan and Banerjee, Biplab and Merchant, Shabbir N},
  booktitle={Proceedings of the Tenth Indian Conference on Computer Vision, Graphics and Image Processing},
  pages={1--8},
  year={2016}
}
```
