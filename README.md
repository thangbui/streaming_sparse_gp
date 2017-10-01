*** Implementation of Streaming sparse Gaussian process approximations

This code was tested using GPflow 0.4.0 and tensorflow 1.2 on a Linux machine and a Mac. Note that latest GPflow might not be backward-compatible.

We also provide an implementation of the batch Power-EP sparse approximation of Bui, Yan and Turner (2017).


References: 

@inproceedings{BuiNguTur17,
  title =  {Streaming sparse {G}aussian process approximations},
  author =   {Bui, Thang D. and Nguyen, Cuong V. and Turner, Richard E.},
  booktitle = {Advances in Neural Information Processing Systems 30},
  year =   {2017}
}

@article{BuiYanTur16,
  title={A Unifying Framework for Sparse {G}aussian Process Approximation using {P}ower {E}xpectation {P}ropagation},
  author={Thang D. Bui and Josiah Yan and Richard E. Turner},
  journal={arXiv preprint arXiv:1605.07066},
  year={2016}
}