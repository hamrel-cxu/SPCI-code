# SPCI code
 Official implementation of the work [Sequential Predictive Conformal Inference for Time Series](https://openreview.net/forum?id=jJeY7w8YRz) (ICML 2023). [Slide](https://bpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/9/2055/files/2023/06/SPCI_slide.pdf) and [Poster](https://bpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/9/2055/files/2023/09/SPCI_poster.pdf) are also available.
 
 Please direct questions regarding implementation to cxu310@gatech.edu.
 
 See [tutorial_electric_EnbPI_SPCI.ipynb](https://github.com/hamrel-cxu/SPCI-code/blob/main/tutorial_electric_EnbPI_SPCI.ipynb) for comparing SPCI against [EnbPI](https://ieeexplore.ieee.org/abstract/document/10121511), which is an earlier method of ours. We demonstrate significant reduction in interval width on the electric dataset, which is also used in [Nex-CP](https://arxiv.org/abs/2202.13415) (Barber et al., 2022).

 Installation of dependency:

 ```
pip install -r requirements.txt
```
 
 If you find our work useful, please consider citing it.
 ```
@InProceedings{xu2023SPCI,
  title = 	 {Sequential Predictive Conformal Inference for Time Series},
  author =       {Xu, Chen and Xie, Yao},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {38707--38727},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/xu23r/xu23r.pdf},
  url = 	 {https://proceedings.mlr.press/v202/xu23r.html}
}
 ```
