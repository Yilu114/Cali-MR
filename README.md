# Cali-MR

We use three public real-world datasets (Coat, Yahoo! R3 and KuaiRec) for real-world experiments.
## Run the code

- For Coat, please run the file:


```python
coat.ipynb
```


- For Yahoo! R3, please run the file:

```python
yahoo.ipynb
```


- For KuaiRec, please run the file:


```python
kuairec.ipynb 
```

## Advanced hyperparameter settings

- If you want to change the parameters J, K, and M in the experiment, please import the Cali_MR_Plus.py into the coat.ipynb/yahoo.ipynb/kuairec.ipynb files for the experiment.For example：
  
```python
......
from Cali_MR_Plus import MF_Cali_MR

......
mf_cali_mr = MF_Cali_MR(num_user, num_item, J=3, K=3, M=10)
......

```

## Environment
The code runs well at python 3.8.18. The required packages are as follows:
-   pytorch == 1.9.0
-   numpy == 1.24.4 
-   scipy == 1.10.1
-   pandas == 2.0.3
-   scikit-learn == 1.3.2

