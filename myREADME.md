#### We use TSGBench to test COSCI-GAN, TimeGAN, and TimeVAE repositories
#### They need different conda env to run

# TimeGAN
- It requries TF=1.15.0
- So we use Python 3.7

```
conda create --name timeGAN python=3.7

conda activate timeGAN

pip install -r requirements.txt
```

```
python3 main_timegan.py --data_name stock --seq_len 24 --module gru --hidden_dim 24 --num_layer 3 --iteration 50000 --batch_size 128 --metric_iteration 10
```