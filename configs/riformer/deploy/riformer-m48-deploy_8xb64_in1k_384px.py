_base_ = '../riformer-m48_8xb64_in1k_384px.py'

model = dict(backbone=dict(deploy=True))
