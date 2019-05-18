a = [128, 64, 32, 16]
for m, (i, j) in enumerate(zip(a, a)):
  if m==0:
    a[0], a[0]
  elif m%2==0 and m!=0:
    a[m-1], a[m-1]
  else:
    a[m-1], a[m]


"""
num_layers: 4
hidden: 16
Block(
  (conv1): DenseSAGEConv(16, 16)
  (conv2): DenseSAGEConv(16, 16)
  (lin): Linear(in_features=32, out_features=16, bias=True)
)
Block(
  (conv1): DenseSAGEConv(16, 16)
  (conv2): DenseSAGEConv(16, 2)
  (lin): Linear(in_features=18, out_features=2, bias=True)
)
"""