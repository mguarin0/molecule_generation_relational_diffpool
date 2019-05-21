def _load_dataset(self):
  if self.dataset=="qm9":
    self.data = SparseMolecularDataset()
    self.data.load(join(self.paths["DATA_DIR"], "gdb9_9nodes.sparsedataset"))

mols, _, _, a, x, _, _, _, _, z = self.exper_config.data.next_train_batch(self.exper_config.batch_size,
